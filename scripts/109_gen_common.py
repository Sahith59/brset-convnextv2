"""Shared pieces of the stage-2 GENERATIVE renderer: structure map, UNet, sampler.

WHAT THIS REPLACES. Stage 2 used to be a 68,346-parameter refiner whose output
was clamped to +-0.15 (low frequency) and +-0.02 (high frequency). It could not
leave a small neighbourhood of its input, and the domain probe stayed at AUC
1.0000 for every variant. That was a capacity limit, not a finding about
synthesis. Stage 2 is now a ~105M-parameter conditional diffusion model.

THE IDEA, after Dr. Ye: one retina, two device renderings. Split an image into
  - a device-INVARIANT state  s : where the vessels, lesions and disc are
  - a device RENDERING        d : colour, noise, sharpness, illumination
and learn the renderer p(image | s, d) from real images. Translation is then
"read s off a BRSET image, render it with d = mBRSET". The generator is trained
on real target images only as far as appearance goes, so everything it paints is
real mBRSET appearance; the LABEL travels in s, which comes from the source.

WHY A SOFT LOCAL-CONTRAST MAP AND NOT EDGES. ContourDiff (Chen et al., 2024) and
FGDM (Li et al., 2023) condition target-only diffusion on contours / frequency
bands and preserve organ-scale anatomy. A DR label lives in lesions a few pixels
wide; an edge detector drops a 4 px microaneurysm or buries it in noise. The map
here is the signed, locally normalised green-channel contrast, so a dark dot is
a negative blob and an exudate a positive one, both at full resolution.

The map is computed from a DEGRADED copy of the image during training (random
blur, random noise). Without that the map carries the device's own pixel noise
and sharpness and the generator would simply copy them through, leaving the
source device's signature in the output.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

GEN_ROOT = Path("/data/users3/sthummala2/brset_gen")
if str(GEN_ROOT / "pylibs") not in sys.path:
    sys.path.insert(0, str(GEN_ROOT / "pylibs"))
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel  # noqa: E402

DOM_BRSET, DOM_MBRSET, DOM_NULL = 0, 1, 2


# ----------------------------------------------------------------- filters
def gblur(x, sigma):
    """Separable Gaussian, reflect-padded. sigma in pixels."""
    if sigma <= 0:
        return x
    r = max(1, int(round(3.0 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"),
                 k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"),
                 k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x


def fundus_mask(x, thresh=0.04):
    """Field of view. Same threshold as scripts 55, 72 and 84."""
    return (x.mean(dim=1, keepdim=True) > thresh).float()


def erode(mask, px):
    if px <= 0:
        return mask
    return -F.max_pool2d(-mask, 2 * px + 1, stride=1, padding=px)


def structure_map(x, mask=None, sigma_l=10.0, eps=0.03, post_sigma=1.0, rim=4):
    """Signed soft local-contrast map of the green channel, in (-1, 1).

    x: (B,3,H,W) in [0,1]. Local mean and variance are mask-normalised so the
    field-of-view rim does not leak in as a giant edge; `rim` pixels at the
    boundary are zeroed for the same reason. eps keeps flat retina from having
    its sensor noise amplified to full scale.
    """
    if mask is None:
        mask = fundus_mask(x)
    g = x[:, 1:2]
    w = gblur(mask, sigma_l).clamp_min(1e-3)
    mu = gblur(g * mask, sigma_l) / w
    d = (g - mu) * mask
    var = gblur(d * d, sigma_l) / w
    s = torch.tanh(0.5 * d / (var.clamp_min(0).sqrt() + eps))
    return gblur(s, post_sigma) * erode(mask, rim)


def degrade_for_cond(x, max_blur, max_noise, p=0.9):
    """Random blur + noise on the copy the structure map is read from."""
    out = []
    for i in range(x.shape[0]):
        xi = x[i:i + 1]
        if float(torch.rand(())) < p:
            sb = float(torch.rand(())) * max_blur
            if sb > 0.15:
                xi = gblur(xi, sb)
            xi = xi + float(torch.rand(())) * max_noise * torch.randn_like(xi)
        out.append(xi)
    return torch.cat(out).clamp(0, 1)


# ------------------------------------------------------------------- model
def build_unet(base=128, mult=(1, 2, 2, 4, 4), layers=2):
    """Fully convolutional (no attention), so a model trained on 256 px crops
    runs unchanged on the full 560 px frame. Input: noisy RGB (3) + structure
    map (1) + field-of-view mask (1). The device token is a class embedding."""
    n = len(mult)
    return UNet2DModel(
        sample_size=256, in_channels=5, out_channels=3, layers_per_block=layers,
        block_out_channels=tuple(base * m for m in mult),
        down_block_types=("DownBlock2D",) * n, up_block_types=("UpBlock2D",) * n,
        add_attention=False, num_class_embeds=3)


def train_scheduler():
    return DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
                         prediction_type="v_prediction", clip_sample=False)


# ----------------------------------------------------------------- sampler
@torch.no_grad()
def render(unet, s, mask, device_id=DOM_MBRSET, x_init=None, strength=1.0,
           steps=50, w_struct=1.5, w_device=0.0, generator=None,
           amp_dtype=torch.bfloat16):
    """Render structure map `s` as device `device_id`. Returns (B,3,H,W) in [0,1].

    x_init + strength < 1 is the DUAL-DOMAIN coupling: start from the stage-1
    physics output noised to an intermediate level instead of from pure noise.
    Physics has already set the global colour, illumination and noise budget,
    so the diffusion stage only has to supply what physics cannot express, and
    it can do so from a lower noise level -- which is what protects lesions.

    w_struct > 1 is classifier-free guidance on the structure map: an explicit
    dial for how hard the source's lesions are enforced. w_device > 0 adds
    guidance on the device token.
    """
    dev = s.device
    B, _, H, W = s.shape
    sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
                          prediction_type="v_prediction", clip_sample=False,
                          timestep_spacing="trailing")
    sched.set_timesteps(steps, device=dev)
    ts = sched.timesteps
    noise = torch.randn((B, 3, H, W), generator=generator, device=dev)
    if x_init is None or strength >= 1.0:
        x = noise * sched.init_noise_sigma
    else:
        ts = ts[int(round(steps * (1.0 - strength))):]
        x = sched.add_noise(x_init * 2 - 1, noise, ts[:1].expand(B))
    cls = torch.full((B,), device_id, device=dev, dtype=torch.long)
    null = torch.full((B,), DOM_NULL, device=dev, dtype=torch.long)
    zero = torch.zeros_like(s)
    for t in ts:
        xs, ss, cs = [x], [s], [cls]
        if w_struct != 1.0:
            xs.append(x); ss.append(zero); cs.append(cls)
        if w_device > 0:
            xs.append(x); ss.append(s); cs.append(null)
        k = len(xs)
        inp = torch.cat([torch.cat(xs), torch.cat(ss), mask.repeat(k, 1, 1, 1)], dim=1)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=dev.type == "cuda"):
            v = unet(inp, t.expand(B * k), class_labels=torch.cat(cs)).sample
        v = v.float().chunk(k)
        out, j = v[0], 1
        if w_struct != 1.0:
            out = out + (w_struct - 1.0) * (v[0] - v[j]); j += 1
        if w_device > 0:
            out = out + w_device * (v[0] - v[j])
        x = sched.step(out, t, x).prev_sample
    return ((x + 1) / 2).clamp(0, 1) * mask
