"""Bounded residual refiner. Stage two of the dual-domain translator.

Stage one (`scripts/72_physical_layer.py`) is a differentiable acquisition model
with 19 bounded, physically interpretable parameters. As of 9 Oct it has
per-channel gain and offset AND per-channel signal-dependent noise coefficients,
so unlike the earlier 9- and 15-parameter versions it can place colour and noise
in blue without placing them in red. That removes two of the three jobs stage 2
was originally drawn up to do. What it does NOT remove is the reason stage 2
exists, and the argument has to be made against the current stage 1, not the old
one.

What the measurement says (paper/CLAIMS.md C5b, 400 images per dataset,
identical under native and disc-matched measurement so not a scale artifact):

    channel fraction   BRSET    mBRSET   ratio      per-channel std   ratio
    red                0.59797  0.58318   0.98      0.1358 -> 0.1395   1.03
    green              0.29678  0.26595   0.90      0.0978 -> 0.1560   1.60
    blue               0.10525  0.15087   1.43      0.0531 -> 0.1518   2.86

The excess high-frequency energy is in BLUE and GREEN and is absent in RED, and
the chromatic shift is a blue excess with a green deficit. Three residuals are
left after the current stage 1 has done everything its functional form allows,
and they live at opposite ends of the frequency axis, which is why this module
has two branches rather than one.

1. LOW FREQUENCY, CHROMA. Stage 1's colour pipeline is a per-channel affine
   followed by two global scalars, so the chroma correction it applies is the
   SAME everywhere in the frame. The measured blue excess is not spatially
   uniform: it is strongest where illumination is weak, because blue carries the
   least signal and therefore the worst signal-to-noise and the largest share of
   the camera's own gain. What is left over is a smooth, spatially varying,
   channel-specific field. That is a low-frequency object. Band-limiting it by
   construction is what makes it safe: a residual produced at 1/16 resolution
   and bilinearly upsampled CANNOT contain lesion-scale structure, no matter
   what the optimiser wants, which is why it gets the generous bound
   (eps_lf ~ 0.15). The self-test measures this rather than assuming it: the
   largest contrast the saturated branch can inject between two ADJACENT pixels
   is what would erase a lesion, and it comes out near 0.005, well under the
   0.04 lesion floor.

2. HIGH FREQUENCY, NOISE TEXTURE. Stage 1's noise is i.i.d. white per pixel with
   amplitude sqrt(a_c * x + b_c^2), a fixed monotone function of local intensity
   with the same coefficients everywhere. Three things it still cannot produce:
     - SPATIAL CORRELATION. Real small-sensor noise is demosaiced and then JPEG
       compressed, so it is spatially and cross-channel correlated, not white.
       A Gram-matrix style loss sees exactly that correlation structure; i.i.d.
       noise cannot match it at any amplitude.
     - amplitude that varies with anything other than local intensity, e.g. more
       noise in the periphery than the signal-dependence alone accounts for.
     - CHANNEL-SPECIFIC ACUTANCE. sharpen_k is one scalar shared by all three
       channels, so deterministic edge sharpness cannot differ by channel. Red
       at 1.03x and blue at 2.86x cannot both be hit with one k.
   This branch is given a bound of eps_hf = 0.02, which is BELOW the lesion
   contrast floor: microaneurysms and dot haemorrhages sit 0.04 to 0.16 in [0,1]
   below the surrounding retina, so a 0.02 edit cannot fill one in. The bound is
   not caution, it is the whole lesion-preservation guarantee. A translated
   BRSET image keeps its BRSET disease label, so a refiner that erases a
   microaneurysm does not merely look wrong, it silently writes a mislabelled
   positive into the downstream training set.

Why the high-frequency branch defaults to NOISE MODULATION rather than a free
residual. The measured residual is sensor noise: zero-mean and stochastic. A
deterministic residual conditioned on the image could learn "wherever there is a
dark dot, add +0.02", i.e. a systematic lesion-filling policy, and only the
0.02 bound would stand between that policy and the training set. In noise mode
the branch instead controls (a) a per-channel, spatially varying AMPLITUDE and
(b) a 3x3 bias-free mixing filter that gives the noise its spatial and
cross-channel correlation. Because the filter has no bias and tanh is odd, the
expected residual is EXACTLY zero at every pixel, so no systematic lesion edit
is representable at all, bound or no bound. The self-test verifies that
empirically over 256 draws. `--hf_mode residual` and `--hf_mode off` exist for
the ablation, and the paper should report all three.

What is deliberately NOT here. Framing (BRSET mean aspect 1.287 with the disc
clipped top and bottom; mBRSET uniform 1600x1600 with the disc inscribed) is
deterministic geometry and is normalised outside the generator, per decision D1
of report/STAGE2_DESIGN.md. Asking a network to learn it invites it to translate
or warp anatomy, which is the lesion-destroying failure this whole design exists
to prevent.

What is deliberately NOT targeted. The downstream trainer already applies
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1) plus
Resize((560,560)) + RandomCrop(512) + flip + RandomRotation(15)
(scripts/30_train_strong_baseline.py, lines 335-343). Any global brightness,
contrast or saturation shift this module produces is therefore WASHED OUT before
the classifier ever sees it, and optimising for it is wasted capacity.
ColorJitter does not touch hue, so the chromatic content of the residual is the
part that actually survives to the downstream task. The loss weights in
scripts/83_train_translator.py are set from that fact.

Also here: the optional PatchGAN discriminator (off by default in training) and
the Translator wrapper that composes stage 1 and stage 2.
"""
import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location(
    "physical_layer", str(Path(__file__).parent / "72_physical_layer.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

fundus_mask = P.fundus_mask

# Shallowest microaneurysm / dot-haemorrhage contrast against surrounding
# retina, in [0,1] units. Every bound in this file is justified against it, so
# it is named once rather than written as a literal in six places.
LESION_CONTRAST_FLOOR = 0.04


def erode_mask(mask, px):
    """Shrink a binary mask by `px` pixels, via min-pooling.

    Why erode at all. AUDIT 1 in scripts/74_audit_physics.py found that the
    field-of-view RIM, where retina meets black surround, is a hard step that
    contaminates the Laplacian-variance statistic. If the refiner is allowed to
    edit the rim it can move that statistic without touching retina at all,
    which is the cheapest possible way to game the objective and produces
    nothing of downstream value. Keeping the residual off the rim removes that
    shortcut and also avoids ringing at the boundary.

    Note on border semantics, which happen to be the ones we want:
    max_pool2d pads with -inf, so padding never wins the max of -mask and the
    erosion does NOT bite at the image frame edge. BRSET's disc is clipped by
    the frame top and bottom, so part of its boundary is the frame rather than a
    rim step, and there is nothing there to protect. Only true rim boundaries
    are eroded.
    """
    if px <= 0:
        return mask
    k = 2 * int(px) + 1
    return -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=int(px))


def residual_mask(mask, erode, taper):
    """The weight the residual is multiplied by: eroded, then tapered to zero.

    The taper is not cosmetic, it fixes a measured violation of this module's
    central safety property. A hard mask multiplies a residual that may sit at
    its bound by 1 on one pixel and 0 on the next, which is a 0.15 step between
    adjacent pixels. The self-test caught exactly that: the band-limited branch
    claimed a maximum adjacent-pixel step of 2*eps_lf/lf_stride = 0.019 and
    actually produced 0.150 at the mask edge, nearly four times the 0.04 lesion
    contrast floor. A hard edge also dumps high-frequency energy right at the
    rim, which is the statistic-gaming shortcut the erosion exists to close.

    Erode by (erode + taper) and then box-blur by taper. The blur spreads the
    support outward by exactly taper, so the result is still identically zero
    outside erode(mask, erode), the hard guarantee survives, and the weight now
    ramps from 0 to 1 over taper pixels. Its own largest per-pixel step is
    1/(2*taper+1), so the residual's step at the taper is at most
    eps_lf/(2*taper+1). At the defaults that is 0.15/33 = 0.0045, and the total
    adjacent-pixel step the branch can produce anywhere is

        2*eps_lf/lf_stride + eps_lf/(2*taper+1) = 0.01875 + 0.0045 = 0.0233

    which is 58 percent of the 0.04 lesion floor. The self-test asserts it.

    count_include_pad=False so the taper does not bite at the image frame edge,
    matching erode_mask: BRSET's disc is clipped by the frame, and a frame edge
    is not a rim with a step in it.
    """
    m = erode_mask(mask, int(erode) + int(taper))
    if taper <= 0:
        return m
    k = 2 * int(taper) + 1
    return F.avg_pool2d(m, kernel_size=k, stride=1, padding=int(taper),
                        count_include_pad=False)


def masked_channel_mean(x, mask):
    """Per-channel mean inside the field of view, shape (B,C).

    Fed to the low-frequency branch as explicit global context. The chroma
    target is a statement about exactly this quantity, so handing it to the
    network directly is cheaper and more interpretable than making a stack of
    3x3 convolutions rediscover it through its receptive field.
    """
    n = mask.sum(dim=(2, 3)).clamp(min=1.0)
    return (x * mask).sum(dim=(2, 3)) / n


class ResidualRefiner(nn.Module):
    """out = clamp(x_phys + mask * (r_lf + r_hf), 0, 1), with hard bounds on both.

    |r_lf| <= eps_lf and |r_hf| <= eps_hf pointwise, enforced by tanh/sigmoid
    saturation rather than by a penalty, so the guarantee holds at every step of
    training and not merely at convergence. Bounding is load-bearing by
    measurement, not by taste: when stage 1 was left unbounded the optimiser hit
    the target statistics by painting blobs, driving spot and hole amplitude to
    2.98 and 1.95 on a [0,1] image (CLAIMS.md C7). A network with tens of
    thousands of free parameters has strictly more ways to do that.

    Size: roughly 68k parameters at the defaults, and a forward pass costs about
    as much as four full-resolution 3x3 convolutions. This is intentional. The
    low-frequency branch's output is only (3 x H/16 x W/16) numbers, so extra
    capacity there buys nothing but overfitting and lesion-erasure risk, and the
    whole translator has to train in hours on one A40, not days.
    """

    def __init__(self, lf_stride=16, lf_width=48, lf_depth=3, hf_width=16,
                 eps_lf=0.15, eps_hf=0.02, hf_mode="noise", mask_erode=4,
                 mask_taper=16):
        super().__init__()
        assert hf_mode in ("noise", "residual", "off")
        self.lf_stride = int(lf_stride)
        self.eps_lf = float(eps_lf)
        self.eps_hf = float(eps_hf)
        self.hf_mode = hf_mode if self.eps_hf > 0 else "off"
        self.mask_erode = int(mask_erode)
        self.mask_taper = int(mask_taper)

        # LOW-FREQUENCY BRANCH. Operates at 1/lf_stride resolution. Input is the
        # downsampled image (3ch) concatenated with the global per-channel mean
        # broadcast back to a map (3ch); see masked_channel_mean.
        w = int(lf_width)
        self.lf_in = nn.Conv2d(6, w, 3, padding=1)
        self.lf_blocks = nn.ModuleList()
        for _ in range(int(lf_depth)):
            self.lf_blocks.append(nn.Sequential(
                nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.GELU()))
        self.lf_head = nn.Conv2d(w, 3, 1)

        # HIGH-FREQUENCY BRANCH. Full resolution, deliberately shallow: its job
        # is a local per-channel noise amplitude, which needs a receptive field
        # of a few pixels, not a few hundred.
        if self.hf_mode != "off":
            hw = int(hf_width)
            self.hf = nn.Sequential(
                nn.Conv2d(3, hw, 3, padding=1), nn.GELU(),
                nn.Conv2d(hw, hw, 3, padding=1), nn.GELU())
            self.hf_head = nn.Conv2d(hw, 3, 1)
            # Noise colouring filter: 3x3, all channels to all channels, NO
            # BIAS. The spatial extent is what turns white noise into the
            # spatially correlated texture a demosaiced and JPEG-compressed
            # sensor actually produces, and the cross-channel terms are what
            # let blue noise correlate with green (measured 2.86x and 1.60x)
            # while red stays clean (1.03x). Bias-free is not an aesthetic
            # choice: with no bias, mix(n) is an odd function of a symmetric
            # random field, tanh is odd, so E[r_hf] = 0 exactly at every pixel.
            # That is what makes a systematic lesion edit unrepresentable.
            # Only built in noise mode; in residual mode there is no noise to
            # colour and an unused parameter would never receive gradient.
            self.hf_mix = (nn.Conv2d(3, 3, 3, padding=1, bias=False)
                           if self.hf_mode == "noise" else None)
        else:
            self.hf = None
            self.hf_head = None
            self.hf_mix = None

        self._init_near_identity()

    def _init_near_identity(self):
        """Start the refiner at (almost) the identity.

        Stage 1 initialised at its fitted values is the best purely physical
        explanation of the gap. Joint training should begin from that solution
        and improve on it, not begin from a random perturbation of it and have
        to climb back. The heads get a tiny random init rather than an exactly
        zero one: exactly zero makes the output residual zero AND the gradient
        into every preceding layer zero on the first step, so the body sits idle
        for an iteration. A 1e-3 scale head gives a residual of order 1e-4,
        which is identity for all practical purposes, while every parameter
        receives gradient immediately.
        """
        for head in (self.lf_head, self.hf_head):
            if head is None:
                continue
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)
        if self.hf_mode == "noise":
            # amp = eps_hf * sigmoid(.), so a negative bias starts the amplitude
            # near zero while keeping sigmoid off its flat region: the local
            # slope at -4 is 0.018, small but not vanishing.
            nn.init.constant_(self.hf_head.bias, -4.0)
            # Identity init for the colouring filter, so the noise starts white
            # and uncorrelated, i.e. starts where stage 1 already is, and any
            # correlation structure has to be earned from the style loss.
            with torch.no_grad():
                self.hf_mix.weight.zero_()
                for c in range(3):
                    self.hf_mix.weight[c, c, 1, 1] = 1.0
                self.hf_mix.weight += torch.randn_like(self.hf_mix.weight) * 1e-3

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def low_freq_residual(self, x, mask):
        b, _, h, w = x.shape
        hd = max(2, (h + self.lf_stride - 1) // self.lf_stride)
        wd = max(2, (w + self.lf_stride - 1) // self.lf_stride)

        # Area (box) downsampling, which is a proper low-pass before decimation.
        # Dividing by the downsampled mask undoes the darkening that the black
        # surround would otherwise introduce into rim-adjacent low-res pixels,
        # so the branch sees retinal colour rather than a blend with the
        # surround.
        xm = F.interpolate(x * mask, size=(hd, wd), mode="area")
        mm = F.interpolate(mask, size=(hd, wd), mode="area")
        xd = xm / mm.clamp(min=1e-3)

        g = masked_channel_mean(x, mask).view(b, 3, 1, 1).expand(b, 3, hd, wd)
        z = F.gelu(self.lf_in(torch.cat([xd, g], dim=1)))
        for blk in self.lf_blocks:
            z = z + blk(z)
        r = self.eps_lf * torch.tanh(self.lf_head(z))

        # Bound BEFORE upsampling. Bilinear interpolation is a convex
        # combination of its inputs, so the bound survives the upsample, and the
        # result is band-limited to roughly 1/lf_stride of Nyquist by
        # construction. That band limit, not a loss term, is what makes a
        # 0.15-amplitude residual safe next to a 0.04-contrast microaneurysm:
        # the per-pixel STEP it can take is at most about 2*eps_lf/lf_stride.
        return F.interpolate(r, size=(h, w), mode="bilinear", align_corners=False)

    def high_freq_residual(self, x, generator=None):
        if self.hf_mode == "off":
            return torch.zeros_like(x)
        z = self.hf(x)
        if self.hf_mode == "residual":
            return self.eps_hf * torch.tanh(self.hf_head(z))
        # Noise mode. amp is deterministic given the image and lies in
        # [0, eps_hf]; the field it multiplies is zero-mean and bounded. tanh
        # bounds the coloured noise to (-1,1), which a raw Gaussian would not,
        # and tanh of a unit normal has std 0.627, so the realised noise std is
        # about 0.627 * amp.
        amp = self.eps_hf * torch.sigmoid(self.hf_head(z))
        n = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        return amp * torch.tanh(self.hf_mix(n))

    def forward(self, x_phys, mask=None, generator=None, return_parts=False):
        """x_phys in [0,1], shape (B,3,H,W). Returns the refined image.

        `mask` should be the field of view of the ORIGINAL input, not of
        x_phys: stage 1 can dim the image enough to move rim pixels across the
        0.04 threshold, and a mask that drifts with the parameters is a mask the
        optimiser can shrink to make statistics easier.
        """
        if mask is None:
            mask = fundus_mask(x_phys)
        m = residual_mask(mask, self.mask_erode, self.mask_taper)

        r_lf = self.low_freq_residual(x_phys, mask)
        r_hf = self.high_freq_residual(x_phys, generator=generator)
        res = (r_lf + r_hf) * m

        out = (x_phys + res).clamp(0.0, 1.0)
        if not return_parts:
            return out
        return out, {"r_lf": r_lf * m, "r_hf": r_hf * m, "residual": res,
                     "mask": mask, "mask_eff": m}

    def eps_total(self):
        return self.eps_lf + (self.eps_hf if self.hf_mode != "off" else 0.0)

    def max_adjacent_step(self):
        """Analytic bound on the adjacent-pixel contrast the refiner can inject.

        This, not the residual's amplitude, is the lesion-risk metric: a
        microaneurysm is visible because it sits 0.04 to 0.16 BELOW its
        immediate surround, so a residual that shifts a whole neighbourhood
        equally leaves it entirely intact, while one that can take a 0.04 step
        across one pixel can fill it in.

        The deterministic and stochastic parts are reported separately because
        only the deterministic part can erase a lesion. A zero-mean noise field
        uncorrelated with image content lowers the local signal-to-noise ratio,
        which is exactly what the target camera itself does (measured: blue std
        0.0531 -> 0.1518, green 0.0978 -> 0.1560), but it cannot implement a
        policy of filling dark dots. `deterministic_total` is therefore the
        number the lesion-preservation guarantee is stated in, and it is what
        the self-test asserts against the 0.04 floor.

        Consequence worth knowing before running an ablation: in residual mode
        the high-frequency branch IS deterministic, so eps_hf = 0.02 puts the
        deterministic total at 0.0633, above the floor. See
        max_safe_eps_hf_residual.
        """
        lf = 2.0 * self.eps_lf / self.lf_stride
        if self.mask_taper > 0:
            lf += self.eps_lf / (2.0 * self.mask_taper + 1.0)
        hf_det = 2.0 * self.eps_hf if self.hf_mode == "residual" else 0.0
        hf_sto = 2.0 * self.eps_hf if self.hf_mode == "noise" else 0.0
        return {"lf": lf, "hf_deterministic": hf_det, "hf_stochastic": hf_sto,
                "deterministic_total": lf + hf_det, "total": lf + hf_det + hf_sto}

    def max_safe_eps_hf_residual(self, floor=None):
        """Largest eps_hf that keeps the DETERMINISTIC adjacent-pixel edit under
        the lesion contrast floor, given how much of the budget the
        low-frequency branch has already spent. Only binds in residual mode; in
        noise mode the high-frequency term is zero-mean and does not count
        against this budget."""
        floor = LESION_CONTRAST_FLOOR if floor is None else floor
        return max(0.0, (floor - self.max_adjacent_step()["lf"]) / 2.0)


class Translator(nn.Module):
    """Stage 1 then stage 2, trained jointly.

    The DuDoNet argument (Lin et al., CVPR 2019) is that neither domain alone
    suffices and that the link between them must be differentiable so the two
    can be optimised together. Here the acquisition-domain stage carries the
    interpretable, provably photometric part of the transform, and the image
    domain stage carries only the bounded residual that the acquisition model's
    functional form cannot express. Training them separately would let stage 2
    compensate for a badly fitted stage 1, and the division of labour, which is
    the claim the paper makes, would stop being true.
    """

    def __init__(self, phys=None, refiner=None, learn_phys=True):
        super().__init__()
        self.phys = phys if phys is not None else P.PhysicalAcquisition(learnable=learn_phys)
        self.refiner = refiner if refiner is not None else ResidualRefiner()

    def forward(self, x, generator=None, return_parts=False):
        mask = fundus_mask(x)
        x1 = self.phys(x, generator=generator)
        out = self.refiner(x1, mask=mask, generator=generator,
                           return_parts=return_parts)
        if not return_parts:
            return out
        out, parts = out
        parts["x_phys"] = x1
        return out, parts


class PatchGAN(nn.Module):
    """Optional discriminator. OFF by default in training.

    Patch-level rather than image-level on purpose: the two datasets differ in
    FOV geometry (BRSET's disc is clipped by the frame, mBRSET's is inscribed),
    which stage 2 is not allowed to change and must not be pushed to fake. A
    discriminator that sees whole images would learn that framing difference
    first, since it is the single most separable cue, and the generator would
    then be rewarded for painting it. Scored on crops drawn from inside the
    fundus, it can only judge retinal texture and colour.

    Spectral normalisation on every layer because GAN mode collapse is the most
    likely way this fails on a 17-day deadline, and the adversarial term is a
    stability risk being accepted only behind a flag.
    """

    def __init__(self, in_ch=3, width=32, n_layers=3):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers, c = [], in_ch
        for i in range(n_layers):
            w = width * (2 ** i)
            layers += [sn(nn.Conv2d(c, w, 4, stride=2, padding=1)),
                       nn.LeakyReLU(0.2, inplace=True)]
            c = w
        layers += [sn(nn.Conv2d(c, 1, 3, padding=1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Self-tests. Synthetic tensors only, no image decoding, so these are allowed
# to run on the login node (AGENTS.md compute rule). They check the properties
# the design actually rests on. Everything else in this file is a comment;
# these are the claims that are mechanically verified.
# ---------------------------------------------------------------------------

def _synthetic_fundus(b=3, h=96, w=96, seed=0, lesions=True):
    """A disc of textured retina on a black surround, with dark lesion-sized
    dots at a realistic depth of 0.04 to 0.16. No patient data involved."""
    g = torch.Generator().manual_seed(seed)
    yy = torch.arange(h).view(-1, 1).float()
    xx = torch.arange(w).view(1, -1).float()
    r = ((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2).sqrt()
    disc = (r < 0.46 * min(h, w)).float().view(1, 1, h, w)
    base = torch.tensor([0.70, 0.36, 0.13]).view(1, 3, 1, 1)  # BRSET-like R:G:B
    x = base + 0.05 * torch.randn(b, 3, h, w, generator=g)
    x = x + 0.10 * torch.rand(b, 3, 1, 1, generator=g)
    if lesions:
        for i in range(b):
            for j in range(4):
                cy = int(torch.randint(h // 4, 3 * h // 4, (1,), generator=g))
                cx = int(torch.randint(w // 4, 3 * w // 4, (1,), generator=g))
                depth = 0.04 + 0.12 * float(torch.rand(1, generator=g))
                x[i, :, cy - 1:cy + 2, cx - 1:cx + 2] -= depth
    return (x * disc).clamp(0, 1), disc


def _max_neighbour_step(r):
    """Largest absolute difference between 4-neighbour pixels of a residual.

    This, not the residual's amplitude, is what can erase a lesion. A
    microaneurysm is visible because it sits 0.04 to 0.16 BELOW its immediate
    surround; a residual that shifts a whole neighbourhood equally leaves that
    contrast intact. So the quantity to compare against the lesion floor is the
    local step the residual can take, and for the band-limited branch that is
    bounded by construction at roughly 2*eps_lf/lf_stride.
    """
    dy = (r[:, :, 1:, :] - r[:, :, :-1, :]).abs().max()
    dx = (r[:, :, :, 1:] - r[:, :, :, :-1]).abs().max()
    return float(torch.maximum(dy, dx))


def _selftest():
    torch.manual_seed(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    x, disc = _synthetic_fundus(b=3, h=96, w=96, seed=0)
    print(f"synthetic batch {tuple(x.shape)}  fov fraction {float(disc.mean()):.3f}  "
          f"lesion contrast floor {LESION_CONTRAST_FLOOR}")

    for hf_mode in ("noise", "residual", "off"):
        eps_hf = 0.02
        print(f"\n--- hf_mode={hf_mode}  eps_lf=0.15  eps_hf={eps_hf}")
        torch.manual_seed(0)
        model = Translator(refiner=ResidualRefiner(
            lf_stride=16, eps_lf=0.15, eps_hf=eps_hf, hf_mode=hf_mode, mask_erode=4))
        print(f"  refiner parameters {model.refiner.n_params():,}   "
              f"stage-1 parameters {model.phys.raw.numel()}")

        out, parts = model(x, return_parts=True)
        x1, res, m, meff = parts["x_phys"], parts["residual"], parts["mask"], parts["mask_eff"]

        # 1. output validity
        check("output finite", bool(torch.isfinite(out).all()))
        check("output in [0,1]", bool((out >= 0).all() and (out <= 1).all()),
              f"min {float(out.min()):.4f} max {float(out.max()):.4f}")

        # 2. the residual bound, measured against stage 1's output
        eps = model.refiner.eps_total()
        dev = float((out - x1).abs().max())
        check(f"|out - x_phys| <= eps_total = {eps:.3f}", dev <= eps + 1e-5,
              f"max {dev:.6f}")
        check("|r_lf| <= eps_lf",
              float(parts["r_lf"].abs().max()) <= model.refiner.eps_lf + 1e-6,
              f"max {float(parts['r_lf'].abs().max()):.6f}")
        check("|r_hf| <= eps_hf",
              float(parts["r_hf"].abs().max()) <= eps_hf + 1e-6,
              f"max {float(parts['r_hf'].abs().max()):.6f}")

        # 3. nothing painted outside the field of view
        outside = 1.0 - m
        check("residual exactly zero outside fundus mask",
              float((res.abs() * outside).max()) == 0.0,
              f"max {float((res.abs() * outside).max()):.2e}")
        # meff is the SOFT residual weight, so "outside" it means weight exactly
        # zero, not weight below one.
        zero_w = (meff == 0).float()
        check("residual exactly zero wherever the residual weight is zero",
              float((res.abs() * zero_w).max()) == 0.0,
              f"max {float((res.abs() * zero_w).max()):.2e}")
        check("output zero outside fundus mask",
              float((out.abs() * outside).max()) == 0.0)
        kept = float(meff.sum() / m.sum().clamp(min=1))
        ref0 = model.refiner
        inset = ref0.mask_erode + ref0.mask_taper
        # The inset is a fixed number of PIXELS, so the fraction of the disc it
        # costs scales with image size. At this 96 px test size it is most of
        # the disc; at the 512 px training size the same 20 px inset leaves
        # about (1 - 2*20/470)^2 = 84% of the disc at full weight.
        print(f"  residual weight retains {100 * kept:.1f}% of in-mask area at "
              f"96 px, inset {inset} px (erode {ref0.mask_erode} + taper "
              f"{ref0.mask_taper}); at 512 px the same inset costs ~16% of the disc")

        # 4. gradients reach BOTH stages. The scalar depends on the image
        # everywhere and on each channel separately, so no branch is trivially
        # disconnected from it.
        model.zero_grad()
        loss = (out ** 2).mean() + out.mean(dim=(0, 2, 3)).sum()
        loss.backward()
        gp = model.phys.raw.grad
        check("gradient reaches stage 1",
              gp is not None and bool(torch.isfinite(gp).all()) and float(gp.abs().max()) > 0,
              f"|g| max {float(gp.abs().max()):.3e}, "
              f"{int((gp.abs() > 0).sum())}/{gp.numel()} params nonzero")
        nog = [n for n, p in model.refiner.named_parameters()
               if p.grad is None or float(p.grad.abs().max()) == 0.0]
        check("gradient reaches every stage 2 parameter", len(nog) == 0,
              f"zero-grad tensors: {nog if nog else 'none'}")

        # 5. the high-frequency branch cannot reach lesion contrast even when
        # saturated, which is the point of eps_hf, and in noise mode it is
        # zero-mean so it cannot make a systematic edit at all.
        if hf_mode != "off":
            ref = model.refiner
            with torch.no_grad():
                nn.init.normal_(ref.hf_head.weight, std=5.0)
                nn.init.constant_(ref.hf_head.bias, 6.0)   # drive amp to its ceiling
                draws = torch.stack([ref.high_freq_residual(x1) for _ in range(256)])
            sat = float(draws.abs().max())
            check("saturated hf edit stays below the lesion contrast floor",
                  sat < LESION_CONTRAST_FLOOR,
                  f"max |r_hf| {sat:.4f} < {LESION_CONTRAST_FLOOR}")
            mu = float(draws.mean(dim=0).abs().max())
            if hf_mode == "noise":
                # 256 draws of a zero-mean field with std ~0.627*eps_hf gives a
                # sample-mean spread of about 0.627*0.02/16 = 7.8e-4. Anything
                # at that scale is sampling noise, not a systematic edit.
                check("noise-mode hf residual is zero-mean (no systematic edit)",
                      mu < 4e-3, f"max |mean over 256 draws| {mu:.2e}, "
                                 f"expected sampling scale 7.8e-4")
            else:
                print(f"  [note] residual-mode hf mean over 256 draws {mu:.4f} "
                      f"(deterministic by design; only the bound protects lesions)")

    # 6. the low-frequency branch's band limit, measured and not assumed. Drive
    # the head until the residual saturates its bound, then ask the only
    # question that matters for a lesion: how big a step can it take between
    # adjacent pixels? Measured twice, before and after masking, because an
    # earlier version of this module passed the first and failed the second.
    print("\n--- band limit of the low-frequency branch, saturated on purpose")
    torch.manual_seed(0)
    ref = ResidualRefiner(eps_lf=0.15, eps_hf=0.0, hf_mode="off",
                          mask_erode=4, mask_taper=16)
    with torch.no_grad():
        nn.init.normal_(ref.lf_head.weight, std=5.0)
        nn.init.normal_(ref.lf_head.bias, std=5.0)
        raw = ref.low_freq_residual(x, disc)          # before the residual mask
        _, pr = ref(x, mask=disc, return_parts=True)  # after it
    r = pr["r_lf"]
    band = 2.0 * ref.eps_lf / ref.lf_stride
    lf_bound = ref.max_adjacent_step()["lf"]
    s_raw, s_msk = _max_neighbour_step(raw), _max_neighbour_step(r)
    free = 0.15 * torch.sign(torch.randn_like(r))   # same amplitude, no band limit
    print(f"  |r_lf| max {float(raw.abs().max()):.4f}  "
          f"(bound {ref.eps_lf:.4f}, so the branch is saturated)")
    print(f"  max adjacent-pixel step, unmasked field {s_raw:.5f} "
          f"vs band limit 2*eps_lf/stride {band:.5f}")
    print(f"  max adjacent-pixel step, after taper     {s_msk:.5f} "
          f"vs analytic bound {lf_bound:.5f}")
    print(f"  the same amplitude WITHOUT a band limit:  "
          f"{_max_neighbour_step(free):.5f}  (would erase lesions)")
    check("unmasked LF field obeys its band limit", s_raw <= band + 1e-6)
    check("masked LF residual obeys the analytic bound", s_msk <= lf_bound + 1e-6)
    check("saturated LF residual cannot reach lesion contrast between adjacent px",
          s_msk < LESION_CONTRAST_FLOOR, f"{s_msk:.5f} < {LESION_CONTRAST_FLOOR}")

    # 7. the whole-refiner version of the same statement: the worst-case
    # lesion-scale edit at the shipped defaults, which is the number the paper
    # should quote for the lesion-preservation guarantee. Only the
    # DETERMINISTIC part can erase a lesion; see max_adjacent_step.
    print(f"\n--- analytic worst-case lesion-scale edit, against the "
          f"{LESION_CONTRAST_FLOOR} floor")
    for mode in ("noise", "residual", "off"):
        rr = ResidualRefiner(hf_mode=mode)
        b = rr.max_adjacent_step()
        print(f"  hf_mode={mode:8s} lf {b['lf']:.4f} + hf_det "
              f"{b['hf_deterministic']:.4f} = deterministic {b['deterministic_total']:.4f} "
              f"({100 * b['deterministic_total'] / LESION_CONTRAST_FLOOR:.0f}% of floor); "
              f"hf_stochastic {b['hf_stochastic']:.4f}")
    dflt = ResidualRefiner()
    check("DEFAULT (hf_mode=noise) deterministic edit is below the lesion floor",
          dflt.max_adjacent_step()["deterministic_total"] < LESION_CONTRAST_FLOOR,
          f"{dflt.max_adjacent_step()['deterministic_total']:.4f} "
          f"< {LESION_CONTRAST_FLOOR}")

    # Honest negative, found by this test rather than asserted away: in
    # residual mode the high-frequency branch is deterministic, so eps_hf=0.02
    # pushes the worst case past the floor. The ablation must use the smaller
    # bound computed here, and 83_train_translator.py warns if it does not.
    rmode = ResidualRefiner(hf_mode="residual")
    safe = rmode.max_safe_eps_hf_residual()
    print(f"  hf_mode=residual at eps_hf=0.02 is NOT safe by this metric "
          f"({rmode.max_adjacent_step()['deterministic_total']:.4f} > "
          f"{LESION_CONTRAST_FLOOR}); max safe eps_hf is {safe:.4f}")
    rsafe = ResidualRefiner(hf_mode="residual", eps_hf=safe)
    check("residual mode at the computed safe eps_hf is below the floor",
          rsafe.max_adjacent_step()["deterministic_total"] <= LESION_CONTRAST_FLOOR + 1e-9,
          f"{rsafe.max_adjacent_step()['deterministic_total']:.4f}")

    # And the stochastic term in context: how much per-pixel noise std the
    # high-frequency branch can add, against the measured per-channel std gap
    # it is meant to help close (blue 0.0531 -> 0.1518, a 0.0988 increase).
    sto_std = 0.627 * dflt.eps_hf
    print(f"  noise-mode hf adds at most {sto_std:.4f} of per-pixel std "
          f"(0.627*eps_hf), which is {100 * sto_std / 0.0988:.0f}% of the measured "
          f"blue std increase; stage 1's per-channel noise supplies the rest")

    # 8. the discriminator, since the flagged GAN path must at least be wired.
    print("\n--- optional PatchGAN")
    d = PatchGAN()
    y = d(x)
    print(f"  params {sum(p.numel() for p in d.parameters()):,}  "
          f"output {tuple(y.shape)} (patch logits, not one scalar)")
    check("discriminator output finite", bool(torch.isfinite(y).all()))

    print(f"\n{'ALL SELF-TESTS PASSED' if ok else 'SELF-TESTS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    ap.print_help()
