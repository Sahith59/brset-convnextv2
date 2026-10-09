"""Differentiable fundus acquisition model. Stage one of the dual-domain translator.

DuDoNet (Lin et al., CVPR 2019) restores CT in two domains at once, the sinogram
domain where the physics lives and the image domain where appearance lives, and
links them with a differentiable Radon inversion layer so the whole thing trains
end to end. The reason that works is that neither domain alone is enough:
sinogram-only restoration leaves secondary artifacts, image-only restoration has
no access to the acquisition process.

The same split applies here. Our hand-fitted degradation chain (script 55) is an
acquisition model: blur, uneven illumination, artifacts, sensor noise and the
camera colour pipeline. Fitting it to the measured mBRSET statistics drove the
normalised appearance distance from 5.13 to 1.36, so it captures the statistics
well. But the fitted images still do not LOOK like mBRSET: framing and hue
dominate perceptually and are invisible to those five statistics. Physics alone
is not enough, exactly as in DuDoNet.

This module is the differentiable version of that chain, so a learned refinement
stage can sit behind it and the two can be optimised jointly. Every operator
from script 55 is reimplemented in torch with its parameters free, and the
numpy version remains the reference that this is tested against.

Parameters are unconstrained internally and mapped to valid ranges by softplus
or sigmoid, so optimisation never has to respect a boundary.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def fundus_mask(x, thresh=0.04):
    """Circular field of view. Fundus images sit on a black surround and
    including it would dominate every statistic. Hard threshold, no gradient
    needed through the mask itself since it is a property of the input."""
    return (x.mean(dim=1, keepdim=True) > thresh).float()


def gaussian_blur(x, sigma, ksize=21):
    """Separable Gaussian with a LEARNABLE sigma.

    The kernel is rebuilt from sigma on every call rather than cached, which is
    what makes d(output)/d(sigma) exist. Support is fixed so the graph shape
    does not depend on the parameter value.
    """
    # sigma: scalar tensor. Guard the degenerate case without breaking gradient.
    sigma = sigma.clamp(min=1e-3)
    half = ksize // 2
    pos = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(pos ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, ksize)
    ky = k.view(1, 1, -1, 1).expand(c, 1, ksize, 1)
    x = F.conv2d(F.pad(x, (half, half, 0, 0), mode="reflect"), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, half, half), mode="reflect"), ky, groups=c)
    return x


def _radial(h, w, device, dtype, cy=None, cx=None):
    yy = torch.arange(h, device=device, dtype=dtype).view(-1, 1).expand(h, w)
    xx = torch.arange(w, device=device, dtype=dtype).view(1, -1).expand(h, w)
    cy = (h - 1) / 2.0 if cy is None else cy
    cx = (w - 1) / 2.0 if cx is None else cx
    return yy, xx, torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


def rgb_to_gray(x):
    return x.mean(dim=1, keepdim=True)


def saturation_map(x):
    mx = x.max(dim=1, keepdim=True).values
    mn = x.min(dim=1, keepdim=True).values
    return (mx - mn) / mx.clamp(min=EPS)


def laplacian(x):
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k)


def image_stats_torch(x, mask=None):
    """The same five statistics script 55 fits, in torch and differentiable.

    Returned per-image so a batch gives a distribution, not just a mean. These
    are what anchor the learned stage to the measured acquisition physics.
    """
    if mask is None:
        mask = fundus_mask(x)
    g = rgb_to_gray(x)
    n = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)

    mean = (g * mask).sum(dim=(1, 2, 3)) / n
    var = ((g - mean.view(-1, 1, 1, 1)) ** 2 * mask).sum(dim=(1, 2, 3)) / n
    lap = laplacian(g)
    lap_mean = (lap * mask).sum(dim=(1, 2, 3)) / n
    lap_var = ((lap - lap_mean.view(-1, 1, 1, 1)) ** 2 * mask).sum(dim=(1, 2, 3)) / n

    b, _, h, w = x.shape
    _, _, r = _radial(h, w, x.device, x.dtype)
    s = min(h, w)
    inner = ((r < 0.25 * s).float().view(1, 1, h, w) * mask)
    outer = ((r > 0.39 * s).float().view(1, 1, h, w) * mask)
    gi = (g * inner).sum(dim=(1, 2, 3)) / inner.sum(dim=(1, 2, 3)).clamp(min=1.0)
    go = (g * outer).sum(dim=(1, 2, 3)) / outer.sum(dim=(1, 2, 3)).clamp(min=1.0)

    sat = (saturation_map(x) * mask).sum(dim=(1, 2, 3)) / n

    out = {"sharpness": lap_var, "brightness": mean, "contrast": var.clamp(min=0).sqrt(),
           "falloff": gi - go, "saturation": sat}

    # CHROMATIC moments. Added 9 Oct after verifying that none of the five
    # statistics above carries chromatic information: every one is computed on
    # grayscale or on (max-min)/max. Nothing in the pipeline had a gradient
    # toward correct hue, which is the single largest perceptual gap.
    for i, c in enumerate("rgb"):
        ch = x[:, i:i + 1]
        cm = (ch * mask).sum(dim=(1, 2, 3)) / n
        cv = ((ch - cm.view(-1, 1, 1, 1)) ** 2 * mask).sum(dim=(1, 2, 3)) / n
        out[f"{c}_mean"] = cm
        out[f"{c}_std"] = cv.clamp(min=0).sqrt()
    tot = out["r_mean"] + out["g_mean"] + out["b_mean"] + 1e-9
    for c in "rgb":
        out[f"{c}_frac"] = out[f"{c}_mean"] / tot
    return out


class PhysicalAcquisition(nn.Module):
    """Learnable fundus acquisition model, the acquisition-domain stage.

    Nine interpretable parameters, the same nine script 57 fitted by random
    search, now trained by gradient descent jointly with the refinement stage.
    Initialising at the fitted values means the learned stage starts from the
    best purely physical explanation of the gap rather than from noise.

    Stochastic placement (where the illumination centre sits, where artifacts
    land) stays random per sample, as in the numpy version. Only the magnitudes
    are learned, which is the part the statistics actually constrain.
    """

    # Fifteen bounded parameters. Extended 9 Oct after verifying the previous
    # nine had NO chromatic degree of freedom: sweeping every one to both
    # extremes moved the R:G:B ratio by at most 0.00088. The earlier claim that
    # the physical model was "exhausted" was therefore false. It was not
    # exhausted, it had no colour parameter. Three additions, each physically
    # interpretable and each provably photometric so none can move anatomy:
    #
    #  - per-channel gain and offset: a different sensor's spectral response and
    #    black level. This is what hue actually is, and the old brightness_gain
    #    (a single scalar on all channels) is a special case of it.
    #  - signal-dependent noise sigma(x) = sqrt(a*x + b^2): shot plus read noise,
    #    the actual physics of a small sensor. The old i.i.d. Gaussian supplied
    #    about 81 percent of the matched sharpness statistic by itself, which is
    #    both wrong physics and destroyed by JPEG and resampling.
    #  - a SIGNED sharpness operator x + k*(x - blur(x)): k < 0 blurs, k > 0
    #    sharpens. The old blur-only parameter fitted to 0.907 and so BLURRED the
    #    source to make room for synthetic noise, which is backwards given the
    #    measured fact that mBRSET carries 5 to 7 times MORE high-frequency
    #    energy. It also destroyed microaneurysm-scale structure, the exact
    #    thing this pipeline promises to preserve.
    FITTED = {"sharpen_k": 0.0, "light_strength": 0.2613, "halo": 0.0857,
              "spot_amp": 0.20, "hole_amp": 0.20, "noise_a": 1e-4, "noise_b": 0.004,
              "gain_r": 1.0, "gain_g": 1.0, "gain_b": 1.0,
              "off_r": 0.0, "off_g": 0.0, "off_b": 0.0,
              "contrast_gain": 0.9160, "saturation_gain": 1.0028}

    # Bounds are load-bearing, not cosmetic. Unbounded, fitting the statistics by
    # gradient descent drove spot and hole amplitude to 2.98 and 1.95 on a [0,1]
    # image: the optimiser hit the targets by painting large blobs, numerically
    # excellent and visually destroyed.
    RANGES = {"sharpen_k": (-0.95, 2.50), "light_strength": (0.0, 0.60),
              "halo": (0.0, 0.30), "spot_amp": (0.0, 0.40), "hole_amp": (0.0, 0.40),
              "noise_a": (0.0, 2e-3), "noise_b": (0.0, 0.03),
              "gain_r": (0.60, 1.60), "gain_g": (0.60, 1.60), "gain_b": (0.60, 1.60),
              "off_r": (-0.15, 0.15), "off_g": (-0.15, 0.15), "off_b": (-0.15, 0.15),
              "contrast_gain": (0.50, 1.60), "saturation_gain": (0.50, 1.60)}

    KEYS = ["sharpen_k", "light_strength", "halo", "spot_amp", "hole_amp",
            "noise_a", "noise_b", "gain_r", "gain_g", "gain_b",
            "off_r", "off_g", "off_b", "contrast_gain", "saturation_gain"]

    BLUR_SIGMA = 1.0   # fixed scale of the unsharp kernel; only k is learned

    def __init__(self, init=None, n_spot=1, n_hole=3, learnable=True):
        super().__init__()
        p = dict(self.FITTED)
        if init:
            p.update(init)
        self.n_spot, self.n_hole = n_spot, n_hole
        raws = []
        for k in self.KEYS:
            lo, hi = self.RANGES[k]
            u = min(max((float(p[k]) - lo) / (hi - lo), 1e-3), 1 - 1e-3)
            raws.append(float(np.log(u / (1.0 - u))))
        self.raw = nn.Parameter(torch.tensor(raws, dtype=torch.float32),
                                requires_grad=learnable)

    def params(self):
        out = {}
        for i, k in enumerate(self.KEYS):
            lo, hi = self.RANGES[k]
            out[k] = lo + (hi - lo) * torch.sigmoid(self.raw[i])
        return out

    def forward(self, x, generator=None):
        """x in [0,1], shape (B,3,H,W).

        Order follows the physical chain: optics, illumination, artifacts,
        sensor noise, then the camera's colour pipeline. Every operation is
        photometric or a smooth global field, so none can displace anatomy.
        """
        p = self.params()
        b, _, h, w = x.shape
        dev, dt = x.device, x.dtype
        mask = fundus_mask(x)

        # signed sharpness: k<0 blurs, k>0 unsharp-masks
        blurred = gaussian_blur(x, torch.as_tensor(self.BLUR_SIGMA, device=dev, dtype=dt))
        out = x + p["sharpen_k"] * (x - blurred)

        # uneven illumination, centre jittered off-axis per sample
        off = (torch.rand(b, 2, device=dev, generator=generator) * 0.44 - 0.22)
        yy, xx, _ = _radial(h, w, dev, dt)
        cy = (h - 1) / 2.0 + off[:, 0].view(-1, 1, 1) * h
        cx = (w - 1) / 2.0 + off[:, 1].view(-1, 1, 1) * w
        d2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * (0.55 * min(h, w)) ** 2)
        out = out * (1.0 + p["light_strength"] * (torch.exp(-d2).unsqueeze(1) - 0.5))

        # halo ring plus bright spots and dark holes
        _, _, r = _radial(h, w, dev, dt)
        R = min(h, w) / 2.0
        ring = torch.exp(-((r - 0.82 * R) ** 2) / (2 * (0.10 * R) ** 2))
        out = out + p["halo"] * ring.view(1, 1, h, w)
        for amp, cnt, sign in ((p["spot_amp"], self.n_spot, 1.0),
                               (p["hole_amp"], self.n_hole, -1.0)):
            for _ in range(cnt):
                c = torch.rand(b, 2, device=dev, generator=generator) * 0.7 + 0.15
                rad = (torch.rand(b, 1, device=dev, generator=generator) * 0.04 + 0.015) * R
                by = c[:, 0].view(-1, 1, 1) * h
                bx = c[:, 1].view(-1, 1, 1) * w
                blob = torch.exp(-(((yy - by) ** 2 + (xx - bx) ** 2)
                                   / (2 * rad.view(-1, 1, 1) ** 2))).unsqueeze(1)
                out = out + sign * amp * blob

        # signal-dependent sensor noise: shot (proportional to signal) + read
        sigma = (p["noise_a"] * out.clamp(min=0.0) + p["noise_b"] ** 2).clamp(min=1e-12).sqrt()
        eps = torch.randn(x.shape, device=dev, dtype=dt, generator=generator)
        out = out + sigma * eps * mask

        # camera colour pipeline: per-channel affine (white balance and black
        # level), then global contrast, then saturation
        gain = torch.stack([p["gain_r"], p["gain_g"], p["gain_b"]]).view(1, 3, 1, 1)
        offs = torch.stack([p["off_r"], p["off_g"], p["off_b"]]).view(1, 3, 1, 1)
        out = out * gain + offs

        gmean = (rgb_to_gray(out) * mask).sum(dim=(1, 2, 3), keepdim=True) \
            / mask.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        out = gmean + (out - gmean) * p["contrast_gain"]
        gray = rgb_to_gray(out)
        out = gray + (out - gray) * p["saturation_gain"]

        return (out * mask).clamp(0.0, 1.0)
