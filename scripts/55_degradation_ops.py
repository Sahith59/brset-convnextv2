"""Fundus degradation operators, and the statistics used to fit them.

EXPERIMENT 3, step 3a and 3b.

The operator set follows the two papers the method builds on. cofe-Net (Shen,
Fu, Shen and Shao, IEEE TMI 2021) derives fundus degradation from the optics of
the ophthalmoscope and names three factors: light transmission disturbance,
image blurring, and retinal artifacts. GDRNet's FundusAug (Che et al., MICCAI
2023) adds colour transformations and applies everything at a fixed probability
of 0.5 with hand-chosen magnitudes.

The contribution here is not the operators, which are published, but that their
parameters are FITTED so that degraded BRSET images match the measured
statistics of real mBRSET images, rather than being chosen by hand for generic
robustness.

Everything below uses image appearance only. No diagnostic label is read.
"""
import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage

# --------------------------------------------------------------------------
# statistics. These are what the fitting matches, so they must be cheap and
# must capture the three degradation factors above.
# --------------------------------------------------------------------------

def fundus_mask(gray, thresh=0.04):
    """The circular field of view. Fundus images sit on a black surround, and
    including it would dominate every statistic."""
    return gray > thresh


def image_stats(rgb):
    """Return the statistics used to fit the degradation.

    sharpness  Laplacian variance inside the field of view. Falls with blur.
    brightness mean intensity inside the field of view.
    contrast   intensity standard deviation inside the field of view.
    falloff    centre minus periphery brightness, which captures uneven
               illumination, the light-transmission factor in cofe-Net.
    saturation mean saturation, which colour jitter moves.
    """
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    g = a.mean(axis=2)
    m = fundus_mask(g)
    if m.sum() < 1000:
        return None
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    inner = m & (r < 0.25 * min(h, w))
    outer = m & (r > 0.39 * min(h, w))
    mx = a.max(axis=2); mn = a.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {
        "sharpness": float(ndimage.laplace(g)[m].var()),
        "brightness": float(g[m].mean()),
        "contrast": float(g[m].std()),
        "falloff": float(g[inner].mean() - g[outer].mean()) if outer.sum() > 500 else np.nan,
        "saturation": float(sat[m].mean()),
    }


# --------------------------------------------------------------------------
# the three cofe-Net degradation factors, plus GDRNet's colour operations
# --------------------------------------------------------------------------

def light_disturbance(rgb, strength, rng):
    """Uneven illumination. A smooth multiplicative light map centred off-axis,
    which is what happens when a handheld camera is not aligned with the pupil."""
    if strength <= 0:
        return rgb
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    h, w = a.shape[:2]
    cy = h / 2 + rng.uniform(-0.22, 0.22) * h
    cx = w / 2 + rng.uniform(-0.22, 0.22) * w
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * (0.55 * min(h, w)) ** 2)
    light = 1.0 + strength * (np.exp(-d2) - 0.5)
    return Image.fromarray((np.clip(a * light[..., None], 0, 1) * 255).astype(np.uint8))


def blur(rgb, sigma):
    """Defocus. Handheld cameras focus through the eye's own lens and often miss."""
    if sigma <= 0:
        return rgb
    a = np.asarray(rgb, dtype=np.float32)
    out = np.stack([ndimage.gaussian_filter(a[..., c], sigma) for c in range(3)], axis=2)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def artifacts(rgb, n_spot, n_hole, halo, rng):
    """Retinal artifacts. Stray light bouncing off cornea and lens produces
    bright spots and halos; dust and debris produce dark holes."""
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    R = min(h, w) / 2.0
    if halo > 0:
        r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
        ring = np.exp(-((r - 0.82 * R) ** 2) / (2 * (0.10 * R) ** 2))
        a = a + halo * ring[..., None]
    for _ in range(int(n_spot)):
        cy, cx = rng.uniform(0.15, 0.85, 2) * np.array([h, w])
        rad = rng.uniform(0.015, 0.055) * R
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2)))
        a = a + rng.uniform(0.10, 0.30) * blob[..., None]
    for _ in range(int(n_hole)):
        cy, cx = rng.uniform(0.15, 0.85, 2) * np.array([h, w])
        rad = rng.uniform(0.015, 0.045) * R
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2)))
        a = a - rng.uniform(0.10, 0.30) * blob[..., None]
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def sensor_noise(rgb, sigma, rng):
    """Additive sensor noise. Step 3a measured that mBRSET has 5 to 7 times MORE
    high-frequency energy than BRSET, not less, which is what a small phone
    sensor and in-camera sharpening produce. Neither cofe-Net nor FundusAug
    provides this, because both assume degradation means blur."""
    if sigma <= 0:
        return rgb
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    g = a.mean(axis=2)
    m = fundus_mask(g)[..., None]
    a = a + rng.normal(0.0, sigma, a.shape).astype(np.float32) * m
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def colour(rgb, brightness_gain, contrast_gain, saturation_gain):
    """DIRECTED colour shift, not the symmetric jitter GDRNet uses. Step 3a
    measured that mBRSET is dimmer, higher contrast and less saturated than
    BRSET, so these must be able to move in a fixed direction rather than
    wobbling either side of 1.0."""
    out = rgb
    for enh, gain in ((ImageEnhance.Brightness, brightness_gain),
                      (ImageEnhance.Contrast, contrast_gain),
                      (ImageEnhance.Color, saturation_gain)):
        if gain is not None and abs(gain - 1.0) > 1e-6:
            out = enh(out).enhance(gain)
    return out


def degrade(rgb, p, rng):
    """Apply the full chain with parameter dict p. Order matters: geometry and
    optics first, then the sensor, then the camera's colour pipeline."""
    out = rgb
    if p.get("blur_sigma", 0) > 0:
        out = blur(out, p["blur_sigma"])
    out = light_disturbance(out, p.get("light_strength", 0.0), rng)
    out = artifacts(out, p.get("n_spot", 0), p.get("n_hole", 0), p.get("halo", 0.0), rng)
    out = sensor_noise(out, p.get("noise_sigma", 0.0), rng)
    out = colour(out, p.get("brightness_gain", 1.0), p.get("contrast_gain", 1.0),
                 p.get("saturation_gain", 1.0))
    return out


# GDRNet FundusAug at its published settings, for the control comparison.
# Symmetric jitter, fixed probability, no noise term.
GDRNET_GENERIC = {"blur_sigma": 1.0, "light_strength": 0.25, "n_spot": 2, "n_hole": 1,
                  "halo": 0.10, "brightness_gain": 1.15, "contrast_gain": 1.15,
                  "saturation_gain": 1.10, "noise_sigma": 0.0}

# Starting point for the fit. Values are found by scripts/57, not chosen by hand.
FIT_INIT = {"blur_sigma": 0.0, "light_strength": 0.20, "n_spot": 1, "n_hole": 1,
            "halo": 0.05, "brightness_gain": 0.92, "contrast_gain": 1.35,
            "saturation_gain": 0.95, "noise_sigma": 0.010}
