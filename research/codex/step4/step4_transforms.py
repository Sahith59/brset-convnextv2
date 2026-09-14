"""Independent Step-4 source-image transformations.

FundusAug behavior is reimplemented from the MICCAI 2023 paper and inspected
release ranges. This file contains no copied upstream source. Adaptations are
explicit in STEP_4_PLAN_DRAFT.md.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _chance(rng: np.random.Generator, probability: float = 0.5) -> bool:
    return bool(rng.random() < probability)


def _as_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def _as_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB")


def _gaussian_blob(size: int, center_y: float, center_x: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    return np.exp(-((yy - center_y) ** 2 + (xx - center_x) ** 2) / (2.0 * max(sigma, 1e-6) ** 2))


def fundusaug_component(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, dict[str, int]]:
    """Apply release-range FundusAug artifact operations at probability 0.5.

    Geometry and color jitter are intentionally supplied by the common B1
    pipeline. Operations here are sharpness, halo, hole, spot and blur.
    """
    output = image
    size = output.size[0]
    if output.size != (size, size):
        raise ValueError("FundusAug component expects a square image")
    ledger = {name: 0 for name in ("sharpness", "halo", "hole", "spot", "blur")}

    if _chance(rng):
        output = ImageEnhance.Sharpness(output).enhance(float(rng.uniform(0.0, 2.0)))
        ledger["sharpness"] = 1

    if _chance(rng):
        array = _as_float(output)
        center = float(rng.integers(size // 2 - size // 8, size // 2 + size // 8 + 1))
        inner_radius = 0.5 * size * float(rng.uniform(0.75, 1.0))
        outer_radius = inner_radius + float(rng.uniform(0.12, 0.42)) * size
        yy, xx = np.mgrid[:size, :size]
        radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
        width = max((outer_radius - inner_radius) / 3.0, 1.0)
        ring = np.exp(-((radius - (inner_radius + outer_radius) / 2.0) ** 2) / (2.0 * width**2))
        weights = np.array([[251, 249, 246], [141, 238, 238], [177, 195, 147]], dtype=np.float32) / 255.0
        color = weights[int(rng.integers(0, len(weights)))]
        strength = float(rng.choice([1.0, 1.2, 1.5, 2.0], p=[0.8, 0.1, 0.05, 0.05]))
        output = _as_image(array + ring[..., None] * color * strength * float(array.mean()))
        ledger["halo"] = 1

    if _chance(rng):
        array = _as_float(output)
        diameter = float(rng.uniform(0.4, 0.9)) * size
        center_y = float(rng.integers(size // 4, 3 * size // 4 + 1))
        center_x = float(rng.integers(3 * size // 8, 5 * size // 8 + 1))
        sigma = float(rng.uniform(0.55, 0.75)) * diameter
        field = array.mean(axis=2) > 0.04
        mean_color = float(array[field].mean()) if field.any() else float(array.mean())
        strength = float(rng.uniform(-0.26 - 0.06 * mean_color, -0.26 - 0.05 * mean_color)) if mean_color > 0.25 else 0.0
        output = _as_image(array + strength * _gaussian_blob(size, center_y, center_x, sigma)[..., None])
        ledger["hole"] = 1

    if _chance(rng):
        array = _as_float(output)
        addition = np.zeros((size, size), dtype=np.float32)
        for _ in range(int(rng.integers(5, 11))):
            radius = float(rng.uniform(0.01, 0.05)) * size
            center_y = float(rng.uniform(radius + 1, size - radius - 1))
            center_x = float(rng.uniform(radius + 1, size - radius - 1))
            strength = float(rng.uniform(0.08, 0.28))
            addition += strength * _gaussian_blob(size, center_y, center_x, max(radius / 2.0, 1.0))
        output = _as_image(array + addition[..., None])
        ledger["spot"] = 1

    if _chance(rng):
        sigma = float(rng.uniform(0.1, 3.0))
        output = output.filter(ImageFilter.GaussianBlur(radius=sigma))
        ledger["blur"] = 1

    return output, ledger


def fitted_component(image: Image.Image, rng: np.random.Generator, params: dict[str, float],
                     appearance_ops) -> Image.Image:
    return appearance_ops.degrade(image, params, rng)


def validate_parameters(params: dict[str, float], overlay_free: bool) -> None:
    required = {
        "blur_sigma", "light_strength", "n_spot", "n_hole", "halo",
        "brightness_gain", "contrast_gain", "saturation_gain", "noise_sigma",
    }
    if set(params) != required:
        raise ValueError("Fitted parameter keys changed")
    if not all(math.isfinite(float(value)) for value in params.values()):
        raise ValueError("Nonfinite fitted parameter")
    if overlay_free and any(float(params[key]) != 0.0 for key in ("n_spot", "n_hole", "halo")):
        raise ValueError("Overlay-free arm contains an overlay parameter")
