"""Allocated-node Step-4 transform checks and private qualitative gallery."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
STEP4 = ROOT / "research/codex/step4"
BASE = ROOT / "research/codex/step2/full_pool/frozen"
sys.path.insert(0, str(STEP4))
sys.path.insert(0, str(BASE))

import baseline_core as core  # noqa: E402
import train_augmentation as train_aug  # noqa: E402
from step4_transforms import fundusaug_component, validate_parameters  # noqa: E402

ops_spec = importlib.util.spec_from_file_location("appearance_ops", ROOT / "scripts/55_degradation_ops.py")
ops = importlib.util.module_from_spec(ops_spec)
assert ops_spec.loader is not None
ops_spec.loader.exec_module(ops)

MANIFEST = ROOT / "research/codex/step2/full_pool/protocol/full_pool_manifest.csv"
REFIT = STEP4 / "degradation_refit.json"
OUT = STEP4 / "preflight.json"
PRIVATE = STEP4 / "private"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def node_check() -> None:
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Image checks require an allocated Slurm compute node")


def prepared(path: str) -> Image.Image:
    with Image.open(path) as raw:
        image = raw.convert("RGB").resize((560, 560), Image.Resampling.BILINEAR)
    return image.crop((24, 24, 536, 536))


def main() -> None:
    node_check()
    manifest = pd.read_csv(MANIFEST)
    fit = manifest[manifest.role == "fit"].reset_index(drop=True)
    target_index = int(fit.index[fit.domain == "mBRSET"][0])
    source_index = int(fit.index[fit.domain == "BRSET"][0])

    base_dataset = core.Images(fit, training=True, seed=123456)
    train_aug.CURRENT_ARM = "A1"
    augmented_dataset = train_aug.AugmentedImages(fit, training=True, seed=123456)
    base_target = base_dataset[(target_index, 17)][0]
    augmented_target = augmented_dataset[(target_index, 17)][0]
    if not torch.equal(base_target, augmented_target):
        raise AssertionError("Target-domain common transform changed relative to B1")
    a = augmented_dataset[(source_index, 17)][0]
    b = augmented_dataset[(source_index, 17)][0]
    if not torch.equal(a, b):
        raise AssertionError("A1 source transform is not deterministic by draw")

    refit = json.loads(REFIT.read_text())
    full = {key: float(value) for key, value in refit["arms"]["full"]["params"].items()}
    overlay_free = {key: float(value) for key, value in refit["arms"]["overlay_free"]["params"].items()}
    validate_parameters(full, False)
    validate_parameters(overlay_free, True)
    for arm in ("A1", "A2", "A3"):
        if train_aug.step4_phase_specs(arm, True)[0]["domain"] != "both":
            raise AssertionError(f"Step-4 smoke is not using the joint B1 pool: {arm}")
        if train_aug.step4_phase_specs(arm, False)[0]["domain"] != "both":
            raise AssertionError(f"Step-4 full run is not using the joint B1 pool: {arm}")

    source = manifest[(manifest.role == "fit") & (manifest.domain == "BRSET")]
    target = manifest[(manifest.role == "fit") & (manifest.domain == "mBRSET")]
    rng = np.random.default_rng(20260914)
    source_rows = source.iloc[rng.choice(len(source), 4, replace=False)]
    target_rows = target.iloc[rng.choice(len(target), 4, replace=False)]
    figure, axes = plt.subplots(4, 5, figsize=(12.0, 9.8))
    titles = ["BRSET original", "FundusAug component", "Target-fitted full", "Target-fitted no overlays", "Real mBRSET (unpaired)"]
    ledger_totals = {key: 0 for key in ("sharpness", "halo", "hole", "spot", "blur")}
    for row_index, (source_row, target_row) in enumerate(zip(source_rows.itertuples(), target_rows.itertuples())):
        original = prepared(source_row.image_path)
        a1, ledger = fundusaug_component(original, np.random.default_rng(core.stream_seed(20260914, "gallery_a1", row_index)))
        for key, value in ledger.items():
            ledger_totals[key] += value
        a2 = ops.degrade(original, full, np.random.default_rng(core.stream_seed(20260914, "gallery_a2", row_index)))
        a3 = ops.degrade(original, overlay_free, np.random.default_rng(core.stream_seed(20260914, "gallery_a3", row_index)))
        real = prepared(target_row.image_path)
        for column, image in enumerate((original, a1, a2, a3, real)):
            axes[row_index, column].imshow(image)
            axes[row_index, column].axis("off")
            if row_index == 0:
                axes[row_index, column].set_title(titles[column], fontsize=10)
    figure.tight_layout()
    PRIVATE.mkdir(exist_ok=True)
    gallery = PRIVATE / "step4_transform_gallery.png"
    figure.savefig(gallery, dpi=150, bbox_inches="tight")
    plt.close(figure)

    output = {
        "pass": True,
        "scope": "original training images only; gallery target references are unrelated to source rows",
        "target_common_transform_bitwise_equal_to_B1": True,
        "a1_repeat_deterministic": True,
        "full_parameters_valid": True,
        "overlay_free_parameters_valid": True,
        "all_step4_arms_use_joint_B1_phase_in_smoke_and_full_runs": True,
        "gallery_rows": 4,
        "gallery_columns": titles,
        "gallery_sha256": sha(gallery),
        "gallery_a1_operator_counts_for_four_display_draws": ledger_totals,
        "input_sha256": {
            "manifest": sha(MANIFEST),
            "refit": sha(REFIT),
            "transform": sha(STEP4 / "step4_transforms.py"),
            "trainer": sha(STEP4 / "train_augmentation.py"),
            "preflight": sha(Path(__file__)),
        },
        "limitations": "Visual examples and pixel-level determinism do not establish lesion preservation or clinical realism.",
    }
    OUT.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
