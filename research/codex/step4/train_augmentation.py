"""Run Step-4 augmentation arms through the frozen full-pool B1 trainer."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
BASE = ROOT / "research/codex/step2/full_pool/frozen"
STEP4 = ROOT / "research/codex/step4"
sys.path.insert(0, str(BASE))

import baseline_core as core  # noqa: E402
import train_baseline as baseline  # noqa: E402
from step4_transforms import fundusaug_component, fitted_component, validate_parameters  # noqa: E402

ops_spec = importlib.util.spec_from_file_location("appearance_ops", ROOT / "scripts/55_degradation_ops.py")
appearance_ops = importlib.util.module_from_spec(ops_spec)
assert ops_spec.loader is not None
ops_spec.loader.exec_module(appearance_ops)

CURRENT_ARM = ""
REFIT = STEP4 / "degradation_refit.json"


class AugmentedImages(torch.utils.data.Dataset):
    def __init__(self, rows, training=False, seed=0):
        self.rows = rows.reset_index(drop=True)
        self.seed = seed
        self.training = training
        if training and not self.rows.role.eq("fit").all():
            raise ValueError("Training dataset contains non-fit images")
        # The extra source-domain operation is inserted after the crop used by
        # the refit and before the ordinary B1 spatial/colour augmentation.
        # Splitting the baseline Compose does not change its torch RNG order.
        self.pre = T.Compose([
            T.Resize((560, 560), interpolation=InterpolationMode.BILINEAR, antialias=True),
            T.RandomCrop(512),
        ]) if training else T.Compose([
            T.Resize((560, 560), interpolation=InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop(512),
        ])
        self.post = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(15, interpolation=InterpolationMode.NEAREST, expand=False, fill=0),
            T.ColorJitter(brightness=.2, contrast=.2, saturation=.1, hue=0),
        ]) if training else None
        self.final = T.Compose([T.ToTensor(), T.Normalize(core.MEAN, core.STD)])
        self.params = None
        if training and CURRENT_ARM in {"A2", "A3"}:
            refit = json.loads(REFIT.read_text())
            key = "full" if CURRENT_ARM == "A2" else "overlay_free"
            self.params = {name: float(value) for name, value in refit["arms"][key]["params"].items()}
            validate_parameters(self.params, overlay_free=CURRENT_ARM == "A3")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        idx, draw = key if isinstance(key, tuple) else (key, 0)
        row = self.rows.iloc[idx]
        with Image.open(row.image_path) as raw:
            image = raw.convert("RGB")
        if self.training:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(core.stream_seed(self.seed, "common_transform", draw))
                image = self.pre(image)
                if row.domain == "BRSET":
                    rng = np.random.default_rng(core.stream_seed(self.seed, f"step4_{CURRENT_ARM}", draw))
                    if CURRENT_ARM == "A1":
                        image, _ = fundusaug_component(image, rng)
                    elif CURRENT_ARM in {"A2", "A3"}:
                        image = fitted_component(image, rng, self.params, appearance_ops)
                    else:
                        raise ValueError(f"Unknown Step-4 arm {CURRENT_ARM}")
                image = self.post(image)
        else:
            image = self.pre(image)
        return self.final(image), torch.tensor(row[core.LABELS].to_numpy(dtype=np.float32)), idx


original_contract = baseline.contract


def step4_phase_specs(_arm, smoke=False):
    """Every Step-4 arm inherits the joint-training B1 phase schedule."""
    return core.phase_specs("B1", smoke)


def step4_contract(args, protocol, initialization):
    result = original_contract(args, protocol, initialization)
    result.update({
        "step4_wrapper_sha256": core.sha(Path(__file__)),
        "step4_transform_sha256": core.sha(STEP4 / "step4_transforms.py"),
        "source_only_extra_transform": True,
        "source_transform_order": "resize_560-random_crop_512-extra_source_transform-flip-rotation-colour_jitter",
        "reference_arm": "B1",
    })
    if args.arm in {"A2", "A3"}:
        result["degradation_refit_sha256"] = core.sha(REFIT)
        result["degradation_parameter_arm"] = "full" if args.arm == "A2" else "overlay_free"
    if args.arm == "A1":
        result["fundusaug_upstream_commit"] = "10f339748de69bd2f4b1bec49858eef95082679c"
        result["fundusaug_scope"] = "independent release-range artifact component adapted to 512; not full GDRNet"
    return result


def main() -> None:
    global CURRENT_ARM
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["A1", "A2", "A3"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    CURRENT_ARM = args.arm
    baseline.Images = AugmentedImages
    baseline.contract = step4_contract
    baseline.phase_specs = step4_phase_specs
    baseline.train(args)


if __name__ == "__main__":
    main()
