"""Training-only refit of full and overlay-free BRSET-to-mBRSET transforms.

Image decoding and numerical work are restricted to an allocated Slurm node.
The public output contains aggregate statistics and identity hashes, not file or
patient identifiers.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get("BRSET_REPO", "/home/users/sthummala2/brset-convnextv2"))
STEP4 = ROOT / "research/codex/step4"
MANIFEST = ROOT / "research/codex/step2/full_pool/protocol/full_pool_manifest.csv"
AUDIT = STEP4 / "appearance_label_audit.json"
OUT = STEP4 / "degradation_refit.json"

spec = importlib.util.spec_from_file_location("appearance_ops", ROOT / "scripts/55_degradation_ops.py")
ops = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ops)

STATS = ["sharpness", "brightness", "contrast", "falloff", "saturation"]
CALIBRATION_N = 96
VALIDATION_N = 96
RANDOM_CANDIDATES = 256
FIT_DRAWS = 2
VALIDATION_DRAWS = 3
SELECTION_SEED = 20260914

RANGES = {
    "blur_sigma": (0.0, 1.2),
    "light_strength": (0.0, 0.6),
    "n_spot": (0, 4),
    "n_hole": (0, 3),
    "halo": (0.0, 0.20),
    "brightness_gain": (0.70, 1.15),
    "contrast_gain": (0.90, 2.0),
    "saturation_gain": (0.50, 1.20),
    "noise_sigma": (0.0, 0.030),
}

_calibration_images: list[Image.Image] = []
_target: dict[str, float] = {}
_scale: dict[str, float] = {}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    raw = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def node_check() -> None:
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Image decoding and fitting require an allocated Slurm compute node")


def load_image(path: str) -> Image.Image:
    with Image.open(path) as raw:
        image = raw.convert("RGB").resize((560, 560), Image.Resampling.BILINEAR)
    return image.crop((24, 24, 536, 536))


def select_patient_disjoint(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rng = np.random.default_rng(SELECTION_SEED)
    shuffled = rows.iloc[rng.permutation(len(rows))]
    unique = shuffled.drop_duplicates("patient_key", keep="first")
    need = CALIBRATION_N + VALIDATION_N
    if len(unique) < need:
        raise ValueError("Insufficient unique source patients")
    chosen = unique.iloc[:need].copy()
    calibration = chosen.iloc[:CALIBRATION_N].reset_index(drop=True)
    validation = chosen.iloc[CALIBRATION_N:].reset_index(drop=True)
    if set(calibration.patient_key) & set(validation.patient_key):
        raise AssertionError("Calibration/validation patient overlap")
    identity_material = "\n".join(
        f"{row.domain}|{row.file}|{row.patient_key}" for row in chosen.itertuples(index=False)
    ).encode()
    return calibration, validation, hashlib.sha256(identity_material).hexdigest()


def sample_parameters(rng: np.random.Generator, overlay_free: bool) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, (low, high) in RANGES.items():
        if overlay_free and name in {"n_spot", "n_hole", "halo"}:
            result[name] = 0.0
        elif name.startswith("n_"):
            result[name] = float(rng.integers(int(low), int(high) + 1))
        else:
            result[name] = float(rng.uniform(low, high))
    return result


def candidate_list(overlay_free: bool) -> list[dict[str, float]]:
    rng = np.random.default_rng(stable_seed(SELECTION_SEED, "overlay_free" if overlay_free else "full"))
    historical = json.loads((ROOT / "results/fitted_degradation_params.json").read_text())
    if overlay_free:
        historical.update({"n_spot": 0.0, "n_hole": 0.0, "halo": 0.0})
    initial = {key: float(value) for key, value in ops.FIT_INIT.items()}
    if overlay_free:
        initial.update({"n_spot": 0.0, "n_hole": 0.0, "halo": 0.0})
    candidates = [historical, initial]
    candidates.extend(sample_parameters(rng, overlay_free) for _ in range(RANDOM_CANDIDATES))
    return candidates


def aggregate(images: list[Image.Image], params: dict[str, float] | None, arm: str,
              candidate_index: int, draws: int) -> dict[str, float]:
    rows = []
    for draw in range(draws):
        for image_index, image in enumerate(images):
            transformed = image if params is None else ops.degrade(
                image, params, np.random.default_rng(stable_seed(SELECTION_SEED, arm, candidate_index, draw, image_index))
            )
            row = ops.image_stats(transformed)
            if row is not None and all(np.isfinite(row[name]) for name in STATS):
                rows.append(row)
    if len(rows) != len(images) * draws:
        raise ValueError("Invalid or nonfinite appearance statistic")
    return {name: float(np.mean([row[name] for row in rows])) for name in STATS}


def distance(means: dict[str, float]) -> float:
    return float(sum(((means[name] - _target[name]) / _scale[name]) ** 2 for name in STATS))


def evaluate_candidate(item: tuple[str, int, dict[str, float]]) -> dict[str, object]:
    arm, index, params = item
    means = aggregate(_calibration_images, params, arm, index, FIT_DRAWS)
    return {"index": index, "distance": distance(means), "means": means, "params": params}


def validate(images: list[Image.Image], params: dict[str, float] | None, arm: str) -> dict[str, object]:
    per_draw = []
    for draw in range(VALIDATION_DRAWS):
        means = aggregate(images, params, arm, -1, 1) if params is None else aggregate(
            images, params, f"{arm}_validation_{draw}", -1, 1
        )
        per_draw.append({"draw": draw, "distance": distance(means), "means": means})
    distances = [row["distance"] for row in per_draw]
    return {
        "per_draw": per_draw,
        "distance_mean": float(np.mean(distances)),
        "distance_min": float(np.min(distances)),
        "distance_max": float(np.max(distances)),
    }


def main() -> None:
    global _calibration_images, _target, _scale
    node_check()
    manifest = pd.read_csv(MANIFEST)
    source = manifest[(manifest.domain == "BRSET") & (manifest.role == "fit")].copy()
    target_rows = manifest[(manifest.domain == "mBRSET") & (manifest.role == "fit")]
    if len(source) != 11372 or len(target_rows) != 3402:
        raise ValueError("Unexpected full-pool training counts")
    audit = json.loads(AUDIT.read_text())
    if audit["scope"] != "original training split only; no validation or test images":
        raise ValueError("Appearance audit scope changed")
    _target = {name: float(audit["by_domain"]["mBRSET"][name]["mean"]) for name in STATS}
    _scale = {name: float(audit["by_domain"]["mBRSET"][name]["sample_sd"]) for name in STATS}
    if any(value <= 0 or not np.isfinite(value) for value in _scale.values()):
        raise ValueError("Invalid target scale")

    calibration_rows, validation_rows, selection_hash = select_patient_disjoint(source)
    _calibration_images = [load_image(path) for path in calibration_rows.image_path]
    validation_images = [load_image(path) for path in validation_rows.image_path]

    arms = {}
    context = mp.get_context("fork")
    for arm, overlay_free in (("full", False), ("overlay_free", True)):
        candidates = candidate_list(overlay_free)
        tasks = [(arm, index, params) for index, params in enumerate(candidates)]
        with context.Pool(processes=min(8, os.cpu_count() or 1)) as pool:
            scored = pool.map(evaluate_candidate, tasks)
        scored.sort(key=lambda row: (row["distance"], row["index"]))
        best = scored[0]
        arms[arm] = {
            "candidate_count": len(candidates),
            "random_candidate_count": RANDOM_CANDIDATES,
            "selected_candidate_index": best["index"],
            "calibration_distance": best["distance"],
            "calibration_means": best["means"],
            "params": best["params"],
            "runner_up_distances": [row["distance"] for row in scored[1:6]],
            "validation": validate(validation_images, best["params"], arm),
        }

    result = {
        "status": "complete",
        "scope": "original training split only; target statistics use all target fit images; source calibration and validation are patient-disjoint",
        "preprocessing": "PIL bilinear 560x560 resize then 512x512 center crop; transformation follows crop",
        "objective": "sum over five statistics of squared transformed-source minus target mean divided by target sample SD",
        "statistics": STATS,
        "source_counts": {"available": len(source), "calibration": CALIBRATION_N, "validation": VALIDATION_N},
        "target_count": len(target_rows),
        "fit_draws_per_candidate_image": FIT_DRAWS,
        "validation_draws": VALIDATION_DRAWS,
        "selection_seed": SELECTION_SEED,
        "selected_identity_sha256": selection_hash,
        "target_means": _target,
        "target_scales": _scale,
        "identity_validation": validate(validation_images, None, "identity"),
        "historical_frozen_validation": validate(
            validation_images, json.loads((ROOT / "results/fitted_degradation_params.json").read_text()), "historical"
        ),
        "arms": arms,
        "ranges": RANGES,
        "input_sha256": {
            "manifest": sha(MANIFEST),
            "appearance_audit": sha(AUDIT),
            "appearance_operators": sha(ROOT / "scripts/55_degradation_ops.py"),
            "historical_parameters": sha(ROOT / "results/fitted_degradation_params.json"),
            "analysis": sha(Path(__file__)),
        },
        "limitations": [
            "The objective matches five global appearance means and is not a physical camera model.",
            "Source calibration and validation subsets are small compute-conscious samples.",
            "Patient-disjoint validation checks transformation fitting but does not establish diagnostic preservation.",
            "No mBRSET validation or test image is accessed.",
        ],
    }
    temporary = OUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, OUT)
    print(json.dumps({"output": str(OUT), "arms": arms, "identity": result["identity_validation"]}, indent=2))


if __name__ == "__main__":
    main()
