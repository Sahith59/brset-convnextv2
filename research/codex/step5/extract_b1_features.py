"""Extract frozen B1 pooled and spatial features on an allocated A40."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import timm


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "research/codex/step5"
BASE = ROOT / "research/codex/step2/full_pool/frozen"
CONFIG = WORK / "feature_v1.json"
sys.path.insert(0, str(BASE))
import baseline_core as core  # noqa: E402
import train_baseline as baseline  # noqa: E402


FLIPS = [(), (-1,), (-2,), (-2, -1)]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> tuple[dict, object, object, object]:
    cfg = json.loads(CONFIG.read_text())
    manifest_path = ROOT / cfg["manifest"]
    rows = core.validate_manifest(manifest_path, cfg["manifest_sha256"])
    fit = rows[(rows.role == "fit") & rows.domain.isin(["BRSET", "mBRSET"])].reset_index(drop=True)
    selection = rows[(rows.role == "selection") & (rows.domain == "mBRSET")].reset_index(drop=True)
    if len(fit) != cfg["fit_images"] or len(selection) != cfg["selection_images"]:
        raise ValueError("Feature cohort counts changed")
    if (rows.role == "assessment").sum() and cfg["assessment_images_allowed"] != 0:
        raise ValueError("Assessment contract changed")
    checkpoint = Path(cfg["checkpoint"])
    predictions = Path(cfg["selection_predictions"])
    if sha(checkpoint) != cfg["checkpoint_sha256"] or sha(predictions) != cfg["selection_predictions_sha256"]:
        raise ValueError("Frozen B1 artifact hash mismatch")
    saved = np.load(predictions, allow_pickle=False)
    if not np.array_equal(saved["file_id"].astype(str), selection.file.to_numpy(dtype=str)):
        raise ValueError("Saved B1 selection order changed")
    if not np.array_equal(saved["y_true"], selection[core.LABELS].to_numpy(dtype=np.float32)):
        raise ValueError("Saved B1 labels changed")
    return cfg, fit, selection, saved


def model_from_checkpoint(cfg: dict):
    state = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model = timm.create_model(cfg["model"], pretrained=False, num_classes=2, drop_path_rate=.3).cuda().eval()
    model.load_state_dict(state["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def batch_features(model, images, flips):
    spatial, pooled, logits = [], [], []
    images = images.cuda(non_blocking=True)
    for dims in flips:
        view = torch.flip(images, dims) if dims else images
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fmap = model.forward_features(view)
            vector = model.forward_head(fmap, pre_logits=True)
            output = model.forward_head(fmap, pre_logits=False)
        spatial.append(fmap.to(torch.float16).cpu().numpy())
        pooled.append(vector.float().cpu().numpy())
        logits.append(output.float().cpu().numpy())
    return np.stack(spatial, axis=1), np.stack(pooled, axis=1), np.stack(logits, axis=1)


def reproduction(saved_logits, observed_logits, cfg):
    logit_error = float(np.max(np.abs(saved_logits - observed_logits)))
    saved_probability = 1 / (1 + np.exp(-saved_logits.astype(np.float64)))
    observed_probability = 1 / (1 + np.exp(-observed_logits.astype(np.float64)))
    probability_error = float(np.max(np.abs(saved_probability.mean(1) - observed_probability.mean(1))))
    return {
        "logit_max_abs_error": logit_error,
        "probability_max_abs_error": probability_error,
        "logit_tolerance": cfg["logit_max_abs_tolerance"],
        "probability_tolerance": cfg["probability_max_abs_tolerance"],
        "pass": logit_error <= cfg["logit_max_abs_tolerance"] and probability_error <= cfg["probability_max_abs_tolerance"],
    }


def smoke(args) -> None:
    cfg, _, selection, saved = load_contract()
    model = model_from_checkpoint(cfg)
    rows = selection.iloc[: args.smoke_images].reset_index(drop=True)
    loader = baseline.loader(rows, False, 0, args.workers)
    all_logits = []
    shapes = None
    started = time.monotonic()
    for images, _, _ in loader:
        spatial, pooled, logits = batch_features(model, images, FLIPS)
        shapes = {"spatial": list(spatial.shape[2:]), "pooled": list(pooled.shape[2:])}
        all_logits.append(logits)
    observed = np.concatenate(all_logits)
    check = reproduction(saved["view_logits"][: len(rows)], observed, cfg)
    result = {
        "status": "pass" if check["pass"] else "fail",
        "job_id": os.environ["SLURM_JOB_ID"],
        "images": len(rows),
        "shapes_per_image": shapes,
        "reproduction": check,
        "seconds": time.monotonic() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "assessment_accessed": False,
        "config_sha256": sha(CONFIG),
        "extractor_sha256": sha(Path(__file__)),
    }
    (WORK / "feature_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not check["pass"]:
        raise SystemExit(1)


def full(args) -> None:
    cfg, fit, selection, saved = load_contract()
    smoke_result = json.loads((WORK / "feature_smoke.json").read_text())
    if (smoke_result["status"] != "pass"
            or smoke_result["config_sha256"] != sha(CONFIG)
            or smoke_result["extractor_sha256"] != sha(Path(__file__))):
        raise ValueError("Feature smoke gate failed or configuration changed")
    model = model_from_checkpoint(cfg)
    output = Path(cfg["private_output"])
    if output.exists():
        raise FileExistsError(f"Private feature output already exists: {output}")
    output.mkdir(parents=True)
    started = time.monotonic()
    shapes = {}
    observed_selection_logits = None

    for role, rows, flips in [("fit", fit, [()]), ("selection", selection, FLIPS)]:
        loader = baseline.loader(rows, False, 0, args.workers)
        spatial_map = pooled_map = None
        logits_parts = []
        cursor = 0
        for images, _, indices in loader:
            if indices.tolist() != list(range(cursor, cursor + len(indices))):
                raise ValueError(f"{role} identity/order mismatch during extraction")
            spatial, pooled, logits = batch_features(model, images, flips)
            if spatial_map is None:
                spatial_map = np.lib.format.open_memmap(
                    output / f"{role}_spatial.npy", mode="w+", dtype=np.float16,
                    shape=(len(rows), len(flips), *spatial.shape[2:]))
                pooled_map = np.lib.format.open_memmap(
                    output / f"{role}_pooled.npy", mode="w+", dtype=np.float32,
                    shape=(len(rows), len(flips), *pooled.shape[2:]))
                shapes[role] = {"spatial": list(spatial_map.shape), "pooled": list(pooled_map.shape)}
            spatial_map[cursor:cursor + len(images)] = spatial
            pooled_map[cursor:cursor + len(images)] = pooled
            logits_parts.append(logits)
            cursor += len(images)
        spatial_map.flush(); pooled_map.flush()
        if cursor != len(rows):
            raise ValueError(f"{role} extraction incomplete")
        if role == "selection":
            observed_selection_logits = np.concatenate(logits_parts)

    check = reproduction(saved["view_logits"], observed_selection_logits, cfg)
    if not check["pass"]:
        shutil.rmtree(output)
        raise ValueError(f"Full extraction failed B1 reproduction: {check}")
    np.savez_compressed(output / "rows.npz",
        fit_y=fit[core.LABELS].to_numpy(dtype=np.float32),
        fit_domain=fit.domain.to_numpy(dtype=str),
        fit_patient=fit.patient_key.to_numpy(dtype=str),
        selection_y=selection[core.LABELS].to_numpy(dtype=np.float32),
        selection_patient=selection.patient_key.to_numpy(dtype=str))
    state = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)["model"]
    np.savez(output / "head.npz",
             weight=state["head.fc.weight"].float().numpy(),
             bias=state["head.fc.bias"].float().numpy())
    files = [output / name for name in ["fit_spatial.npy", "fit_pooled.npy", "selection_spatial.npy", "selection_pooled.npy", "rows.npz", "head.npz"]]
    result = {
        "status": "pass",
        "job_id": os.environ["SLURM_JOB_ID"],
        "fit_images": len(fit),
        "selection_images": len(selection),
        "assessment_accessed": False,
        "shapes": shapes,
        "reproduction": check,
        "seconds": time.monotonic() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "private_artifact_sha256": {path.name: sha(path) for path in files},
        "config_sha256": sha(CONFIG),
        "extractor_sha256": sha(Path(__file__)),
        "privacy": "Features and row-level metadata remain in private shared storage; this summary contains aggregate shapes and hashes only.",
    }
    (WORK / "feature_extraction_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "full"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke-images", type=int, default=32)
    args = parser.parse_args()
    baseline.node_check(True)
    baseline.setup(0)
    smoke(args) if args.mode == "smoke" else full(args)


if __name__ == "__main__":
    main()
