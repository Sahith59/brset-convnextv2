"""Allocated-CPU evidence gate for appearance-conditioned Step-5 modeling."""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "research/codex/step5"
MANIFEST = ROOT / "research/codex/step2/full_pool/protocol/full_pool_manifest.csv"
PREDICTIONS = [
    ROOT / f"research/codex/step2/full_pool/runs/B1_seed{seed}/selection_predictions.npz"
    for seed in range(3)
]
OPS_PATH = ROOT / "scripts/55_degradation_ops.py"
PROTOCOL = WORK / "EVIDENCE_GATE_PROTOCOL.md"
OUTPUT = WORK / "appearance_error_gate.json"
EXPECTED_MANIFEST_SHA256 = "04181fe8ba018839181ecade097228856622cab8d428b77a22635b881c5bc3fc"
LABELS = ["diabetic_retinopathy", "macular_edema"]
STATS = ["sharpness", "brightness", "contrast", "falloff", "saturation"]

spec = importlib.util.spec_from_file_location("step5_appearance_ops", OPS_PATH)
OPS = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(OPS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def node_check() -> None:
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Image decoding and numerical analysis require an allocated Slurm node")


def measure(record: tuple[str, str]) -> tuple[str, list[float]]:
    file_id, path = record
    with Image.open(path) as raw:
        image = raw.convert("RGB").resize((560, 560), Image.Resampling.BILINEAR)
    image = image.crop((24, 24, 536, 536))
    values = OPS.image_stats(image)
    if values is None or not all(np.isfinite(values[name]) for name in STATS):
        raise ValueError("Invalid appearance statistics")
    return file_id, [float(values[name]) for name in STATS]


def bh_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for rank_index in range(len(values) - 1, -1, -1):
        original = order[rank_index]
        rank = rank_index + 1
        running = min(running, values[original] * len(values) / rank)
        adjusted[original] = running
    return adjusted.tolist()


def grouped_oof(x_base: np.ndarray, x_full: np.ndarray, outcome: np.ndarray,
                groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    base_pred = np.full(len(outcome), np.nan)
    full_pred = np.full(len(outcome), np.nan)
    folds = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train, valid) in enumerate(splitter.split(x_full, outcome, groups)):
        # The fitting portion needs both outcomes. A held-out fold may contain
        # no errors (especially in the positive-only ME analysis); that is a
        # valid out-of-fold prediction set because AUROC is computed only after
        # all five held-out folds are concatenated.
        if len(np.unique(outcome[train])) != 2:
            raise ValueError(f"Fold {fold} training portion lacks both error classes")
        base = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=20260915))
        full = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=20260915))
        base.fit(x_base[train], outcome[train])
        full.fit(x_full[train], outcome[train])
        base_pred[valid] = base.predict_proba(x_base[valid])[:, 1]
        full_pred[valid] = full.predict_proba(x_full[valid])[:, 1]
        folds.append({"fold": fold, "train_observations": int(len(train)),
                      "validation_observations": int(len(valid)),
                      "validation_patients": int(len(np.unique(groups[valid])))})
    if not np.isfinite(base_pred).all() or not np.isfinite(full_pred).all():
        raise AssertionError("Incomplete out-of-fold predictions")
    return base_pred, full_pred, folds


def patient_bootstrap(outcome: np.ndarray, base_pred: np.ndarray, full_pred: np.ndarray,
                      groups: np.ndarray, seed: int) -> dict:
    patients = np.unique(groups)
    by_patient = {patient: np.flatnonzero(groups == patient) for patient in patients}
    rng = np.random.default_rng(seed)
    base_values, full_values, differences = [], [], []
    attempts = 0
    while len(differences) < 2000 and attempts < 10000:
        attempts += 1
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([by_patient[patient] for patient in sampled])
        if len(np.unique(outcome[indices])) != 2:
            continue
        base_auc = float(roc_auc_score(outcome[indices], base_pred[indices]))
        full_auc = float(roc_auc_score(outcome[indices], full_pred[indices]))
        base_values.append(base_auc); full_values.append(full_auc); differences.append(full_auc - base_auc)
    if len(differences) != 2000:
        raise RuntimeError("Could not obtain 2,000 valid patient-bootstrap draws")
    interval = lambda values: [float(x) for x in np.percentile(values, [2.5, 97.5])]
    return {"draws": 2000, "patients": int(len(patients)),
            "base_auc_point": float(roc_auc_score(outcome, base_pred)),
            "appearance_auc_point": float(roc_auc_score(outcome, full_pred)),
            "difference_point": float(roc_auc_score(outcome, full_pred) - roc_auc_score(outcome, base_pred)),
            "base_auc_95pct": interval(base_values), "appearance_auc_95pct": interval(full_values),
            "difference_95pct": interval(differences)}


def main() -> None:
    node_check()
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Manifest hash mismatch")
    manifest = pd.read_csv(MANIFEST)
    selection = manifest[(manifest.domain == "mBRSET") & (manifest.role == "selection")].copy()
    if len(selection) != 725 or selection.patient_key.nunique() != 193:
        raise ValueError("Unexpected mBRSET validation cohort")
    selection = selection.set_index("file", drop=False)

    archives = []
    for seed, path in enumerate(PREDICTIONS):
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata_json"].item()))
        files = data["file_id"].astype(str)
        if len(files) != 725 or len(np.unique(files)) != 725:
            raise ValueError(f"Seed {seed} prediction identity failure")
        rows = selection.loc[files]
        y_true = data["y_true"].astype(int)
        expected_y = rows[LABELS].to_numpy(dtype=int)
        if not np.array_equal(y_true, expected_y):
            raise ValueError(f"Seed {seed} label/order mismatch")
        if not np.array_equal(data["patient_id"].astype(str), rows.patient_key.astype(str).to_numpy()):
            raise ValueError(f"Seed {seed} patient/order mismatch")
        probabilities = data["probabilities"].astype(float)
        thresholds = np.asarray(metadata["thresholds"], dtype=float)
        if probabilities.shape != (725, 2) or thresholds.shape != (2,) or not np.isfinite(probabilities).all():
            raise ValueError(f"Seed {seed} prediction shape/value failure")
        archives.append({"seed": seed, "files": files, "patients": rows.patient_key.astype(str).to_numpy(),
                         "y": y_true, "p": probabilities, "thresholds": thresholds,
                         "sha256": sha256(path)})
    for archive in archives[1:]:
        for key in ("files", "patients", "y"):
            if not np.array_equal(archive[key], archives[0][key]):
                raise ValueError(f"Cross-seed {key} mismatch")

    records = [(file_id, str(selection.loc[file_id, "image_path"])) for file_id in archives[0]["files"]]
    measured = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        for file_id, values in pool.map(measure, records, chunksize=8):
            measured[file_id] = values
    stats = np.asarray([measured[file_id] for file_id in archives[0]["files"]], dtype=float)
    if stats.shape != (725, 5) or not np.isfinite(stats).all():
        raise AssertionError("Appearance measurement failure")

    seed_eye = np.eye(3, dtype=float)[:, :2]
    results = {}
    for label_index, label in enumerate(LABELS):
        truth = archives[0]["y"][:, label_index]
        errors, nll = [], []
        for archive in archives:
            probability = np.clip(archive["p"][:, label_index], 1e-7, 1 - 1e-7)
            prediction = probability >= archive["thresholds"][label_index]
            errors.append((prediction != truth).astype(int))
            nll.append(-(truth * np.log(probability) + (1 - truth) * np.log(1 - probability)))
        errors = np.stack(errors)
        nll = np.stack(nll)

        correlations = []
        raw_p = []
        for stat_index, name in enumerate(STATS):
            rho, p_value = spearmanr(stats[:, stat_index], nll.mean(axis=0))
            correlations.append({"statistic": name, "spearman_rho": float(rho),
                                 "p_value_unadjusted": float(p_value)})
            raw_p.append(float(p_value))
        for record, adjusted in zip(correlations, bh_adjust(raw_p)):
            record["p_value_bh_within_label"] = float(adjusted)

        outcome = errors.reshape(-1)
        tiled_truth = np.tile(truth, 3)
        tiled_stats = np.tile(stats, (3, 1))
        tiled_groups = np.tile(archives[0]["patients"], 3)
        seed_features = np.repeat(seed_eye, 725, axis=0)
        base_x = np.column_stack([tiled_truth, seed_features])
        full_x = np.column_stack([base_x, tiled_stats])
        base_pred, full_pred, folds = grouped_oof(base_x, full_x, outcome, tiled_groups)
        overall = patient_bootstrap(outcome, base_pred, full_pred, tiled_groups, 20260915 + label_index)

        positive = tiled_truth == 1
        positive_result = {"computable": False, "positive_observations": int(positive.sum()),
                           "positive_patients": int(len(np.unique(tiled_groups[positive]))),
                           "false_negatives": int(outcome[positive].sum())}
        if len(np.unique(outcome[positive])) == 2:
            pos_base, pos_full, pos_folds = grouped_oof(
                seed_features[positive], np.column_stack([seed_features[positive], tiled_stats[positive]]),
                outcome[positive], tiled_groups[positive])
            positive_result.update({"computable": True, "folds": pos_folds,
                                    "bootstrap": patient_bootstrap(outcome[positive], pos_base, pos_full,
                                                                   tiled_groups[positive], 20261015 + label_index)})
        results[label] = {"images": 725, "patients": 193, "seed_observations": int(len(outcome)),
                          "errors": int(outcome.sum()), "error_rate": float(outcome.mean()),
                          "appearance_nll_correlations": correlations, "overall_error_model": overall,
                          "folds": folds, "positive_false_negative_model": positive_result}

    primary = []
    for label in LABELS:
        record = results[label]
        primary.append(record["overall_error_model"]["difference_95pct"][0] > 0)
    other_ok = []
    for label in LABELS:
        other_ok.append(results[label]["overall_error_model"]["difference_point"] >= -0.01)
    positive_signal = any(
        results[label]["positive_false_negative_model"]["computable"]
        and results[label]["positive_false_negative_model"]["bootstrap"]["appearance_auc_point"] > 0.60
        for label in LABELS
    )
    retain = bool(positive_signal and ((primary[0] and other_ok[1]) or (primary[1] and other_ok[0])))
    result = {"status": "complete", "date_utc": "2026-09-15",
              "scope": "mBRSET validation only; no test access and no model fitting",
              "cohort": {"images": 725, "patients": 193, "prediction_seeds": [0, 1, 2]},
              "preprocessing": "bilinear 560x560 resize then center 512x512 crop",
              "inputs": {"manifest_sha256": sha256(MANIFEST),
                         "prediction_sha256": {f"seed{a['seed']}": a["sha256"] for a in archives},
                         "appearance_ops_sha256": sha256(OPS_PATH),
                         "protocol_sha256": sha256(PROTOCOL), "analysis_sha256": sha256(Path(__file__))},
              "labels": results,
              "decision": {"retain_appearance_conditioned_transition": retain,
                           "positive_false_negative_signal": bool(positive_signal),
                           "next": "freeze W1/W2/W3" if retain else "pivot to label-specific patch/shared-private mechanism"},
              "limits": ["Observational association does not establish a camera-effect cause.",
                         "Validation data have already served model selection and remain development evidence.",
                         "Bootstrap conditions on fixed out-of-fold predictions and does not cover all design or model-training uncertainty."],
              "privacy": "Aggregate output only; no image, file or patient identifier is stored."}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "decision": result["decision"],
                      "summary": {label: {"error_rate": results[label]["error_rate"],
                          "appearance_error_auc_delta": results[label]["overall_error_model"]["difference_point"],
                          "appearance_error_auc_delta_95pct": results[label]["overall_error_model"]["difference_95pct"],
                          "positive_fn_appearance_auc": results[label]["positive_false_negative_model"].get("bootstrap", {}).get("appearance_auc_point")}
                                  for label in LABELS}}, indent=2))


if __name__ == "__main__":
    main()
