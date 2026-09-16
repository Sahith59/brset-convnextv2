"""Aggregate failure-transition and patient-bootstrap analysis for Step 5."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "research/codex/step5"
BASE = ROOT / "research/codex/step2/full_pool/frozen"
PRIVATE = Path("/data/users3/sthummala2/brset-codex/step5/b1_features")
SOURCE = Path("/data/users3/sthummala2/brset-codex/step5/source_only_seed0")
B1 = Path("/data/users3/sthummala2/brset-codex/step2-full-pool/B1_seed0")
LABELS = ["diabetic_retinopathy", "macular_edema"]
sys.path.insert(0, str(BASE))
import baseline_core as core  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transitions(truth, before_probability, before_threshold, after_probability, after_threshold):
    truth = truth.astype(bool)
    before = before_probability >= before_threshold
    after = after_probability >= after_threshold
    return {
        "errors_fixed": int((before != truth)[after == truth].sum()),
        "new_errors": int((before == truth)[after != truth].sum()),
        "false_negatives_fixed": int((truth & ~before & after).sum()),
        "new_false_negatives": int((truth & before & ~after).sum()),
        "false_positives_fixed": int((~truth & before & ~after).sum()),
        "new_false_positives": int((~truth & ~before & after).sum()),
    }


def metrics(truth, probability, threshold):
    prediction = probability >= threshold
    return {
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "auroc": float(roc_auc_score(truth, probability)),
        "average_precision": float(average_precision_score(truth, probability)),
    }


def patient_bootstrap(spatial_records, private_records, patients, chosen_k, draws=2000):
    unique = np.unique(patients)
    groups = {patient: np.flatnonzero(patients == patient) for patient in unique}
    rng = np.random.default_rng(20260915)
    output = {label: {metric: [] for metric in ["f1", "auroc", "average_precision"]} for label in LABELS}
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([groups[patient] for patient in sampled])
        for label_index, label in enumerate(LABELS):
            differences = {metric: [] for metric in output[label]}
            for public, private in zip(spatial_records, private_records):
                truth = private["truth"][indices, label_index]
                before = private["G0"][indices, label_index]
                after = private[f"G1_k{chosen_k}"][indices, label_index]
                before_threshold = public["heads"]["G0"]["thresholds"][label_index]
                after_threshold = public["heads"][f"G1_k{chosen_k}"]["thresholds"][label_index]
                if len(np.unique(truth)) != 2:
                    continue
                first = metrics(truth, before, before_threshold)
                second = metrics(truth, after, after_threshold)
                for metric in differences:
                    differences[metric].append(second[metric] - first[metric])
            for metric, values in differences.items():
                output[label][metric].append(float(np.mean(values)))
    return {label: {metric: {
                "point_mean": float(np.mean(values)),
                "patient_bootstrap_95pct": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
            } for metric, values in record.items()} for label, record in output.items()}


def load_spatial_private(seed):
    data = np.load(PRIVATE / f"spatial_heads_seed{seed}.npz", allow_pickle=False)
    probabilities = json.loads(str(data["probabilities_json"]))
    return {"truth": data["selection_y"], **{name: np.asarray(value) for name, value in probabilities.items()}}


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Allocated node required")
    spatial_summary = json.loads((WORK / "spatial_diagnostic_summary.json").read_text())
    gfp = json.loads((WORK / "gfp_summary.json").read_text())
    source_summary = json.loads((SOURCE / "assessment_summary.json").read_text())
    b1_summary = json.loads((B1 / "assessment_summary.json").read_text())
    source = np.load(SOURCE / "assessment_predictions.npz", allow_pickle=False)
    b1 = np.load(B1 / "assessment_predictions.npz", allow_pickle=False)
    if not np.array_equal(source["file_id"], b1["file_id"]) or not np.array_equal(source["y_true"], b1["y_true"]):
        raise ValueError("Source-only and B1 assessment cohorts differ")

    source_vs_b1 = {}
    for index, label in enumerate(LABELS):
        source_vs_b1[label] = {
            "source_metrics": source_summary["metrics"][label],
            "joint_b1_metrics": b1_summary["metrics"][label],
            "transitions_source_to_joint_b1": transitions(
                source["y_true"][:, index], source["probabilities"][:, index],
                source_summary["metadata"]["thresholds"][index], b1["probabilities"][:, index],
                b1_summary["metadata"]["thresholds"][index]),
            "warning": "Post-hoc historical assessment description; training resources differ and these transitions must not select the next method.",
        }

    chosen_k = spatial_summary["chosen_k"]
    spatial_records = [json.loads((WORK / f"spatial_head_seed{seed}.json").read_text()) for seed in range(3)]
    private_records = [load_spatial_private(seed) for seed in range(3)]
    rows = np.load(PRIVATE / "rows.npz", allow_pickle=False)
    transition_by_seed = {}
    for seed, (public, private) in enumerate(zip(spatial_records, private_records)):
        transition_by_seed[str(seed)] = {}
        for index, label in enumerate(LABELS):
            transition_by_seed[str(seed)][label] = transitions(
                private["truth"][:, index], private["G0"][:, index], public["heads"]["G0"]["thresholds"][index],
                private[f"G1_k{chosen_k}"][:, index],
                public["heads"][f"G1_k{chosen_k}"]["thresholds"][index])

    bootstrap = patient_bootstrap(spatial_records, private_records, rows["selection_patient"], chosen_k)
    result = {
        "status": "pass",
        "job_id": os.environ["SLURM_JOB_ID"],
        "source_only_vs_joint_b1_assessment": source_vs_b1,
        "gfp_failure": {
            "selected_group_size": gfp["selected"]["group_size"],
            "retained_features_dr_me": gfp["selected"]["retained_features"],
            "delta_from_unmasked_b1": gfp["delta_from_unmasked_b1"],
            "engineering_screen_pass": gfp["engineering_screen_pass"],
        },
        "spatial_failure": {
            "chosen_k": chosen_k,
            "aggregate": spatial_summary["aggregate"],
            "transitions_by_seed": transition_by_seed,
            "patient_bootstrap": bootstrap,
            "engineering_screen_pass": spatial_summary["engineering_screen_pass"],
        },
        "assessment_boundary": "Assessment transition analysis is descriptive only; all mechanism decisions use mBRSET validation diagnostics.",
        "input_sha256": {
            "source_predictions": sha(SOURCE / "assessment_predictions.npz"),
            "b1_predictions": sha(B1 / "assessment_predictions.npz"),
            "gfp_summary": sha(WORK / "gfp_summary.json"),
            "spatial_summary": sha(WORK / "spatial_diagnostic_summary.json"),
        },
        "privacy": "Aggregate counts and intervals only; no image, filename or patient identifier is exported.",
    }
    (WORK / "diagnostic_error_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
