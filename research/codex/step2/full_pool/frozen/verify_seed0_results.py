"""Independently recompute seed-0 assessment metrics and provenance on Slurm CPU."""
import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np

from baseline_core import LABELS, metrics, sha

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
WORK = ROOT / "research/codex/step2/full_pool"


def read(path):
    return json.loads(Path(path).read_text())


def close(a, b, tolerance=1e-12):
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tolerance


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Use an allocated Slurm compute node")

    combined = read(WORK / "baseline_assessment_seed0.json")
    expected_protocol = sha(WORK / "protocol/v1.json")
    expected_manifest = read(WORK / "protocol/v1.json")["manifest_sha256"]
    arrays = {}
    checks = []

    for arm in ["B0", "B1", "B2"]:
        folder = WORK / "runs" / f"{arm}_seed0"
        complete = read(folder / "complete.json")
        selection = read(folder / "selection_summary.json")
        assessment = read(folder / "assessment_summary.json")
        contract = read(folder / "contract.json")
        archive = np.load(folder / "assessment_predictions.npz", allow_pickle=False)
        arrays[arm] = archive

        thresholds = selection["metadata"]["thresholds"]
        recomputed = metrics(archive["y_true"], archive["probabilities"], thresholds)
        for label in LABELS:
            for name in ["threshold", "f1_positive", "f1_class_macro", "auroc",
                         "average_precision", "precision", "sensitivity", "specificity"]:
                if not close(recomputed[label][name], assessment["metrics"][label][name]):
                    raise AssertionError(f"{arm} {label} {name} mismatch")
            if recomputed[label]["confusion_matrix"] != assessment["metrics"][label]["confusion_matrix"]:
                raise AssertionError(f"{arm} {label} confusion mismatch")
        if recomputed != combined["observed"][arm]:
            raise AssertionError(f"{arm} combined summary mismatch")

        meta = selection["metadata"]
        assert complete["contract"] == contract
        assert contract["seed"] == 0 and contract["arm"] == arm
        assert contract["protocol_sha256"] == expected_protocol
        assert contract["manifest_sha256"] == expected_manifest
        assert meta["checkpoint_sha256"] == sha(folder / "best.pth")
        assert combined["prediction_sha256"][arm] == sha(folder / "assessment_predictions.npz")
        assert archive["role"].tolist() == ["assessment"] * 732
        assert len(np.unique(archive["file_id"])) == 732
        assert len(np.unique(archive["patient_id"])) == 193
        assert archive["y_true"].sum(axis=0).tolist() == [159.0, 63.0]
        checks.append({
            "arm": arm,
            "checkpoint_update": meta["checkpoint_update"],
            "thresholds": thresholds,
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "prediction_sha256": combined["prediction_sha256"][arm],
            "recomputed_metrics_match": True,
        })

    reference = arrays["B0"]
    for arm in ["B1", "B2"]:
        for field in ["file_id", "patient_id", "y_true", "role"]:
            assert np.array_equal(reference[field], arrays[arm][field]), f"assessment cohort differs: {arm} {field}"

    result = {
        "pass": True,
        "scope": "Independent deterministic recomputation of saved seed-0 predictions and provenance; bootstrap intervals were not independently regenerated.",
        "assessment_images": 732,
        "assessment_patients": 193,
        "positive_images": {"DR": 159, "ME": 63},
        "same_ordered_assessment_cohort_all_arms": True,
        "checks": checks,
        "combined_summary_sha256": sha(WORK / "baseline_assessment_seed0.json"),
    }
    output = WORK / "seed0_independent_verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
