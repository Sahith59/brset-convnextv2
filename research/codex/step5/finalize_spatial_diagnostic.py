"""Select common top-k and summarize three frozen-head seeds."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "research/codex/step5"
LABELS = ["diabetic_retinopathy", "macular_edema"]
KS = [1, 4, 8, 16, 32]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Allocated node required")
    paths = [WORK / f"spatial_head_seed{seed}.json" for seed in range(3)]
    records = [json.loads(path.read_text()) for path in paths]
    if any(record["status"] != "pass" or record["assessment_accessed"] for record in records):
        raise ValueError("Seed result gate failed")
    script_hashes = {record["script_sha256"] for record in records}
    feature_hashes = {record["feature_summary_sha256"] for record in records}
    if len(script_hashes) != 1 or len(feature_hashes) != 1:
        raise ValueError("Seed provenance mismatch")

    selection_scores = {}
    for k in KS:
        name = f"G1_k{k}"
        values = [np.mean([record["heads"][name]["metrics"][label]["auroc"] for label in LABELS])
                  for record in records]
        selection_scores[k] = float(np.mean(values))
    best = max(selection_scores.values())
    chosen_k = max(k for k, value in selection_scores.items() if np.isclose(value, best, atol=1e-12, rtol=0))

    aggregate = {}
    gate_values = {"f1": [], "auroc": []}
    for label in LABELS:
        aggregate[label] = {}
        for metric in ["f1_positive", "auroc", "average_precision", "sensitivity", "specificity"]:
            global_values = np.asarray([record["heads"]["G0"]["metrics"][label][metric] for record in records])
            spatial_values = np.asarray([record["heads"][f"G1_k{chosen_k}"]["metrics"][label][metric]
                                         for record in records])
            differences = spatial_values - global_values
            aggregate[label][metric] = {
                "G0_mean": float(global_values.mean()),
                "G1_mean": float(spatial_values.mean()),
                "difference_mean": float(differences.mean()),
                "differences_by_seed": differences.tolist(),
                "G0_sample_sd": float(global_values.std(ddof=1)),
                "G1_sample_sd": float(spatial_values.std(ddof=1)),
            }
        gate_values["f1"].append(aggregate[label]["f1_positive"]["difference_mean"])
        gate_values["auroc"].append(aggregate[label]["auroc"]["difference_mean"])
    f1, auc = gate_values["f1"], gate_values["auroc"]
    gate = ((f1[0] >= .01 and f1[1] >= -.01) or (f1[1] >= .01 and f1[0] >= -.01)) and min(auc) >= -.01
    result = {
        "status": "pass",
        "job_id": os.environ["SLURM_JOB_ID"],
        "head_seeds": [0, 1, 2],
        "chosen_k": chosen_k,
        "k_selection_mean_dr_me_auroc": {str(k): value for k, value in selection_scores.items()},
        "aggregate": aggregate,
        "engineering_screen_pass": bool(gate),
        "assessment_accessed": False,
        "input_sha256": {path.name: sha(path) for path in paths},
        "interpretation": "Three-seed frozen-backbone development diagnostic. Passing supports full-model patch controls; it does not establish lesion localization or novelty.",
    }
    (WORK / "spatial_diagnostic_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
