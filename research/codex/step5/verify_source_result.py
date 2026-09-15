"""Independent aggregate verification for strict source-only seed 0."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "research/codex/step5"
RUN = Path("/data/users3/sthummala2/brset-codex/step5/source_only_seed0")
BASE = ROOT / "research/codex/step2/full_pool/frozen"
sys.path.insert(0, str(BASE))
import baseline_core as core  # noqa: E402


def close(left, right, path="root") -> None:
    if isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError(f"Key mismatch at {path}")
        for key in left:
            close(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if len(left) != len(right):
            raise AssertionError(f"Length mismatch at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            close(a, b, f"{path}[{index}]")
    elif isinstance(left, float):
        if not np.isclose(left, right, atol=1e-12, rtol=0):
            raise AssertionError(f"Float mismatch at {path}: {left} != {right}")
    elif left != right:
        raise AssertionError(f"Mismatch at {path}: {left} != {right}")


def load_predictions(path: Path):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    return data, metadata


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Allocated node required")
    protocol = json.loads((WORK / "source_only_v1.json").read_text())
    manifest = core.validate_manifest(WORK / "private/source_only_manifest.csv",
                                      protocol["private_manifest_sha256"])
    complete = json.loads((RUN / "complete.json").read_text())
    selection_summary = json.loads((RUN / "selection_summary.json").read_text())
    assessment_summary = json.loads((RUN / "assessment_summary.json").read_text())
    selection, selection_meta = load_predictions(RUN / "selection_predictions.npz")
    assessment, assessment_meta = load_predictions(RUN / "assessment_predictions.npz")

    if not complete["training_complete"] or complete["smoke"] or complete["updates"] != 5775:
        raise AssertionError("Training completion contract failed")
    if complete["exposure"][1] != [0, 0, 0]:
        raise AssertionError("Target-domain exposure occurred")
    if selection_meta != assessment_meta or selection_meta != selection_summary["metadata"]:
        raise AssertionError("Prediction metadata changed between roles")
    if core.sha(RUN / "best.pth") != selection_meta["checkpoint_sha256"]:
        raise AssertionError("Checkpoint hash mismatch")

    rows_by_role = {
        "selection": manifest[(manifest.domain == "BRSET") & (manifest.role == "selection")].reset_index(drop=True),
        "assessment": manifest[(manifest.domain == "mBRSET") & (manifest.role == "assessment")].reset_index(drop=True),
    }
    arrays = {"selection": selection, "assessment": assessment}
    summaries = {"selection": selection_summary, "assessment": assessment_summary}
    recomputed = {}
    for role, rows in rows_by_role.items():
        data = arrays[role]
        if len(data["y_true"]) != len(rows):
            raise AssertionError(f"{role} row count changed")
        if not np.array_equal(data["file_id"].astype(str), rows.file.to_numpy(dtype=str)):
            raise AssertionError(f"{role} order/identity mismatch")
        if not np.array_equal(data["y_true"], rows[core.LABELS].to_numpy(dtype=np.float32)):
            raise AssertionError(f"{role} labels mismatch")
        metrics = core.metrics(data["y_true"], data["probabilities"], selection_meta["thresholds"])
        close(metrics, summaries[role]["metrics"], role)
        recomputed[role] = metrics

    result = {
        "status": "pass",
        "job_id": os.environ["SLURM_JOB_ID"],
        "seed": 0,
        "cohorts": {
            "fit_images": 11372,
            "selection_images": len(selection["y_true"]),
            "assessment_images": len(assessment["y_true"]),
            "assessment_positive_images": {
                "diabetic_retinopathy": int(assessment["y_true"][:, 0].sum()),
                "macular_edema": int(assessment["y_true"][:, 1].sum()),
            },
        },
        "target_training_exposure": complete["exposure"][1],
        "checkpoint_update": selection_meta["checkpoint_update"],
        "thresholds_selected_on_brset": selection_meta["thresholds"],
        "assessment_metrics": recomputed["assessment"],
        "sha256": {
            "checkpoint": core.sha(RUN / "best.pth"),
            "selection_predictions": core.sha(RUN / "selection_predictions.npz"),
            "assessment_predictions": core.sha(RUN / "assessment_predictions.npz"),
        },
        "checks": [
            "training completion and update count",
            "zero target-domain training exposure",
            "checkpoint and metadata chain",
            "ordered cohort identity and labels",
            "independent metric recomputation",
        ],
        "interpretation": "Strict source-only seed-0 evidence on historically reused mBRSET assessment; one seed does not measure training variability.",
    }
    (WORK / "source_only_seed0_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
