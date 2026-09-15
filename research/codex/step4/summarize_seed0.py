"""Verify and summarize the Step-4 seed-0 validation screen. No test access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
STEP4 = ROOT / "research/codex/step4"
RUNS = STEP4 / "runs"
BASE = ROOT / "research/codex/step2/full_pool/runs/B1_seed0"
OUT = STEP4 / "seed0_validation_summary.json"
LABELS = ["diabetic_retinopathy", "macular_edema"]
METRICS = ["f1_positive", "auroc", "average_precision", "precision", "sensitivity", "specificity"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(path: Path, expected_arm: str, *, forbid_assessment: bool) -> dict:
    complete = json.loads((path / "complete.json").read_text())
    summary = json.loads((path / "selection_summary.json").read_text())
    if not complete["training_complete"] or complete["smoke"]:
        raise ValueError(f"Incomplete research run: {expected_arm}")
    if complete["updates"] != 5775 or complete["contract"]["arm"] != expected_arm:
        raise ValueError(f"Run contract mismatch: {expected_arm}")
    if summary["metadata"]["arm"] != expected_arm or summary["metadata"]["checkpoint_sha256"] != sha(path / "best.pth"):
        raise ValueError(f"Selection provenance mismatch: {expected_arm}")
    if forbid_assessment and any((path / name).exists() for name in ("assessment_summary.json", "assessment_predictions.npz")):
        raise ValueError(f"Step-4 test assessment exists unexpectedly: {expected_arm}")
    return summary


def main() -> None:
    # B1 is the completed Step-2 reference and legitimately has its historical
    # Step-2 assessment artifacts. This comparison reads only its validation
    # selection summary. The Step-4 arms themselves must remain test-free.
    summaries = {"B1": read_run(BASE, "B1", forbid_assessment=False)}
    for arm in ("A1", "A2", "A3"):
        summaries[arm] = read_run(RUNS / f"{arm}_seed0", arm, forbid_assessment=True)
    reference = summaries["B1"]["metrics"]
    arms = {}
    for arm in ("A1", "A2", "A3"):
        current = summaries[arm]["metrics"]
        delta = {
            label: {metric: float(current[label][metric] - reference[label][metric]) for metric in METRICS}
            for label in LABELS
        }
        f1 = [delta[label]["f1_positive"] for label in LABELS]
        auroc = [delta[label]["auroc"] for label in LABELS]
        screen = max(f1) >= 0.01 and min(f1) >= -0.01 and min(auroc) >= -0.01
        arms[arm] = {"metrics": current, "delta_from_B1": delta, "replication_screen_pass": bool(screen)}
    output = {
        "status": "complete",
        "scope": "mBRSET validation only; no assessment/test access",
        "seed": 0,
        "reference": {"arm": "B1", "metrics": reference},
        "arms": arms,
        "screen_rule": "at least one label F1 delta >= +0.01; other label F1 delta >= -0.01; both AUROC deltas >= -0.01",
        "screen_interpretation": "engineering replication gate, not significance or clinical margin",
        "input_sha256": {
            "B1_selection": sha(BASE / "selection_summary.json"),
            **{f"{arm}_selection": sha(RUNS / f"{arm}_seed0/selection_summary.json") for arm in ("A1", "A2", "A3")},
            "analysis": sha(Path(__file__)),
        },
    }
    OUT.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
