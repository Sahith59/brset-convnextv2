"""Verify the corrected Step-4 smoke artifacts without decoding images."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
SMOKE = ROOT / "research/codex/step4/smoke"
OUT = ROOT / "research/codex/step4/smoke_verification.json"


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main() -> None:
    records = {}
    shared = None
    exposures = []
    for arm in ("A1", "A2", "A3"):
        folder = SMOKE / arm
        complete = json.loads((folder / "complete.json").read_text())
        selection = json.loads((folder / "selection_summary.json").read_text())
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
        contract = complete["contract"]
        if not complete["training_complete"] or not complete["smoke"] or complete["updates"] != 6:
            raise ValueError(f"Invalid smoke completion: {arm}")
        if contract["arm"] != arm or contract["phases"][0]["domain"] != "both":
            raise ValueError(f"Smoke did not inherit joint B1 phase: {arm}")
        if selection["metrics"]["diabetic_retinopathy"]["n_images"] != 725:
            raise ValueError(f"Wrong validation cohort: {arm}")
        if selection["metrics"]["macular_edema"]["n_images"] != 725:
            raise ValueError(f"Wrong validation cohort: {arm}")
        if any((folder / name).exists() for name in ("assessment_summary.json", "assessment_predictions.npz")):
            raise ValueError(f"Unexpected assessment artifact: {arm}")
        phase_events = [event for event in events if event["event"] == "phase_complete"]
        if len(phase_events) != 1:
            raise ValueError(f"Wrong phase event count: {arm}")
        exposure = phase_events[0]["exposure"]
        if exposure[0][0] <= 0 or exposure[1][0] <= 0 or sum(row[0] for row in exposure) != 384:
            raise ValueError(f"Invalid source/target exposure: {arm}")
        if not finite_tree(complete) or not finite_tree(selection) or not finite_tree(events):
            raise ValueError(f"Nonfinite smoke artifact: {arm}")
        key = {name: contract[name] for name in (
            "protocol_sha256", "manifest_sha256", "trainer_sha256", "core_sha256",
            "initialization_sha256", "seed", "step4_wrapper_sha256", "step4_transform_sha256",
            "source_transform_order",
        )}
        if shared is None:
            shared = key
        elif key != shared:
            raise ValueError(f"Cross-arm contract mismatch: {arm}")
        exposures.append(exposure)
        records[arm] = {
            "updates": complete["updates"],
            "fit_images": next(event["fit_images"] for event in events if event["event"] == "phase"),
            "exposure": exposure,
            "median_update_seconds": phase_events[0]["median_update_seconds"],
            "validation_images_per_label": 725,
            "assessment_performed": False,
        }
    if not all(exposure == exposures[0] for exposure in exposures[1:]):
        raise ValueError("Smoke arms did not receive identical domain/label exposure")
    output = {
        "pass": True,
        "job": "4397264",
        "scope": "six-update code/numerical gate; metrics are not research results",
        "joint_pool_expected_images": 14774,
        "records": records,
        "shared_contract": shared,
        "invalid_predecessor": {
            "job": "4395006",
            "state": "cancelled and excluded",
            "reason": "target-only inherited smoke phase did not exercise source transforms",
        },
    }
    OUT.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
