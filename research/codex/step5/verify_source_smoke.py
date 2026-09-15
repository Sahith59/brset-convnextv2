"""Verify source-only stop/resume equivalence and no-target smoke behavior."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "research/codex/step5"
FULL = Path("/data/users3/sthummala2/brset-codex/step5/source_smoke_full")
RESUME = Path("/data/users3/sthummala2/brset-codex/step5/source_smoke_resume")
OUTPUT = WORK / "source_smoke_checks.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Allocated node required")
    states = [torch.load(folder / "last.pth", map_location="cpu", weights_only=False)
              for folder in [FULL, RESUME]]
    mismatches = []
    for field in ["model", "ema"]:
        for key in states[0][field]:
            if not torch.equal(states[0][field][key], states[1][field][key]):
                mismatches.append(f"{field}:{key}")
    for key in states[0]["optimizer"]["state"]:
        for field, first in states[0]["optimizer"]["state"][key].items():
            second = states[1]["optimizer"]["state"][key][field]
            if torch.is_tensor(first) and not torch.equal(first, second):
                mismatches.append(f"optimizer:{key}:{field}")
    for field in ["update", "exposure", "best"]:
        if states[0][field] != states[1][field]:
            mismatches.append(field)
    if states[0]["update"] != 6:
        raise AssertionError("Smoke did not complete six updates")
    for folder in [FULL, RESUME]:
        contract = json.loads((folder / "contract.json").read_text())
        if contract["fit_domain"] != "BRSET" or contract["selection_domain"] != "BRSET":
            raise ValueError("Smoke domain boundary changed")
        if (folder / "assessment_predictions.npz").exists():
            raise ValueError("Smoke accessed target assessment")
    events = [json.loads(line) for line in (FULL / "events.jsonl").read_text().splitlines()]
    update_seconds = [item["seconds"] for item in events if item["event"] == "update" and item["global_update"] > 1]
    selection_seconds = [item["inference_seconds"] for item in events if item["event"] == "selection"]
    complete = json.loads((FULL / "complete.json").read_text())
    estimated_hours = (np.median(update_seconds) * 5775 + np.median(selection_seconds) * 26
                       + np.median(complete["checkpoint_write_seconds"]) * 25) / 3600
    result = {"status": "pass" if not mismatches else "fail", "state_mismatches": mismatches,
              "updates": 6, "interruption_after_update": 3, "target_assessment_accessed": False,
              "median_update_seconds": float(np.median(update_seconds)),
              "median_selection_seconds": float(np.median(selection_seconds)),
              "estimated_full_hours": float(estimated_hours),
              "gpu_peak_allocated_gib": complete["max_gpu_bytes"] / 1024**3,
              "checkpoint_sha256": {folder.name: sha(folder / "last.pth") for folder in [FULL, RESUME]},
              "scope": "Six-update deterministic smoke with full BRSET validation inference; not performance or convergence evidence."}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

