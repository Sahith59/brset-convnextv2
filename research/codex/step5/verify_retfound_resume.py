"""Verify exact RETFound smoke interruption/resume equivalence and result boundaries."""
import json
import os
import socket
from pathlib import Path

import torch


ROOT = Path("/home/users/sthummala2/brset-convnextv2")
HERE = ROOT / "research/codex/step5"
BASE = Path("/data/users3/sthummala2/brset-codex/step5")
OUTPUT = HERE / "retfound_resume_checks.json"


def compare(left, right, path="root"):
    mismatches = []
    if torch.is_tensor(left):
        if not torch.equal(left, right):
            mismatches.append(path)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            mismatches.append(path + ".keys")
        else:
            for key in left:
                mismatches.extend(compare(left[key], right[key], path + "." + str(key)))
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            mismatches.append(path + ".length")
        else:
            for index, (a, b) in enumerate(zip(left, right)):
                mismatches.extend(compare(a, b, path + f"[{index}]"))
    elif left != right:
        mismatches.append(path)
    return mismatches


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Allocated compute node required")
    full_dir = BASE / "retfound_smoke_full"; resume_dir = BASE / "retfound_smoke_resume"
    full = torch.load(full_dir / "last.pth", map_location="cpu", weights_only=False)
    resumed = torch.load(resume_dir / "last.pth", map_location="cpu", weights_only=False)
    mismatches = []
    for key in ("contract", "model", "ema", "optimizer", "update", "best", "exposure"):
        mismatches.extend(compare(full[key], resumed[key], key))
    full_complete = json.loads((full_dir / "complete.json").read_text())
    resumed_complete = json.loads((resume_dir / "complete.json").read_text())
    for record in (full_complete, resumed_complete):
        if not record["training_complete"] or record["updates"] != 6 or not record["smoke"] or record["assessment_performed"]:
            mismatches.append("complete_boundary")
    result = {"pass": not mismatches, "job_id": os.environ["SLURM_JOB_ID"],
              "state_mismatches": mismatches, "updates": 6, "assessment_accessed": False,
              "full_max_gpu_gib": full_complete["max_gpu_bytes"] / 1024 ** 3,
              "resume_max_gpu_gib": resumed_complete["max_gpu_bytes"] / 1024 ** 3}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if mismatches:
        raise RuntimeError(mismatches[:20])


if __name__ == "__main__":
    main()
