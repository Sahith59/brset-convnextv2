"""Allocated-CPU source-only code, cohort and provenance gate."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import socket
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "research/codex/step5"
FILES = [WORK / "train_source_only.py", WORK / "source_only_v1.json",
         WORK / "SOURCE_ONLY_PROTOCOL.md", WORK / "private/source_only_manifest.csv"]
OUTPUT = WORK / "source_preflight.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Allocated Slurm node required")
    for path in FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
    ast.parse((WORK / "train_source_only.py").read_text())
    protocol = json.loads((WORK / "source_only_v1.json").read_text())
    data = pd.read_csv(WORK / "private/source_only_manifest.csv")
    if sha(WORK / "private/source_only_manifest.csv") != protocol["private_manifest_sha256"]:
        raise ValueError("Manifest hash mismatch")
    expected = {("BRSET", "fit"): 11372, ("BRSET", "selection"): 2451,
                ("mBRSET", "assessment"): 732}
    if data.groupby(["domain", "role"]).size().to_dict() != expected:
        raise ValueError("Cohort counts changed")
    if len(data[(data.domain == "mBRSET") & (data.role != "assessment")]) != 0:
        raise ValueError("Target image available before assessment")
    if data.groupby("patient_key").role.nunique().max() != 1:
        raise ValueError("Patient crosses roles")
    integrity = json.loads((WORK / "source_integrity_summary.json").read_text())
    if integrity["status"] != "pass" or integrity["manifest_sha256"] != protocol["private_manifest_sha256"]:
        raise ValueError("Decoded-image integrity gate missing or stale")
    result = {"status": "pass", "counts": {f"{domain}_{role}": int(count)
              for (domain, role), count in expected.items()}, "target_preassessment_images": 0,
              "integrity_sha256": sha(WORK / "source_integrity_summary.json"),
              "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in FILES},
              "checks": ["Python syntax", "manifest hash", "role counts", "patient separation",
                         "zero target pre-assessment access", "decoded-image integrity"]}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

