"""Allocated CPU provenance/data gate for the matched RETFound feasibility smoke."""
import hashlib
import json
import os
import socket
from pathlib import Path

import pandas as pd


ROOT = Path("/home/users/sthummala2/brset-convnextv2")
HERE = ROOT / "research/codex/step5"
CONFIG = HERE / "retfound_matched_v1.json"
PROTOCOL = HERE / "RETFOUND_MATCHED_PROTOCOL.md"
MODEL_CODE = Path("/home/users/sthummala2/mbrset-retfound/RETFound_MAE/models_vit.py")
POS_CODE = Path("/home/users/sthummala2/mbrset-retfound/RETFound_MAE/util/pos_embed.py")
OUTPUT = HERE / "retfound_preflight.json"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Allocated compute node required")
    config = json.loads(CONFIG.read_text())
    manifest = (HERE / config["manifest"]).resolve()
    if sha(manifest) != config["manifest_sha256"]:
        raise ValueError("Manifest hash mismatch")
    if sha(Path(config["weight_path"])) != config["weight_sha256"]:
        raise ValueError("Weight hash mismatch")
    frame = pd.read_csv(manifest)
    fit = frame[frame.role == "fit"]
    selection = frame[(frame.role == "selection") & (frame.domain == "mBRSET")]
    assessment = frame[frame.role == "assessment"]
    fit_counts = fit.domain.value_counts().to_dict()
    checks = {
        "fit_counts": fit_counts == {"BRSET": 11372, "mBRSET": 3402},
        "selection_images": len(selection) == 725,
        "selection_patients": selection.patient_key.nunique() == 193,
        "two_binary_labels": all(frame[label].isin([0, 1]).all() for label in config["labels"]),
        "both_labels_in_fit": all(fit[label].nunique() == 2 for label in config["labels"]),
        "both_labels_in_selection": all(selection[label].nunique() == 2 for label in config["labels"]),
        "no_assessment_in_fit_or_selection": not set(assessment.file).intersection(set(fit.file) | set(selection.file)),
        "all_paths_exist": frame.image_path.map(Path).map(Path.exists).all(),
        "code_files_exist": MODEL_CODE.exists() and POS_CODE.exists(),
        "assessment_disabled": config["assessment_access"] is False,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise AssertionError(checks)
    result = {
        "status": "pass",
        "job_id": os.environ["SLURM_JOB_ID"],
        "checks": checks,
        "counts": {
            "fit_images": int(len(fit)),
            "fit_patients": int(fit.patient_key.nunique()),
            "fit_by_domain": {key: int(value) for key, value in fit_counts.items()},
            "selection_images": int(len(selection)),
            "selection_patients": int(selection.patient_key.nunique()),
            "assessment_rows_present_but_forbidden": int(len(assessment)),
        },
        "sha256": {
            "config": sha(CONFIG),
            "protocol": sha(PROTOCOL),
            "manifest": sha(manifest),
            "weight": sha(Path(config["weight_path"])),
            "models_vit": sha(MODEL_CODE),
            "pos_embed": sha(POS_CODE),
        },
        "privacy": "Aggregate counts and hashes only.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
