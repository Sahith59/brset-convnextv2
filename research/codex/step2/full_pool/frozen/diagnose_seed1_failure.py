"""Diagnose the seed-1 data-read failure and inspect resumable checkpoints."""
import hashlib
import json
import os
import socket
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
WORK = ROOT / "research/codex/step2/full_pool"
MISSING_AT_FAILURE = Path("/data/users4/nshaik3/Datasets/mBRSET/physionet.org/files/mbrset/1.0/images/507.1.jpg")
EXPECTED_SHA = "5464e6f6baae7a6d722801ceb4724de9374c9054a9723028fb7bd935f5c29159"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Use an allocated Slurm node")
    repeated = []
    for _ in range(20):
        with Image.open(MISSING_AT_FAILURE) as image:
            image.load()
            repeated.append([image.width, image.height, image.mode])
    actual_sha = sha(MISSING_AT_FAILURE)
    if actual_sha != EXPECTED_SHA:
        raise RuntimeError("The failed file now has unexpected content")

    manifest = pd.read_csv(WORK / "protocol/full_pool_manifest.csv")
    target = manifest[(manifest.domain == "mBRSET") & (manifest.role == "fit")]
    errors = []
    for row in target.itertuples(index=False):
        try:
            with Image.open(row.image_path) as image:
                image.load()
        except Exception as exc:
            errors.append({"file": row.file, "error": repr(exc)})

    checkpoints = []
    for arm in ["B0", "B1", "B2"]:
        path = WORK / "runs" / f"{arm}_seed1" / "last.pth"
        state = torch.load(path, map_location="cpu", weights_only=False)
        checkpoints.append({
            "arm": arm,
            "sha256": sha(path),
            "phase_index": state["phase"],
            "phase_update": state["phase_update"],
            "global_update": state["global_update"],
            "consumed_draws": state["consumed_draws"],
            "contract_seed": state["contract"]["seed"],
            "contract_arm": state["contract"]["arm"],
        })
    result = {
        "pass": not errors,
        "job": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "failure_classification": "transient_or_node_specific_visibility if current full check passes; not evidence of image deletion or model failure",
        "failed_path_now_exists": MISSING_AT_FAILURE.is_file(),
        "failed_path_sha256": actual_sha,
        "repeated_successful_decodes": len(repeated),
        "mBRSET_fit_images_decoded": len(target),
        "decode_errors": errors,
        "checkpoints": checkpoints,
    }
    (WORK / "seed1_failure_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
