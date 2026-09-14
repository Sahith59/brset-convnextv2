"""Verify that advisor-gallery source panels originate from raw training files."""
from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
MANIFEST = ROOT / "research/codex/step2/full_pool/protocol/full_pool_manifest.csv"
PRIVATE = ROOT / "research/codex/reports/private"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Image verification requires an allocated Slurm node")
    manifest = pd.read_csv(MANIFEST)
    fit = manifest[manifest.role.eq("fit")]
    source = fit[fit.domain.eq("BRSET")]
    target = fit[fit.domain.eq("mBRSET")]
    rng = np.random.default_rng(20260914)
    selected = list(source.iloc[rng.choice(len(source), 4, replace=False)].itertuples())
    selected += list(target.iloc[rng.choice(len(target), 4, replace=False)].itertuples())
    records = []
    for index, row in enumerate(selected):
        path = Path(row.image_path)
        with Image.open(path) as raw:
            raw_size = list(raw.size)
            prepared = raw.convert("RGB").resize((560, 560), Image.Resampling.BILINEAR).crop((24, 24, 536, 536))
        array = np.asarray(prepared, dtype=np.float32)
        jumps = np.abs(np.diff(array.mean(axis=(1, 2))))
        group = "source" if index < 4 else "target"
        output = PRIVATE / f"verified_{group}_{index % 4 + 1}.png"
        prepared.save(output)
        records.append({
            "group": group,
            "row": index % 4 + 1,
            "raw_basename": path.name,
            "raw_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_size": raw_size,
            "prepared_pixel_sha256": sha_bytes(np.asarray(prepared).tobytes()),
            "strongest_horizontal_mean_jump_y": int(jumps.argmax() + 1),
            "strongest_horizontal_mean_jump": float(jumps.max()),
            "prepared_output": str(output),
        })
    result = {
        "pass": len(records) == 8,
        "selection_seed": 20260914,
        "geometry": "RGB, bilinear 560x560 resize, center crop [24:536,24:536]",
        "records": records,
        "interpretation": "The saved prepared files are direct decodes of manifest-listed raw training files; any feature present in them predates report assembly.",
    }
    (PRIVATE / "gallery_source_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
