"""Decode strict source-only cohort and audit exact cross-role duplicates."""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import socket
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "research/codex/step5"
MANIFEST = WORK / "private/source_only_manifest.csv"
PRIVATE = WORK / "private/source_integrity_rows.csv"
OUTPUT = WORK / "source_integrity_summary.json"


def inspect(row: dict) -> dict:
    try:
        with Image.open(row["image_path"]) as image:
            image.load()
            rgb = image.convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            canonical = ImageOps.exif_transpose(image).convert("RGB")
            canonical_digest = hashlib.sha256(str(canonical.size).encode() + canonical.tobytes()).hexdigest()
        return {**row, "rgb_sha256": digest, "exif_rgb_sha256": canonical_digest,
                "width": rgb.width, "height": rgb.height, "error": ""}
    except Exception as exc:
        return {**row, "error": repr(exc)}


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Image decoding requires an allocated Slurm node")
    data = pd.read_csv(MANIFEST)
    expected = {("BRSET", "fit"): 11372, ("BRSET", "selection"): 2451,
                ("mBRSET", "assessment"): 732}
    counts = data.groupby(["domain", "role"]).size().to_dict()
    if counts != expected or data.duplicated(["domain", "file"]).any():
        raise ValueError(f"Unexpected source-only manifest: {counts}")
    rows = []
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for index, row in enumerate(pool.map(inspect, data.to_dict("records"))):
            rows.append(row)
            if (index + 1) % 1000 == 0:
                print(f"decoded {index + 1}/{len(data)}", flush=True)
    checked = pd.DataFrame(rows)
    PRIVATE.parent.mkdir(parents=True, exist_ok=True)
    checked.to_csv(PRIVATE, index=False)
    duplicates = []
    for kind in ["rgb_sha256", "exif_rgb_sha256"]:
        for digest, group in checked[checked.error == ""].groupby(kind):
            if len(group) > 1:
                duplicates.append({"kind": kind, "hash": digest, "members": int(len(group)),
                                   "cross_role": bool(group.role.nunique() > 1),
                                   "cross_patient": bool(group.patient_key.nunique() > 1),
                                   "cross_domain": bool(group.domain.nunique() > 1)})
    errors = checked[checked.error != ""]
    result = {"status": "pass" if len(errors) == 0 and not any(x["cross_role"] for x in duplicates) else "fail",
              "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              "images": int(len(data)), "decode_errors": int(len(errors)),
              "duplicate_groups": duplicates,
              "cross_role_groups": int(sum(x["cross_role"] for x in duplicates)),
              "cross_domain_groups": int(sum(x["cross_domain"] for x in duplicates)),
              "scope": "Exact native RGB and EXIF-normalized RGB; near duplicates, recompressions and registered-eye identity are not excluded.",
              "privacy": "Aggregate duplicate metadata only; identities retained in ignored private CSV."}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

