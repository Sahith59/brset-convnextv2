"""Training-only label-composition sensitivity of BRSET/mBRSET appearance gaps."""
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "research/codex/step4"
MANIFEST = ROOT / "research/codex/step2/full_pool/protocol/full_pool_manifest.csv"
OPS_PATH = ROOT / "scripts/55_degradation_ops.py"
OUTPUT = WORK / "appearance_label_audit.json"
EXPECTED_MANIFEST_SHA256 = "04181fe8ba018839181ecade097228856622cab8d428b77a22635b881c5bc3fc"
LABELS = ["diabetic_retinopathy", "macular_edema"]
STATS = ["sharpness", "brightness", "contrast", "falloff", "saturation"]

spec = importlib.util.spec_from_file_location("step4_appearance_ops", OPS_PATH)
OPS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OPS)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(record):
    path, domain, dr, me = record
    with Image.open(path) as raw:
        image = raw.convert("RGB").resize((560, 560), Image.Resampling.BILINEAR)
    image = image.crop((24, 24, 536, 536))
    values = OPS.image_stats(image)
    if values is None or not all(np.isfinite(values[name]) for name in STATS):
        raise ValueError("Invalid appearance statistics")
    return {"domain": domain, "dr": int(dr), "me": int(me), **values}


def describe(frame):
    result = {"n": int(len(frame))}
    for name in STATS:
        values = frame[name].to_numpy(dtype=float)
        result[name] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "median": float(np.median(values)),
        }
    return result


def main():
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Manifest hash mismatch")
    manifest = pd.read_csv(MANIFEST)
    fit = manifest[manifest.role.eq("fit")].copy()
    if fit.duplicated(["domain", "file"]).any():
        raise ValueError("Duplicate image identity")
    expected_counts = {"BRSET": 11372, "mBRSET": 3402}
    counts = fit.domain.value_counts().to_dict()
    if counts != expected_counts or not fit[LABELS].isin([0, 1]).all().all():
        raise ValueError(f"Unexpected fit cohort: {counts}")

    records = list(fit[["image_path", "domain", *LABELS]].itertuples(index=False, name=None))
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        for row in pool.map(measure, records, chunksize=16):
            rows.append(row)
    measured = pd.DataFrame(rows)
    if len(measured) != len(fit):
        raise AssertionError("Not every fit image was measured")

    measured["stratum"] = measured.dr.astype(str) + measured.me.astype(str)
    strata = ["00", "01", "10", "11"]
    by_domain = {domain: describe(measured[measured.domain.eq(domain)]) for domain in expected_counts}
    by_domain_stratum = {
        domain: {cell: describe(measured[measured.domain.eq(domain) & measured.stratum.eq(cell)]) for cell in strata}
        for domain in expected_counts
    }

    pooled_counts = measured.stratum.value_counts().reindex(strata).astype(int)
    weights = (pooled_counts / pooled_counts.sum()).to_dict()
    standardized = {}
    for domain in expected_counts:
        standardized[domain] = {}
        for name in STATS:
            standardized[domain][name] = float(sum(
                weights[cell] * by_domain_stratum[domain][cell][name]["mean"] for cell in strata
            ))

    sensitivity = {}
    for name in STATS:
        raw_gap = by_domain["mBRSET"][name]["mean"] - by_domain["BRSET"][name]["mean"]
        standardized_gap = standardized["mBRSET"][name] - standardized["BRSET"][name]
        scale = max(by_domain["mBRSET"][name]["sample_sd"], 1e-12)
        sensitivity[name] = {
            "raw_target_minus_source": float(raw_gap),
            "label_standardized_target_minus_source": float(standardized_gap),
            "change_after_standardization": float(standardized_gap - raw_gap),
            "raw_gap_in_target_sd": float(raw_gap / scale),
            "standardized_gap_in_target_sd": float(standardized_gap / scale),
            "direction_changed": bool(np.sign(raw_gap) != np.sign(standardized_gap)),
        }

    result = {
        "status": "complete",
        "date_utc": "2026-09-14",
        "scope": "original training split only; no validation or test images",
        "preprocessing": "bilinear 560x560 resize then center 512x512 crop",
        "manifest_sha256": sha256(MANIFEST),
        "analysis_sha256": sha256(Path(__file__)),
        "appearance_ops_sha256": sha256(OPS_PATH),
        "image_counts": expected_counts,
        "joint_label_codes": {"00": "DR0/ME0", "01": "DR0/ME1", "10": "DR1/ME0", "11": "DR1/ME1"},
        "common_composition_weights": {key: float(value) for key, value in weights.items()},
        "by_domain": by_domain,
        "by_domain_and_joint_label": by_domain_stratum,
        "label_standardized_means": standardized,
        "gap_sensitivity": sensitivity,
        "interpretation_limit": "Descriptive label-composition sensitivity. Standardization does not isolate camera effects from population, site, quality, other disease, or unmeasured differences.",
        "privacy": "Aggregate statistics only; no file paths or patient identifiers in output."
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": result["status"],
        "image_counts": result["image_counts"],
        "common_composition_weights": result["common_composition_weights"],
        "gap_sensitivity": result["gap_sensitivity"],
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()

