"""Compute Step-3 label-stratum and exposure facts on an allocated node."""
import json
import os
import socket
from pathlib import Path

import pandas as pd

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
MANIFEST = ROOT / "research/codex/step2/full_pool/protocol/full_pool_manifest.csv"
OUTPUT = ROOT / "research/codex/step3/design_counts.json"
LABELS = ["diabetic_retinopathy", "macular_edema"]
UPDATES = 5775
BATCH = 64


def group_record(domain, group):
    patients = group.groupby("patient_key")[LABELS].max()
    return {
        "domain": domain,
        "images": len(group),
        "patients": group.patient_key.nunique(),
        "image_label_strata": {
            f"DR{dr}_ME{me}": int(((group[LABELS[0]] == dr) & (group[LABELS[1]] == me)).sum())
            for dr in [0, 1] for me in [0, 1]
        },
        "patient_any_image_strata": {
            f"DR{dr}_ME{me}": int(((patients[LABELS[0]] == dr) & (patients[LABELS[1]] == me)).sum())
            for dr in [0, 1] for me in [0, 1]
        },
        "positive_images": {label: int(group[label].sum()) for label in LABELS},
        "positive_image_fraction": {label: float(group[label].mean()) for label in LABELS},
    }


def exposure(name, source_draws, target_draws, source_count, target_count):
    return {
        "name": name,
        "total_draws": source_draws + target_draws,
        "BRSET_draws": source_draws,
        "mBRSET_draws": target_draws,
        "BRSET_nominal_passes": source_draws / source_count,
        "mBRSET_nominal_passes": target_draws / target_count,
    }


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Use an allocated Slurm compute node")
    data = pd.read_csv(MANIFEST)
    fit = data[data.role == "fit"].copy()
    source = fit[fit.domain == "BRSET"]
    target = fit[fit.domain == "mBRSET"]
    total_draws = UPDATES * BATCH
    natural_source = total_draws * len(source) / len(fit)
    natural_target = total_draws * len(target) / len(fit)
    half = total_draws / 2
    target_matched_updates = 2 * UPDATES
    result = {
        "manifest": str(MANIFEST),
        "common_updates": UPDATES,
        "effective_batch": BATCH,
        "groups": [group_record("BRSET", source), group_record("mBRSET", target),
                   group_record("joint", fit)],
        "expected_exposures": [
            exposure("B0 target-only", 0, total_draws, len(source), len(target)),
            exposure("B1 natural joint", natural_source, natural_target, len(source), len(target)),
            exposure("equal-domain joint", half, half, len(source), len(target)),
            exposure("B2 sequential", 2887 * BATCH, 2888 * BATCH, len(source), len(target)),
            exposure("equal-domain joint with B0-matched target draws", total_draws,
                     total_draws, len(source), len(target)),
        ],
        "target_exposure_matched_equal_domain_updates": target_matched_updates,
        "target_exposure_matched_runtime_warning": "11,550 updates are twice the common optimizer budget and require a separate compute control and likely checkpoint resume under the 12-hour limit.",
        "interpretation": "Expected natural-joint draws are fractional expectations. Actual exposure is recorded during training. Label balancing must use all four joint label strata to preserve DR/ME co-occurrence rather than balancing each label independently.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
