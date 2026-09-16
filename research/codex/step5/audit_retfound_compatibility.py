"""Audit whether the existing RETFound companion results match the frozen B1 protocol."""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path("/home/users/sthummala2/brset-convnextv2")
COMPANION = Path("/home/users/sthummala2/mbrset-retfound")
CURRENT_MANIFEST = ROOT / "research/codex/step2/full_pool/protocol/full_pool_manifest.csv"
CURRENT_PROTOCOL = ROOT / "research/codex/step2/full_pool/protocol/v1.json"
OLD_SPLIT = COMPANION / "results/splits.csv"
OLD_TRAIN = COMPANION / "scripts/05_focal_finetune.slurm"
OLD_SPLITTER = COMPANION / "scripts/01_prepare_splits.py"
WEIGHT_CONFIG = COMPANION / "checkpoints/RETFound_dinov2_shanghai/config.json"
WEIGHT = COMPANION / "checkpoints/RETFound_dinov2_shanghai/RETFound_dinov2_shanghai.pth"
OUTPUT = ROOT / "research/codex/step5/retfound_compatibility_audit.json"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def counts(frame, split_col, patient_col):
    result = {}
    for split, group in frame.groupby(split_col):
        result[str(split)] = {
            "images": int(len(group)),
            "patients": int(group[patient_col].nunique()),
        }
    return result


def main():
    required = (CURRENT_MANIFEST, CURRENT_PROTOCOL, OLD_SPLIT, OLD_TRAIN, OLD_SPLITTER, WEIGHT_CONFIG, WEIGHT)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    current = pd.read_csv(CURRENT_MANIFEST)
    current = current[current["domain"] == "mBRSET"].copy()
    current["patient"] = current["patient_key"].str.replace("mBRSET:", "", regex=False)
    old = pd.read_csv(OLD_SPLIT)
    old["patient"] = old["patient"].astype(str)
    current["patient"] = current["patient"].astype(str)

    merged = current[["file", "original_split", "diabetic_retinopathy", "macular_edema"]].merge(
        old[["file", "split", "referable", "final_icdr"]], on="file", how="inner", validate="one_to_one"
    )
    same_split = merged["original_split"].eq(merged["split"])
    same_dr_label = merged["diabetic_retinopathy"].eq(merged["referable"])

    train_text = OLD_TRAIN.read_text()
    splitter_text = OLD_SPLITTER.read_text()
    config = json.loads(WEIGHT_CONFIG.read_text())
    protocol = json.loads(CURRENT_PROTOCOL.read_text())
    old_epochs = int(re.search(r"--epochs\s+(\d+)", train_text).group(1))
    old_batch = int(re.search(r"--batch_size\s+(\d+)", train_text).group(1))
    old_size = int(re.search(r"--input_size\s+(\d+)", train_text).group(1))

    result = {
        "status": "pass",
        "comparison_ready": False,
        "companion_repository": str(COMPANION),
        "existing_result_scope": "Separate mBRSET-only or BRSET-only RETFound runs for referable DR; no protocol-matched joint any-DR plus ME result.",
        "cohorts": {
            "current_mbrset": counts(current, "original_split", "patient"),
            "companion_mbrset": counts(old, "split", "patient"),
            "common_files": int(len(merged)),
            "current_only_files": int(len(set(current["file"]) - set(old["file"]))),
            "companion_only_files": int(len(set(old["file"]) - set(current["file"]))),
            "same_split_common_files": int(same_split.sum()),
            "different_split_common_files": int((~same_split).sum()),
        },
        "label_alignment": {
            "current_dr": "any DR binary label",
            "companion_dr": "referable DR, final ICDR grade >= 2",
            "same_dr_label_common_files": int(same_dr_label.sum()),
            "different_dr_label_common_files": int((~same_dr_label).sum()),
            "companion_me": "absent",
            "splitter_definition_verified": bool(re.search(r'final_icdr"\]\s*>=\s*2', splitter_text)),
        },
        "protocol_matrix": [
            {"item": "backbone", "B1": "ConvNeXt V2 Large", "existing_companion": config["architectures"][0], "aligned": False},
            {"item": "pretraining", "B1": "external ConvNeXt initialization", "existing_companion": "RETFound DINOv2 Shanghai CFP", "aligned": False},
            {"item": "outputs", "B1": "two sigmoid labels: any DR and ME", "existing_companion": "two-class softmax: referable DR only", "aligned": False},
            {"item": "input", "B1": "bilinear 560 resize, 512 crop", "existing_companion": f"{old_size} px input", "aligned": False},
            {"item": "optimization budget", "B1": f"{protocol['recipe']['updates']} optimizer updates, effective batch {protocol['recipe']['effective_batch']}", "existing_companion": f"{old_epochs} epochs, batch {old_batch}", "aligned": False},
            {"item": "split", "B1": "frozen original patient split", "existing_companion": "new seed-42 70/15/15 patient split after quality/label filtering", "aligned": False},
            {"item": "selection", "B1": "mean DR/ME validation AUROC then label-wise F1 thresholds", "existing_companion": "RETFound trainer validation criterion and argmax", "aligned": False},
            {"item": "training domains", "B1": "joint BRSET + mBRSET", "existing_companion": "separate dataset runs", "aligned": False},
            {"item": "assessment", "B1": "732-image reused historical split with fixed thresholds", "existing_companion": "different held-out split; previously assessed", "aligned": False},
        ],
        "blocking_mismatches": [
            "Different clinical task: referable DR instead of any DR, and no ME output.",
            "Different patient split and inclusion cohort.",
            "Different image resolution, training budget, selection rule and prediction threshold.",
            "Existing runs are source-only or target-only, not a joint B1-style comparator.",
            "Existing test metrics were already used and cannot be relabeled as a fresh matched comparison.",
        ],
        "usable_assets": {
            "official_code_checkout_present": True,
            "official_code_commit": "ae9a9ecf37857cf47b8aa9f87cd6f710d75db287",
            "pretrained_weight_present": True,
            "pretrained_weight_bytes": int(WEIGHT.stat().st_size),
            "pretrained_weight_sha256": sha(WEIGHT),
            "weight_config": config,
        },
        "decision": "Do not cite the old numbers as a B1 comparator and do not launch the old scripts. Build a new protocol-matched RETFound adapter only after a frozen implementation contract and allocated preflight/smoke pass.",
        "input_sha256": {str(path): sha(path) for path in required if path != WEIGHT},
        "privacy": "Aggregate counts only; no patient identifier or row-level label exported.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
