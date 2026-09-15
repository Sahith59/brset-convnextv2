"""Allocated-CPU cohort and artifact preflight for revised Step 5."""
from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "research/codex/step5"
MANIFEST = ROOT / "research/codex/step2/full_pool/protocol/full_pool_manifest.csv"
SOURCE_SPLITS = ROOT / "results/splits_multilabel.csv"
SOURCE_ONLY_MANIFEST = WORK / "private/source_only_manifest.csv"
PROTOCOL = ROOT / "research/codex/step2/full_pool/protocol/v1.json"
INTEGRITY = ROOT / "research/codex/step2/full_pool/image_integrity_summary.json"
OUTPUT = WORK / "design_preflight.json"
LABELS = ["diabetic_retinopathy", "macular_edema"]
EXPECTED_MANIFEST_SHA = "04181fe8ba018839181ecade097228856622cab8d428b77a22635b881c5bc3fc"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def node_check() -> None:
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Numerical preflight requires an allocated Slurm node")


def describe(frame: pd.DataFrame) -> dict:
    if len(frame) == 0:
        return {"images": 0, "patients": 0,
                **{label: {"positive_images": 0, "positive_image_fraction": None,
                           "positive_patients": 0, "positive_patient_fraction": None}
                   for label in LABELS}}
    result = {"images": int(len(frame)), "patients": int(frame.patient_key.nunique())}
    for label in LABELS:
        positive_images = int(frame[label].sum())
        positive_patients = int(frame.groupby("patient_key")[label].max().sum())
        result[label] = {"positive_images": positive_images,
                         "positive_image_fraction": float(positive_images / len(frame)),
                         "positive_patients": positive_patients,
                         "positive_patient_fraction": float(positive_patients / frame.patient_key.nunique())}
    return result


def main() -> None:
    node_check()
    if sha(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise ValueError("Manifest hash mismatch")
    data = pd.read_csv(MANIFEST)
    if data.duplicated(["domain", "file"]).any():
        raise ValueError("Duplicate image identity")
    if not data[LABELS].isin([0, 1]).all().all():
        raise ValueError("Invalid labels")
    if data.groupby("patient_key").role.nunique().max() != 1:
        raise ValueError("Patient crosses roles")
    expected_role = data.original_split.map({"train": "fit", "val": "selection", "test": "assessment"})
    if expected_role.isna().any() or not expected_role.eq(data.role).all():
        raise ValueError("Original split/role mismatch")

    source_root = Path(data[(data.domain == "BRSET") & (data.role == "fit")].iloc[0].image_path).parent
    raw_source = pd.read_csv(SOURCE_SPLITS)
    if raw_source.duplicated("file").any() or not raw_source[LABELS].isin([0, 1]).all().all():
        raise ValueError("Invalid original BRSET split table")
    raw_source["domain"] = "BRSET"
    raw_source["patient_key"] = "BRSET:" + raw_source.patient_id.astype(str)
    raw_source["original_split"] = raw_source.split
    raw_source["role"] = raw_source.split.map({"train": "fit", "val": "selection", "test": "assessment"})
    raw_source["image_path"] = raw_source.file.map(lambda value: str(source_root / value))
    raw_source["fold"] = -1
    if raw_source.role.isna().any() or raw_source.groupby("patient_key").role.nunique().max() != 1:
        raise ValueError("Original BRSET split roles or patient separation invalid")
    if not raw_source.image_path.map(lambda value: Path(value).is_file()).all():
        raise ValueError("Missing BRSET image in original split table")
    source = raw_source[["domain", "file", "patient_key", *LABELS,
                         "original_split", "fold", "role", "image_path"]].copy()
    target = data[data.domain == "mBRSET"].copy()
    cohorts = {"BRSET": {}, "mBRSET": {}}
    for role in ["fit", "selection", "assessment"]:
        cohorts["BRSET"][role] = describe(source[source.role == role])
        cohorts["mBRSET"][role] = describe(target[target.role == role])

    prediction_hashes = {}
    checkpoint_hashes = {}
    for seed in range(3):
        folder = ROOT / f"research/codex/step2/full_pool/runs/B1_seed{seed}"
        pred = folder / "selection_predictions.npz"
        checkpoint = folder / "best.pth"
        summary = json.loads((folder / "selection_summary.json").read_text())
        archive = np.load(pred, allow_pickle=False)
        if archive["probabilities"].shape != (725, 2) or len(np.unique(archive["file_id"])) != 725:
            raise ValueError(f"B1 seed{seed} validation predictions invalid")
        if summary["metadata"]["checkpoint_sha256"] != sha(checkpoint):
            raise ValueError(f"B1 seed{seed} checkpoint provenance mismatch")
        prediction_hashes[f"seed{seed}"] = sha(pred)
        checkpoint_hashes[f"seed{seed}"] = sha(checkpoint)

    integrity = json.loads(INTEGRITY.read_text())
    if not integrity["pass"] or integrity["manifest_sha256"] != EXPECTED_MANIFEST_SHA:
        raise ValueError("Existing decoded-image integrity gate invalid")

    source_train = source[source.role == "fit"]
    source_selection = source[source.role == "selection"]
    target_assessment = target[target.role == "assessment"]
    source_only = pd.concat([source[source.role.isin(["fit", "selection"])], target_assessment], ignore_index=True)
    if source_only.groupby("patient_key").role.nunique().max() != 1:
        raise ValueError("Source-only manifest patient boundary failure")
    SOURCE_ONLY_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    source_only.to_csv(SOURCE_ONLY_MANIFEST, index=False)
    strict_source_contract = {
        "fit": "BRSET original train only",
        "selection": "BRSET original validation only",
        "assessment": "mBRSET original test only after model and thresholds freeze",
        "fit_images": int(len(source_train)),
        "fit_patients": int(source_train.patient_key.nunique()),
        "selection_images": int(len(source_selection)),
        "selection_patients": int(source_selection.patient_key.nunique()),
        "assessment_images": int(len(target_assessment)),
        "assessment_patients": int(target_assessment.patient_key.nunique()),
        "target_images_used_before_assessment": 0,
        "manifest_sha256": sha(SOURCE_ONLY_MANIFEST),
        "new_decoded_duplicate_gate_required": True,
    }
    result = {"status": "pass", "date_utc": "2026-09-15",
              "cohorts": cohorts, "strict_source_contract": strict_source_contract,
              "b1_frozen_artifacts": {"prediction_sha256": prediction_hashes,
                                      "checkpoint_sha256": checkpoint_hashes},
              "inputs": {"manifest_sha256": sha(MANIFEST), "source_split_sha256": sha(SOURCE_SPLITS),
                         "source_only_manifest_sha256": sha(SOURCE_ONLY_MANIFEST), "protocol_sha256": sha(PROTOCOL),
                         "integrity_sha256": sha(INTEGRITY), "script_sha256": sha(Path(__file__))},
              "checks": ["original split roles", "within-domain patient separation", "binary labels",
                         "existing decoded-image integrity gate", "original BRSET image paths",
                         "three B1 checkpoint/prediction provenance chains"],
              "privacy": "Aggregate counts and hashes only; no file or patient identifiers."}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "strict_source_contract": strict_source_contract,
                      "cohorts": cohorts}, indent=2))


if __name__ == "__main__":
    main()
