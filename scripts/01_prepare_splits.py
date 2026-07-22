"""
Prepare full BRSET for ConvNeXt V2 fine-tuning: filter to quality-passing images,
use DR_ICDR (0-4) directly as the label, split by patient (not image) to avoid
leakage, and materialize train/val/test/<class>/ folders of symlinks.

Mirrors the patient-level splitting methodology used for mBRSET in the sibling
RETFound project (../mbrset-retfound/scripts/01_prepare_splits.py), applied here
to the full BRSET dataset supplied by Nagur Shareef Shaik.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_CSV = Path("/data/users4/nshaik3/Datasets/BRSET/physionet.org/files/brazilian-ophthalmological/1.0.1/labels_brset.csv")
RAW_IMAGES = Path("/data/users4/nshaik3/Datasets/BRSET/physionet.org/files/brazilian-ophthalmological/1.0.1/fundus_photos")
OUT_DIR = Path("/home/users/sthummala2/brset-convnextv2/data/finetune_icdr5")
SPLITS_CSV = Path("/home/users/sthummala2/brset-convnextv2/results/splits.csv")
# Found via a full-dataset PIL verify/load pass (scripts/00_check_corruption.py-style scan);
# these 7 files raise "broken data stream" errors and crash the DataLoader if included.
CORRUPTED_FILES_LIST = Path("/home/users/sthummala2/brset-convnextv2/results/corrupted_files.txt")

CLASS_NAMES = {
    0: "0_no_dr",
    1: "1_mild_npdr",
    2: "2_moderate_npdr",
    3: "3_severe_npdr",
    4: "4_proliferative_dr",
}
SEED = 42


def build_dataframe():
    df = pd.read_csv(RAW_CSV)
    n_total = len(df)

    df = df[df["quality"] == "Adequate"].copy()
    n_quality = len(df)

    df = df.dropna(subset=["DR_ICDR"]).copy()
    df["label"] = df["DR_ICDR"].astype(int)
    n_labeled = len(df)

    df["file"] = df["image_id"].astype(str) + ".jpg"

    n_corrupted = 0
    if CORRUPTED_FILES_LIST.exists():
        corrupted = {line.split("\t")[0] for line in CORRUPTED_FILES_LIST.read_text().splitlines() if line.strip()}
        n_before = len(df)
        df = df[~df["file"].isin(corrupted)].copy()
        n_corrupted = n_before - len(df)

    print(f"total images:              {n_total}")
    print(f"after quality filter:      {n_quality}")
    print(f"after dropping unlabeled:  {n_labeled}")
    print(f"dropped corrupted files:   {n_corrupted}")
    print(f"unique patients:           {df['patient_id'].nunique()}")
    print(df["label"].value_counts(normalize=True).sort_index().rename("image-level balance"))
    return df


def patient_split(df):
    patient_label = df.groupby("patient_id")["label"].max()
    patients = patient_label.index.to_numpy()
    labels = patient_label.to_numpy()

    train_p, rest_p, train_y, rest_y = train_test_split(
        patients, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    val_p, test_p, _, _ = train_test_split(
        rest_p, rest_y, test_size=0.50, stratify=rest_y, random_state=SEED
    )

    split_of = {}
    for p in train_p:
        split_of[p] = "train"
    for p in val_p:
        split_of[p] = "val"
    for p in test_p:
        split_of[p] = "test"

    df["split"] = df["patient_id"].map(split_of)
    print()
    print(df.groupby("split")["patient_id"].nunique().rename("patients per split"))
    print(df.groupby(["split", "label"]).size().unstack())
    return df


def materialize(df, dry_run=False):
    for split in ("train", "val", "test"):
        for cls in CLASS_NAMES.values():
            (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    n_linked = 0
    for row in df.itertuples():
        src = RAW_IMAGES / row.file
        dst = OUT_DIR / row.split / CLASS_NAMES[row.label] / row.file
        if not src.exists():
            print(f"WARNING: missing source image {src}")
            continue
        if dry_run:
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        n_linked += 1
    print(f"\nsymlinked {n_linked} images into {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="compute split, skip symlinking")
    args = parser.parse_args()

    df = build_dataframe()
    df = patient_split(df)

    SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[["patient_id", "file", "DR_ICDR", "label", "split"]].to_csv(SPLITS_CSV, index=False)
    print(f"\nwrote split manifest to {SPLITS_CSV}")

    materialize(df, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
