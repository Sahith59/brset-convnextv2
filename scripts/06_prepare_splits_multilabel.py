"""
Prepare full BRSET for multi-label binary classification of two specific
findings: diabetic_retinopathy and macular_edema (both 1=present/0=absent,
independent of the DR_ICDR severity scale used in the earlier 5-class work).

Per Dr. Ye's revised direction: use ALL 16,266 images (matching the original
BRSET paper's own inclusion criteria, no quality filter), minus only images
confirmed corrupted/unreadable via a full-dataset integrity scan. Patient-level
70/15/15 split, stratified on the joint (diabetic_retinopathy, macular_edema)
category so both labels' prevalence is preserved across splits.

Multi-label doesn't fit torchvision's ImageFolder (one folder per class), so
images are symlinked into a flat per-split directory and a labels CSV per
split maps filename -> [diabetic_retinopathy, macular_edema].
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_CSV = Path("/data/users4/nshaik3/Datasets/BRSET/physionet.org/files/brazilian-ophthalmological/1.0.1/labels_brset.csv")
RAW_IMAGES = Path("/data/users4/nshaik3/Datasets/BRSET/physionet.org/files/brazilian-ophthalmological/1.0.1/fundus_photos")
CORRUPTED_FILES_LIST = Path("/home/users/sthummala2/brset-convnextv2/results/corrupted_files_full16266.txt")
OUT_DIR = Path("/home/users/sthummala2/brset-convnextv2/data/finetune_multilabel")
SPLITS_CSV = Path("/home/users/sthummala2/brset-convnextv2/results/splits_multilabel.csv")

LABEL_COLS = ["diabetic_retinopathy", "macular_edema"]
SEED = 42


def build_dataframe():
    df = pd.read_csv(RAW_CSV)
    n_total = len(df)
    df["file"] = df["image_id"].astype(str) + ".jpg"

    n_corrupted = 0
    if CORRUPTED_FILES_LIST.exists():
        corrupted = {line.split("\t")[0] for line in CORRUPTED_FILES_LIST.read_text().splitlines() if line.strip()}
        n_before = len(df)
        df = df[~df["file"].isin(corrupted)].copy()
        n_corrupted = n_before - len(df)

    print(f"total images:              {n_total}")
    print(f"dropped corrupted files:   {n_corrupted}")
    print(f"remaining images:          {len(df)}")
    print(f"unique patients:           {df['patient_id'].nunique()}")
    for col in LABEL_COLS:
        print(f"{col}: positive={int(df[col].sum())} ({df[col].mean()*100:.2f}%)")
    return df


def patient_split(df):
    # stratify on the joint (dr, me) category per patient (patient-level max of each)
    patient_labels = df.groupby("patient_id")[LABEL_COLS].max()
    strat_key = patient_labels[LABEL_COLS[0]].astype(str) + "_" + patient_labels[LABEL_COLS[1]].astype(str)

    patients = patient_labels.index.to_numpy()
    train_p, rest_p, train_k, rest_k = train_test_split(
        patients, strat_key, test_size=0.30, stratify=strat_key, random_state=SEED
    )
    val_p, test_p, _, _ = train_test_split(
        rest_p, rest_k, test_size=0.50, stratify=rest_k, random_state=SEED
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
    for col in LABEL_COLS:
        print(f"\n{col} positive rate by split:")
        print(df.groupby("split")[col].mean())
    return df


def materialize(df, dry_run=False):
    for split in ("train", "val", "test"):
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    n_linked = 0
    for row in df.itertuples():
        src = RAW_IMAGES / row.file
        dst = OUT_DIR / row.split / row.file
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = build_dataframe()
    df = patient_split(df)

    SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[["patient_id", "file"] + LABEL_COLS + ["split"]].to_csv(SPLITS_CSV, index=False)
    print(f"\nwrote split manifest to {SPLITS_CSV}")

    materialize(df, dry_run=args.dry_run)

    # per-split label CSVs (what the training script actually reads)
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split][["file"] + LABEL_COLS]
        sub.to_csv(OUT_DIR / split / "labels.csv", index=False)


if __name__ == "__main__":
    main()
