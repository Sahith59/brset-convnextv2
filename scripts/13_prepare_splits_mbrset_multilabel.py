"""
Prepare mBRSET for the same multi-label binary task as BRSET: diabetic
retinopathy presence and macular edema presence (0/1 each).

mBRSET's schema differs from full BRSET's: there is no standalone
diabetic_retinopathy flag, only final_icdr (the 0-4 ICDR severity grade) and
final_edema (yes/no). Verified against BRSET's own data that its
diabetic_retinopathy column corresponds almost exactly to DR_ICDR >= 1 (a
99%+ match), so the same rule is applied here: diabetic_retinopathy =
(final_icdr >= 1). final_edema maps directly to macular_edema (yes/no -> 1/0).

Unlike BRSET (where quality filtering was skipped specifically to match that
paper's stated "all images" methodology), there is no equivalent published
benchmark forcing that choice for mBRSET, and the existing mbrset-retfound
project already used final_quality=="yes" consistently — so that filter is
applied here for consistency with that precedent.

Same patient-level 70/15/15 stratified split methodology, same flat
per-split symlink + labels.csv structure as scripts/06_prepare_splits_multilabel.py.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_CSV = Path("/home/users/sthummala2/mbrset-retfound/data/raw/labels_mbrset.csv")
RAW_IMAGES = Path("/home/users/sthummala2/mbrset-retfound/data/raw/images")
OUT_DIR = Path("/home/users/sthummala2/brset-convnextv2/data/finetune_mbrset_multilabel")
SPLITS_CSV = Path("/home/users/sthummala2/brset-convnextv2/results/splits_mbrset_multilabel.csv")

LABEL_COLS = ["diabetic_retinopathy", "macular_edema"]
SEED = 42


def build_dataframe():
    df = pd.read_csv(RAW_CSV)
    n_total = len(df)

    df = df[df["final_quality"] == "yes"].copy()
    n_quality = len(df)

    df["diabetic_retinopathy"] = (df["final_icdr"] >= 1).astype("Int64")
    df.loc[df["final_icdr"].isna(), "diabetic_retinopathy"] = pd.NA
    df["macular_edema"] = df["final_edema"].map({"yes": 1, "no": 0})

    df = df.dropna(subset=LABEL_COLS).copy()
    df["diabetic_retinopathy"] = df["diabetic_retinopathy"].astype(int)
    df["macular_edema"] = df["macular_edema"].astype(int)
    n_labeled = len(df)

    print(f"total images:              {n_total}")
    print(f"after quality filter:      {n_quality}")
    print(f"after dropping unlabeled:  {n_labeled}")
    print(f"unique patients:           {df['patient'].nunique()}")
    for col in LABEL_COLS:
        print(f"{col}: positive={int(df[col].sum())} ({df[col].mean()*100:.2f}%)")
    return df


def patient_split(df):
    # mBRSET has far fewer patients than BRSET (1,285 vs 8,524), so stratifying
    # on the joint (dr, me) category is too fine-grained - some combinations end
    # up with only 1-2 patients after the first split, which sklearn can't
    # stratify further. Stratify on diabetic_retinopathy (the primary, more
    # prevalent label) alone instead; DR and ME are highly correlated in this
    # data (per the BRSET cross-tab), so ME balance across splits should still
    # hold reasonably well - verified in the printed per-split rates below.
    patient_labels = df.groupby("patient")[LABEL_COLS].max()
    strat_key = patient_labels[LABEL_COLS[0]].astype(str)

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

    df["split"] = df["patient"].map(split_of)
    print()
    print(df.groupby("split")["patient"].nunique().rename("patients per split"))
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
    df[["patient", "file"] + LABEL_COLS + ["split"]].to_csv(SPLITS_CSV, index=False)
    print(f"\nwrote split manifest to {SPLITS_CSV}")

    materialize(df, dry_run=args.dry_run)

    for split in ("train", "val", "test"):
        sub = df[df["split"] == split][["file"] + LABEL_COLS]
        sub.to_csv(OUT_DIR / split / "labels.csv", index=False)


if __name__ == "__main__":
    main()
