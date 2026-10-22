"""Build a ZERO-mBRSET-LABEL classifier data folder (research/codex D099).

  data/<name>/train  synthetic images only (BRSET-train twins from --twins, optionally extra synthetic folders
                     via --cascade with their own labels_shard*.csv); labels = BRSET labels
  data/<name>/val    SYNTHETIC BRSET validation: --val_dir renderings of BRSET's own validation split (2,451 photos),
                     BRSET labels -> used for epoch and cut-off choice (no mBRSET label)
  data/<name>/test   -> real mBRSET test (data/finetune_joint/test), read once for the final score

Refuses if any train/val path is missing or outside brset_gen/synth.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
GEN = Path("/data/users3/sthummala2/brset_gen")
SYN = GEN / "synth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--twins", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--cascade", nargs="*", default=[])
    a = ap.parse_args()
    jt = pd.read_csv(ROOT / "data/finetune_joint/train/labels.csv")
    tr = jt[jt.domain == 0].copy()
    tr["file"] = [str(Path(a.twins) / (Path(f).stem + ".png")) for f in tr.file]
    parts = [tr[["file", "diabetic_retinopathy", "macular_edema", "domain"]]]
    for cd in a.cascade:
        lab = pd.concat([pd.read_csv(f) for f in sorted(Path(cd).glob("labels_shard*.csv"))], ignore_index=True)
        lab["domain"] = 1
        parts.append(lab[["file", "diabetic_retinopathy", "macular_edema", "domain"]])
    train = pd.concat(parts, ignore_index=True)
    hm = pd.read_csv(GEN / "cache/brset_heldout_meta.csv")
    hv = hm[hm.split == "val"]
    val = pd.DataFrame(dict(file=[str(Path(a.val_dir) / (Path(f).stem + ".png")) for f in hv.file],
                            diabetic_retinopathy=hv.dr.values, macular_edema=hv.edema.values, domain=0))
    for nm, df in (("train", train), ("val", val)):
        bad = [f for f in df.file if not Path(f).resolve().is_relative_to(SYN) or not Path(f).exists()]
        if bad:
            raise SystemExit(f"refusing: {len(bad)} {nm} paths missing or not synthetic, e.g. {bad[:3]}")
    out = ROOT / "data" / a.name
    for s in ("train", "val"):
        (out / s).mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train/labels.csv", index=False)
    val.to_csv(out / "val/labels.csv", index=False)
    if not (out / "test").exists():
        (out / "test").symlink_to(ROOT / "data/finetune_joint/test")
    print(f"{out}: train {len(train)} synthetic (DR+ {int(train.diabetic_retinopathy.sum())}); val {len(val)} synthetic BRSET "
          f"(DR+ {int(val.diabetic_retinopathy.sum())}); test -> real mBRSET (read once)", flush=True)


if __name__ == "__main__":
    main()
