"""Build a SYNTHETIC-ONLY classifier data folder (research/codex D091).

data/<name>/train/labels.csv lists ONLY synthetic images (absolute paths under brset_gen/synth);
data/<name>/val and data/<name>/test are symlinks to the real mBRSET validation and test splits
of data/finetune_joint (model selection / cut-off and the one-time test read, as in every arm).

Sources (any combination):
  --twins DIR     the BRSET-train twin of every BRSET training image (label = that image's label,
                  domain 0); every one must exist
  --cascade DIR   new cascade images (scripts/131; labels from its labels_shard*.csv, domain 1);
                  --exclude JSON drops the files the copy check flagged
The script refuses to write if any training path is outside brset_gen/synth or missing.
"""
import argparse, json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
SYN = Path("/data/users3/sthummala2/brset_gen/synth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--twins", default="")
    ap.add_argument("--cascade", nargs="*", default=[])
    ap.add_argument("--exclude", default="", help="copy-check JSON (full record) whose 'copies' are dropped")
    a = ap.parse_args()
    parts = []
    if a.twins:
        jt = pd.read_csv(ROOT / "data/finetune_joint/train/labels.csv")
        b = jt[jt.domain == 0].copy()
        b["file"] = [str(Path(a.twins) / (Path(f).stem + ".png")) for f in b.file]
        parts.append(b[["file", "diabetic_retinopathy", "macular_edema", "domain"]])
        print(f"twins: {len(b)} BRSET-train twins from {a.twins}", flush=True)
    drop = set()
    if a.exclude:
        rec = json.loads(Path(a.exclude).read_text())
        for s in rec["sets"].values():
            drop |= set(s.get("copies", []))
    for cd in a.cascade:
        lab = pd.concat([pd.read_csv(f) for f in sorted(Path(cd).glob("labels_shard*.csv"))], ignore_index=True)
        n0 = len(lab)
        lab = lab[~lab.file.map(lambda f: Path(f).name).isin(drop)]
        lab["domain"] = 1
        parts.append(lab[["file", "diabetic_retinopathy", "macular_edema", "domain"]])
        print(f"cascade: {len(lab)} of {n0} images from {cd} ({n0 - len(lab)} dropped by the copy check)", flush=True)
    df = pd.concat(parts, ignore_index=True)
    bad_root = [f for f in df.file if not Path(f).resolve().is_relative_to(SYN)]
    missing = [f for f in df.file if not Path(f).exists()]
    if bad_root or missing:
        raise SystemExit(f"refusing: {len(bad_root)} paths outside {SYN}, {len(missing)} missing "
                         f"(e.g. {(bad_root + missing)[:3]})")
    if df.file.duplicated().any():
        raise SystemExit("refusing: duplicated training paths")
    out = ROOT / "data" / a.name
    (out / "train").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "train/labels.csv", index=False)
    for s in ("val", "test"):
        link = out / s
        if not link.exists():
            link.symlink_to(ROOT / "data/finetune_joint" / s)
    print(f"{out}: {len(df)} synthetic training images (DR+ {int(df.diabetic_retinopathy.sum())}, "
          f"oedema {int(df.macular_edema.sum())}); val/test -> real mBRSET splits", flush=True)


if __name__ == "__main__":
    main()
