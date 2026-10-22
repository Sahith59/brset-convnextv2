"""Eye-level, agreement-filtered pseudo-labels for UNLABELLED mBRSET train images (research/codex D079).

Teacher score per photo = mean over the teacher seeds, then mean over the photos of the same eye
(patient + laterality: acquisition metadata, read with usecols so no diagnosis column is parsed).
A photo is kept only if the eye-level score is confident (< --neg or > --pos) AND every seed's own
eye-level score falls on the same side. Oedema masked, as in 121.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
MB = "/data/users4/nshaik3/Datasets/mBRSET/physionet.org/files/mbrset/1.0/labels_mbrset.csv"
ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True); ap.add_argument("--neg", type=float, required=True)
ap.add_argument("--pos", type=float, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
d = np.load(a.scores, allow_pickle=True); P = d["prob"][:, :, 0]               # (teachers, n)
meta = pd.read_csv(MB, usecols=["file", "patient", "laterality"])
assert sorted(meta.columns) == ["file", "laterality", "patient"], "no diagnosis column may be parsed"
df = pd.DataFrame({"file": d["files"]}).merge(meta, on="file", how="left")
assert df.patient.notna().all()
eye = df.patient.astype(str) + "_" + df.laterality.astype(str)
E = np.stack([pd.Series(p).groupby(eye.values).transform("mean").values for p in P])   # per-seed eye scores
s = E.mean(0)
neg = (s < a.neg) & (E < 0.5).all(0); pos = (s > a.pos) & (E > 0.5).all(0)
keep = neg | pos
files = [str(ROOT / "data/finetune_joint/train" / f) for f in df.file]
m = pd.DataFrame({"file": np.array(files)[keep], "diabetic_retinopathy": pos[keep].astype(int), "macular_edema": 0,
                  "_mdr": 1.0, "_med": 0.0, "teacher_score": s[keep]})
m.to_csv(a.out, index=False)
print(f"{a.scores}: {df.patient.nunique()} patients, {eye.nunique()} eyes; kept {keep.sum()} of {len(s)} ({keep.mean():.0%}); "
      f"pseudo DR+ {int(pos.sum())}, pseudo DR- {int(neg.sum())} -> {a.out}")
