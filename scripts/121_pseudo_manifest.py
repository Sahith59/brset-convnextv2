"""Teacher scores on UNLABELLED mBRSET train images -> confident pseudo-labels.

Keeps an image only if the teacher is confident: DR score < --neg (pseudo-negative)
or > --pos (pseudo-positive); everything in between is left out, so the ambiguous
mild cases the teacher tends to miss are never taught as healthy. The oedema term
is masked (teacher oedema labels are not used). Thresholds come from VALIDATION
purity of the same teacher (research/codex D073), never from test.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True, help="npz from scripts/120_predict.py")
ap.add_argument("--seeds", default="all", help="'all' or comma list of teacher indices to average")
ap.add_argument("--neg", type=float, required=True)
ap.add_argument("--pos", type=float, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = np.load(a.scores, allow_pickle=True)
P = d["prob"]                                         # (teachers, n, 2)
if a.seeds != "all":
    P = P[[int(i) for i in a.seeds.split(",")]]
s = P.mean(0)[:, 0]
keep = (s < a.neg) | (s > a.pos)
files = [str(ROOT / "data/finetune_joint/train" / f) for f in d["files"]]
m = pd.DataFrame({"file": np.array(files)[keep], "diabetic_retinopathy": (s[keep] > a.pos).astype(int),
                  "macular_edema": 0, "_mdr": 1.0, "_med": 0.0, "teacher_score": s[keep]})
m.to_csv(a.out, index=False)
print(f"{a.scores} teachers={P.shape[0]}: kept {keep.sum()} of {len(s)} ({keep.mean():.0%}); "
      f"pseudo DR+ {int((s > a.pos).sum())}, pseudo DR- {int((s < a.neg).sum())}, left out {int((~keep).sum())} -> {a.out}")
