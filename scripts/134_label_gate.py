"""Label fidelity of NEW synthetic images (research/codex D091 gate ii).

Do the generated images show the disease they were asked for? Frozen judges score each image
(same eval transform, flip-4 TTA and weights as the trainer -- scripts/120 score()); reported
per judge: DR AUROC against the conditioning label, oedema AUROC, mean DR score per asked
grade. Judges: real mBRSET->mBRSET (mbo_lebase_s0) and BRSET-only (lebase_b0_s0). They are
diagnostics only -- never used to make or select training images beyond the pre-registered
choice of snapshot / guidance / strength (D091). Reference row: the same judges on the real
mBRSET VALIDATION photos (what a perfect generator would score).
"""
import argparse, importlib.util, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parent
sp = importlib.util.spec_from_file_location("P120", str(HERE / "120_predict.py"))
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)
JUDGES = {"mbrset_real": "/data/users3/sthummala2/brset_gen/cls_runs/mbo_lebase_s0",
          "brset_only": str(ROOT / "results/lebase_b0_s0")}


def summarise(prob, dr, me, grade):
    r = {"n": int(len(dr)), "dr_auroc": float(roc_auc_score(dr, prob[:, 0])) if 0 < dr.sum() < len(dr) else None,
         "me_auroc": float(roc_auc_score(me, prob[:, 1])) if 0 < me.sum() < len(me) else None}
    if grade is not None:
        r["mean_dr_score_by_grade"] = {int(g): float(prob[grade == g, 0].mean()) for g in np.unique(grade)}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth_dir", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--with_reference", type=int, default=1)
    a = ap.parse_args()
    out = {}
    if a.with_reference:
        v = pd.read_csv(ROOT / "data/finetune_joint/val/labels.csv")
        paths = [ROOT / "data/finetune_joint/val" / f for f in v.file]
        out["real_mbrset_val"] = {j: summarise(P.score(d, paths), v.diabetic_retinopathy.values,
                                               v.macular_edema.values, None) for j, d in JUDGES.items()}
        print("real mBRSET val", json.dumps(out["real_mbrset_val"]), flush=True)
    for sd in a.synth_dir:
        lab = pd.concat([pd.read_csv(f) for f in sorted(Path(sd).glob("labels_shard*.csv"))], ignore_index=True)
        lab = lab[[Path(f).exists() for f in lab.file]].reset_index(drop=True)
        res = {j: summarise(P.score(d, list(lab.file)), lab.diabetic_retinopathy.values,
                            lab.macular_edema.values, lab.grade.values) for j, d in JUDGES.items()}
        out[Path(sd).name] = res
        print(Path(sd).name, json.dumps(res), flush=True)
    Path(ROOT / f"results/labelgate_{a.tag}.json").write_text(json.dumps(out, indent=1))
    print("->", ROOT / f"results/labelgate_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
