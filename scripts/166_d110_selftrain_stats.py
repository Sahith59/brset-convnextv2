"""D110 read-out (registered 21 Oct 2026, before any seed-3/4 output): self-training contrasts at 5 seeds.

Same bootstrap as scripts/162_extra_stats.py: 2,000 resamples of the 193 mBRSET test patients (rng seed 1, shared by
both arms) and, independently per arm, a resample of its seeds with replacement (rng seed 2), so the two-level interval
includes training-seed variation. Contrasts (descriptive context, OUTSIDE the Holm-7 family):
  STG - STH   twins inside self-training, same teacher (the "do not stack" contrast)
  STG - STF   twin-trained teacher + twins vs standard self-training
  STB - S11   best validation-chosen self-training variant vs ours
  STB - STF   best variant vs standard self-training
  STF - S11   standard self-training vs ours (unchanged from D109; consistency check)
Writes results/d110_selftrain_5seed.json. CPU only, saved predictions only: run through Slurm (srun -p qTRDGPU --gres=gpu:0).
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/users/sthummala2/brset-convnextv2"); CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
rd = lambda t: (CLS if t.startswith(("syn_", "pst_", "mbo_", "cgtt_")) else ROOT / "results") / t
ARMS = {"S11": "syn_both_b0_s{s}_p0.25", "STF": "pst_F_base1loose_notwins_s{s}", "STG": "pst_G_syn1loose_twins_s{s}",
        "STH": "pst_H_syn1loose_notwins_s{s}", "STB": "pst_B_syn3loose_s{s}"}
PAIRS = [("STG", "STH"), ("STG", "STF"), ("STB", "S11"), ("STB", "STF"), ("STF", "S11")]
SEEDS = [0, 1, 2, 3, 4]
NB = 2000


def ld(t):
    d = np.load(rd(t) / "test_predictions.npz")
    return d["y_true"][:, 0].astype(bool), d["y_prob"][:, 0], float(d["thresholds"][0])


def f1(y, q):
    tp = (q & y).sum(); return 2 * tp / max(2 * tp + (q & ~y).sum() + (~q & y).sum(), 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "results/d110_selftrain_5seed.json")); a = ap.parse_args()
    tl = pd.read_csv(ROOT / "data/finetune_joint/test/labels.csv"); pid = tl.file.astype(str).str.split(".").str[0].values
    pats = np.unique(pid); ix = {p: np.where(pid == p)[0] for p in pats}; yt = ld("lebase_b0_s0")[0]
    rng = np.random.default_rng(1)
    BOOT = [np.concatenate([ix[p] for p in rng.choice(pats, len(pats), replace=True)]) for _ in range(NB)]
    rng2 = np.random.default_rng(2)
    A = {}
    for k, t in ARMS.items():
        SK = [s for s in SEEDS if (rd(t.format(s=s)) / "test_predictions.npz").exists()]
        P, Q = [], []
        for s in SK:
            y, p, thr = ld(t.format(s=s)); assert (y == yt).all(), k
            P.append(p); Q.append(p >= thr)
        NK = len(SK)
        A[k] = dict(seeds_used=SK, auc=np.array([[roc_auc_score(yt[b], P[i][b]) for b in BOOT] for i in range(NK)]),
                    f1=np.array([[f1(yt[b], Q[i][b]) for b in BOOT] for i in range(NK)]),
                    full_auc=[roc_auc_score(yt, P[i]) for i in range(NK)], full_f1=[f1(yt, Q[i]) for i in range(NK)],
                    draw=rng2.integers(0, NK, size=(NB, NK)))
        print(f"{k}: {NK} seeds {SK}; AUROC {np.mean(A[k]['full_auc']):.4f} per seed {[round(v, 4) for v in A[k]['full_auc']]}; "
              f"F1 {np.mean(A[k]['full_f1']):.4f}", flush=True)
    out = {"rule": "D110: two-level (seed x patient) bootstrap, 2,000 resamples, patients rng 1, seeds rng 2; descriptive, outside Holm-7",
           "arms": {k: {"run": ARMS[k], "seeds": A[k]["seeds_used"], "auc": float(np.mean(A[k]["full_auc"])),
                        "f1": float(np.mean(A[k]["full_f1"])), "auc_seeds": [float(v) for v in A[k]["full_auc"]]} for k in A},
           "pairs": []}
    for x, y in PAIRS:
        c = {"x": x, "y": y, "dauc": float(np.mean(A[x]["full_auc"]) - np.mean(A[y]["full_auc"])),
             "df1": float(np.mean(A[x]["full_f1"]) - np.mean(A[y]["full_f1"]))}
        for m in ("auc", "f1"):
            pat = A[x][m].mean(0) - A[y][m].mean(0)
            two = np.array([A[x][m][A[x]["draw"][i], i].mean() - A[y][m][A[y]["draw"][i], i].mean() for i in range(NB)])
            for name, d in (("patient", pat), ("two_level", two)):
                c[f"{m}_{name}_CI"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
                c[f"{m}_{name}_p"] = float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        c["comparable_auc"] = bool(c["auc_two_level_CI"][0] <= 0 <= c["auc_two_level_CI"][1])
        out["pairs"].append(c)
        print(f"{x} - {y}: AUROC {c['dauc']:+.4f} two-level [{c['auc_two_level_CI'][0]:+.4f}, {c['auc_two_level_CI'][1]:+.4f}] "
              f"p {c['auc_two_level_p']:.3f} | patient [{c['auc_patient_CI'][0]:+.4f}, {c['auc_patient_CI'][1]:+.4f}] | "
              f"F1 {c['df1']:+.4f} two-level [{c['f1_two_level_CI'][0]:+.4f}, {c['f1_two_level_CI'][1]:+.4f}]", flush=True)
    p = Path(a.out); p.write_text(json.dumps(out, indent=1)); print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
