"""D111 read-out (registered 22 Oct 2026 before any classifier output; amended at about 12:55 UTC, still before any output).

Same-camera re-rendering controls: the BRSET twins rendered with the renderer's camera token set to BRSET instead of mBRSET.
  CTRLB  camera-model start, BRSET token  (our exact pipeline; only the camera target differs)  syn_bothphysbr_b0_s{s}_p0.25
  CTRLA  raw-photo start,   BRSET token  (the registered control)                                syn_bothsrcbr_b0_s{s}_p0.25
Table-1 row metrics exactly as scripts/165_table1_5seed.py (5-seed means of test AUROC, F1 at the stored validation cut-off,
sensitivity at 90 % specificity, validation AUROC, per-grade test AUROC vs healthy). Contrasts exactly as
scripts/166_d110_selftrain_stats.py: 2,000 resamples of the 193 test patients (rng 1, shared by all arms) and, per arm, a
resample of its seeds with replacement (rng 2), so the two-level interval includes training-seed variation.
  Pre-declared, Holm over the two (two-level p): S11 - CTRLB, S11 - CTRLA.
  Descriptive: AQ3 - CTRLA (token effect at raw start), CTRLB - B0, CTRLA - B0, S11 - B0 (consistency check).
Reading rule (DECISIONS D111): two-level 95 % CI above 0 -> phone targeting adds to re-rendering; includes 0 -> not
distinguishable; below 0 -> same-camera re-rendering did better. Outside the Holm-7 family.
Writes results/d111_samecam.json (results/table1_5seed.json is not touched). CPU only: run through Slurm.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path("/home/users/sthummala2/brset-convnextv2"); CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
rd = lambda t: (CLS if t.startswith(("syn_", "pst_", "mbo_", "cgtt_")) else ROOT / "results") / t
ARMS = {"B0": "lebase_b0_s{s}", "S11": "syn_both_b0_s{s}_p0.25", "AQ3": "syn_bothsrc_b0_s{s}_p0.25",
        "CTRLB": "syn_bothphysbr_b0_s{s}_p0.25", "CTRLA": "syn_bothsrcbr_b0_s{s}_p0.25"}
PRIMARY = [("S11", "CTRLB"), ("S11", "CTRLA")]
DESCRIPTIVE = [("AQ3", "CTRLA"), ("CTRLB", "B0"), ("CTRLA", "B0"), ("S11", "B0")]
SEEDS = [0, 1, 2, 3, 4]
NB = 2000


def f1(y, q):
    tp = (q & y).sum(); return 2 * tp / max(2 * tp + (q & ~y).sum() + (~q & y).sum(), 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "results/d111_samecam.json"))
    ap.add_argument("--dry", action="store_true", help="pipeline test only: stand existing arms in for the two controls")
    a = ap.parse_args()
    if a.dry:
        ARMS.update(CTRLB="syn_physonly_b0_s{s}_p0.25", CTRLA="syn_bothsrc_b0_s{s}_p0.25")
    raw = pd.read_csv("/home/users/sthummala2/mbrset-retfound/data/raw/labels_mbrset.csv").set_index("file")
    tl = pd.read_csv(ROOT / "data/finetune_joint/test/labels.csv"); g = raw.loc[tl.file, "final_icdr"].astype(int).values
    pid = tl.file.astype(str).str.split(".").str[0].values
    pats = np.unique(pid); ix = {p: np.where(pid == p)[0] for p in pats}
    yt = np.load(rd("lebase_b0_s0") / "test_predictions.npz")["y_true"][:, 0].astype(bool)
    rng = np.random.default_rng(1)
    BOOT = [np.concatenate([ix[p] for p in rng.choice(pats, len(pats), replace=True)]) for _ in range(NB)]
    rng2 = np.random.default_rng(2)
    A, rows = {}, {}
    for k, t in ARMS.items():
        S = [s for s in SEEDS if (rd(t.format(s=s)) / "test_predictions.npz").exists()]
        if len(S) < 5:
            raise SystemExit(f"{k}: only seeds {S} exist; D111 is read once, with all 5 seeds")
        r = {m: [] for m in ("auc", "f1", "s90", "val", "g1", "g2", "g3", "g4", "miss_g1", "miss_g2p", "fa")}
        P, Q = [], []
        for s in S:
            z = np.load(rd(t.format(s=s)) / "test_predictions.npz"); y = z["y_true"][:, 0].astype(bool); p = z["y_prob"][:, 0]
            assert (y == yt).all(), k
            c = float(z["thresholds"][0]); q = p >= c; v = np.load(rd(t.format(s=s)) / "val_predictions.npz")
            fpr, tpr, _ = roc_curve(y, p)
            r["auc"].append(roc_auc_score(y, p)); r["f1"].append(f1(y, q)); r["s90"].append(float(np.interp(0.10, fpr, tpr)))
            r["val"].append(roc_auc_score(v["y_true"][:, 0], v["y_prob"][:, 0]))
            for j in (1, 2, 3, 4):
                m = (g == 0) | (g == j); r[f"g{j}"].append(roc_auc_score(g[m] == j, p[m]))
            r["miss_g1"].append(int(((g == 1) & ~q).sum())); r["miss_g2p"].append(int(((g >= 2) & ~q).sum())); r["fa"].append(int(((g == 0) & q).sum()))
            P.append(p); Q.append(q)
        rows[k] = {"run": t, "seeds": S, **{m: float(np.mean(v)) for m, v in r.items()}, "auc_seeds": [float(x) for x in r["auc"]],
                   "auc_min": float(min(r["auc"])), "auc_max": float(max(r["auc"]))}
        A[k] = dict(auc=np.array([[roc_auc_score(yt[b], P[i][b]) for b in BOOT] for i in range(len(S))]),
                    f1=np.array([[f1(yt[b], Q[i][b]) for b in BOOT] for i in range(len(S))]),
                    draw=rng2.integers(0, len(S), size=(NB, len(S))))
        o = rows[k]
        print(f"{k:6s} AUROC {o['auc']:.4f} [{o['auc_min']:.4f}-{o['auc_max']:.4f}] F1 {o['f1']:.4f} sens90 {o['s90']:.3f} val {o['val']:.4f} "
              f"g1 {o['g1']:.3f} g2 {o['g2']:.3f} g3 {o['g3']:.3f} g4 {o['g4']:.3f}", flush=True)

    def contrast(x, y):
        c = {"x": x, "y": y, "dauc": rows[x]["auc"] - rows[y]["auc"], "df1": rows[x]["f1"] - rows[y]["f1"]}
        for m in ("auc", "f1"):
            pat = A[x][m].mean(0) - A[y][m].mean(0)
            two = np.array([A[x][m][A[x]["draw"][i], i].mean() - A[y][m][A[y]["draw"][i], i].mean() for i in range(NB)])
            for name, d in (("patient", pat), ("two_level", two)):
                c[f"{m}_{name}_CI"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
                c[f"{m}_{name}_p"] = float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        lo, hi = c["auc_two_level_CI"]
        c["reading"] = "above 0" if lo > 0 else ("below 0" if hi < 0 else "includes 0")
        return c

    prim = [contrast(x, y) for x, y in PRIMARY]
    for m in ("auc", "f1"):   # Holm over the two pre-declared contrasts, two-level p
        order = np.argsort([c[f"{m}_two_level_p"] for c in prim]); run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (len(prim) - rank) * prim[i][f"{m}_two_level_p"])); prim[i][f"{m}_holm2_p"] = run
    desc = [contrast(x, y) for x, y in DESCRIPTIVE]
    for c in prim + desc:
        print(f"{c['x']} - {c['y']}: AUROC {c['dauc']:+.4f} two-level [{c['auc_two_level_CI'][0]:+.4f}, {c['auc_two_level_CI'][1]:+.4f}] "
              f"p {c['auc_two_level_p']:.3f}{(' Holm-2 %.3f' % c['auc_holm2_p']) if 'auc_holm2_p' in c else ''} -> {c['reading']} | "
              f"F1 {c['df1']:+.4f} two-level [{c['f1_two_level_CI'][0]:+.4f}, {c['f1_two_level_CI'][1]:+.4f}]", flush=True)
    out = {"rule": "D111 (DECISIONS.md): two-level (seed x patient) bootstrap, 2,000 resamples, patients rng 1, seeds rng 2; "
                   "Holm over S11-CTRLB and S11-CTRLA; descriptive contrasts unadjusted; outside Holm-7",
           "rows": rows, "primary": prim, "descriptive": desc}
    Path(a.out).write_text(json.dumps(out, indent=1)); print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
