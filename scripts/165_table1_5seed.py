"""Final paper numbers (D109 reporting rule): 5-seed means (3 where an arm has only 3) of test AUROC, F1 at the stored
validation cut-off, sensitivity at 90 % specificity read off the test ROC, validation AUROC, and per-grade test AUROC vs
healthy plus operating-point counts. Descriptive; significance comes from results/extra_stats_2026-10-20_withcg_5seed.json.
Writes results/table1_5seed.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
ROOT = Path("/home/users/sthummala2/brset-convnextv2"); CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
rd = lambda t: (CLS if t.startswith(("syn_", "pst_", "mbo_", "cgtt_")) else ROOT / "results") / t
ARMS = {"B0": "lebase_b0_s{s}", "S11": "syn_both_b0_s{s}_p0.25", "BFDA": "syn_fda_b0_s{s}_p0.25", "BRH": "syn_reinhard_b0_s{s}_p0.25",
        "BCC": "syn_classcond_b0_s{s}_p0.25", "CGR25": "syn_cyclegan_residual_b0_s{s}_p0.25", "CGP25": "syn_cyclegan_plain_b0_s{s}_p0.25",
        "CGTTR": "cgtt_residual_s{s}", "CGTTI": "cgtt_ident_s{s}", "AQ1": "syn_physonly_b0_s{s}_p0.25", "AQ2": "syn_bothnone_b0_s{s}_p0.25",
        "AQ3": "syn_bothsrc_b0_s{s}_p0.25", "S11DC": "syn_both_dc_b0_s{s}_p0.25", "STF": "pst_F_base1loose_notwins_s{s}",
        "B250": "lebase_b250_s{s}", "B500": "lebase_b500_s{s}", "B1000": "lebase_b1000_s{s}", "MBO": "mbo_lebase_s{s}",
        "C1": "syn_both_dc_b0_s{s}_p1.0", "CGR100": "syn_cyclegan_residual_b0_s{s}_p1.0"}
import os
OUT = ROOT / "results/table1_5seed.json"
if os.environ.get("D113_FDA_RUN"):   # D113: add the FOV-restricted FDA arm and write a separate file; default unchanged
    ARMS["BFDAFOV"] = os.environ["D113_FDA_RUN"]; OUT = ROOT / "results/table1_5seed_fdafov.json"
raw = pd.read_csv("/home/users/sthummala2/mbrset-retfound/data/raw/labels_mbrset.csv").set_index("file")
tl = pd.read_csv(ROOT / "data/finetune_joint/test/labels.csv"); g = raw.loc[tl.file, "final_icdr"].astype(int).values
def f1(y, q):
    tp = (q & y).sum(); return 2 * tp / max(2 * tp + (q & ~y).sum() + (~q & y).sum(), 1)
out = {}
for k, t in ARMS.items():
    S = [s for s in range(5) if (rd(t.format(s=s)) / "test_predictions.npz").exists()]
    r = {m: [] for m in ("auc", "f1", "s90", "val", "g1", "g2", "g3", "g4", "miss_g1", "miss_g2p", "fa", "sens")}
    for s in S:
        z = np.load(rd(t.format(s=s)) / "test_predictions.npz"); y = z["y_true"][:, 0].astype(bool); p = z["y_prob"][:, 0]; c = float(z["thresholds"][0])
        v = np.load(rd(t.format(s=s)) / "val_predictions.npz")
        fpr, tpr, _ = roc_curve(y, p); q = p >= c
        r["auc"].append(roc_auc_score(y, p)); r["f1"].append(f1(y, q)); r["s90"].append(float(np.interp(0.10, fpr, tpr)))
        r["val"].append(roc_auc_score(v["y_true"][:, 0], v["y_prob"][:, 0]))
        for j in (1, 2, 3, 4):
            m = (g == 0) | (g == j); r[f"g{j}"].append(roc_auc_score(g[m] == j, p[m]))
        r["miss_g1"].append(int(((g == 1) & ~q).sum())); r["miss_g2p"].append(int(((g >= 2) & ~q).sum()))
        r["fa"].append(int(((g == 0) & q).sum())); r["sens"].append(float(q[g >= 1].mean()))
    out[k] = {"run": t, "n_seeds": len(S), **{m: float(np.mean(v)) for m, v in r.items()}, "auc_seeds": [float(x) for x in r["auc"]],
              "auc_min": float(min(r["auc"])), "auc_max": float(max(r["auc"]))}
    o = out[k]
    print(f"{k:6s} n={o['n_seeds']} test AUROC {o['auc']:.4f} [{o['auc_min']:.4f}-{o['auc_max']:.4f}] F1 {o['f1']:.4f} sens90 {o['s90']:.3f} val {o['val']:.4f} | "
          f"g1 {o['g1']:.3f} g2 {o['g2']:.3f} g3 {o['g3']:.3f} g4 {o['g4']:.3f} | miss g1 {o['miss_g1']:.1f}/40 g2+ {o['miss_g2p']:.1f}/119 FA {o['fa']:.1f}/573")
OUT.write_text(json.dumps(out, indent=1)); print("wrote", OUT)
