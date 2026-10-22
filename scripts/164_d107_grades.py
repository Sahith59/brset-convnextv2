"""D107 descriptive (not part of any Holm family): per-grade test AUROC vs healthy and operating-point counts at each
run's stored validation cut-off, 3-seed means, for BRSET-only, ours (S11), residual CycleGAN in the 25 % mix, standard
self-training, and real phone training. Grades from labels_mbrset.csv final_icdr. Writes results/d107_grades.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path("/home/users/sthummala2/brset-convnextv2"); CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
rd = lambda t: (CLS if t.startswith(("syn_", "pst_", "mbo_", "cgtt_")) else ROOT / "results") / t
ARMS = {"B0": "lebase_b0_s{s}", "S11": "syn_both_b0_s{s}_p0.25", "CGR25": "syn_cyclegan_residual_b0_s{s}_p0.25",
        "STF": "pst_F_base1loose_notwins_s{s}", "MBO": "mbo_lebase_s{s}"}
raw = pd.read_csv("/home/users/sthummala2/mbrset-retfound/data/raw/labels_mbrset.csv").set_index("file")
tl = pd.read_csv(ROOT / "data/finetune_joint/test/labels.csv"); g = raw.loc[tl.file, "final_icdr"].astype(int).values
out = {}
for k, t in ARMS.items():
    r = {f"g{j}_auc": [] for j in (1, 2, 3, 4)}; r.update(missed_g1=[], missed_g2plus=[], false_alarms=[], sens=[])
    for s in (0, 1, 2):
        z = np.load(rd(t.format(s=s)) / "test_predictions.npz"); p = z["y_prob"][:, 0]; c = float(z["thresholds"][0])
        for j in (1, 2, 3, 4):
            m = (g == 0) | (g == j); r[f"g{j}_auc"].append(float(roc_auc_score(g[m] == j, p[m])))
        q = p >= c
        r["missed_g1"].append(int(((g == 1) & ~q).sum())); r["missed_g2plus"].append(int(((g >= 2) & ~q).sum()))
        r["false_alarms"].append(int(((g == 0) & q).sum())); r["sens"].append(float(q[g >= 1].mean()))
    out[k] = {kk: float(np.mean(v)) for kk, v in r.items()}
    o = out[k]
    print(f"{k:6s} AUROC vs healthy g1 {o['g1_auc']:.3f} g2 {o['g2_auc']:.3f} g3 {o['g3_auc']:.3f} g4 {o['g4_auc']:.3f} | "
          f"at cut-off: missed g1 {o['missed_g1']:.1f}/40, missed g2+ {o['missed_g2plus']:.1f}/119, false alarms {o['false_alarms']:.1f}/573, sens {o['sens']:.3f}")
(ROOT / "results/d107_grades.json").write_text(json.dumps(out, indent=1))
