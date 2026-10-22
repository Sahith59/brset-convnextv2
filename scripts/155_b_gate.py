"""D105/D105a experiment B validation gate (registered before any B output), evaluated on mBRSET VALIDATION only.
X-mild must reach, as 3-seed means: grade-1-vs-grade-0 AUROC >= 0.860 AND healthy mean P(DR) <= 0.15 AND healthy false-alarm
rate at the run's own validation-chosen cut-off <= 0.052 AND DR AUROC >= 0.9376. X-nat is reported with the same numbers
(never promoted). KB5 (diagnostic only): seed-0 X-mild healthy mean P(DR) > 0.20.
Only val_predictions.npz and the stored validation-chosen threshold (test_predictions.npz['thresholds']) are read; no test
prediction or metric is opened. Writes results/d105_B_gate.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/users/sthummala2/brset-convnextv2"); CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
raw = pd.read_csv("/home/users/sthummala2/mbrset-retfound/data/raw/labels_mbrset.csv").set_index("file")
vl = pd.read_csv(ROOT / "data/finetune_joint/val/labels.csv"); g = raw.loc[vl.file, "final_icdr"].astype(int).values
out = {}
for arm, tpl in (("XMILD", "syn_xmild2_s{s}"), ("XNAT", "syn_xnat2_s{s}"), ("C1PREV", "syn_c1prev_s{s}"), ("C1", "syn_both_dc_b0_s{s}_p1.0")):
    r = {"g1v0": [], "hmean": [], "fa": [], "auc": [], "g1sens": []}
    for s in (0, 1, 2):
        z = np.load(CLS / tpl.format(s=s) / "val_predictions.npz"); p = z["y_prob"][:, 0]; y = z["y_true"][:, 0].astype(bool)
        assert (y == (g >= 1)).all()
        thr = float(np.load(CLS / tpl.format(s=s) / "test_predictions.npz")["thresholds"][0])
        m = g <= 1
        r["g1v0"].append(float(roc_auc_score(g[m] == 1, p[m]))); r["hmean"].append(float(p[g == 0].mean()))
        r["fa"].append(float((p[g == 0] >= thr).mean())); r["auc"].append(float(roc_auc_score(y, p))); r["g1sens"].append(float((p[g == 1] >= thr).mean()))
    r["mean"] = {k: float(np.mean(v)) for k, v in r.items() if isinstance(v, list)}
    out[arm] = r
mx = out["XMILD"]["mean"]
out["gate"] = {"g1v0>=0.860": mx["g1v0"] >= 0.860, "hmean<=0.15": mx["hmean"] <= 0.15, "fa<=0.052": mx["fa"] <= 0.052, "auc>=0.9376": mx["auc"] >= 0.9376}
out["gate_pass"] = all(out["gate"].values())
out["KB5_seed0_hmean_gt_0.20"] = out["XMILD"]["hmean"][0] > 0.20
for arm in ("XMILD", "XNAT", "C1PREV", "C1"):
    m_ = out[arm]["mean"]
    print(f"{arm:7s} val DR AUROC {m_['auc']:.4f} | grade1-vs-0 AUROC {m_['g1v0']:.3f} (seeds {' / '.join(f'{v:.3f}' for v in out[arm]['g1v0'])}) | "
          f"healthy mean P {m_['hmean']:.3f} | healthy FA {m_['fa']:.3f} | grade-1 sensitivity {m_['g1sens']:.3f}", flush=True)
print("gate", json.dumps(out["gate"]), "PASS" if out["gate_pass"] else "FAIL", "| KB5", out["KB5_seed0_hmean_gt_0.20"], flush=True)
(ROOT / "results/d105_B_gate.json").write_text(json.dumps(out, indent=1))
