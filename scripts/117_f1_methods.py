"""Cut-off and ensembling methods for DR F1, chosen on VALIDATION, then one test read.

Methods (all deterministic given the validation set; none looks at test):
  current      the trainer's own cut-off (bootstrap-median argmax-F1 on validation)
  argmax       plain F1-optimal cut-off on validation
  prevalence   cut-off flagging the validation positive rate
  ensemble+X   average the 3 seeds' probabilities, then method X
Held-out estimate: 400 patient-split halves inside validation (tune on one half,
score on the other). The method with the best held-out validation F1, pooled over
ALL arms, is applied to ALL arms on test once.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/users/sthummala2/brset-convnextv2"); CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
SEEDS = (0, 1, 2); GRID = np.arange(0.05, 0.95, 0.01)
ARMS = ["lebase_b0", "syn_tgt_b0", "syn_both_b0", "lebase_b250", "syn_tgt_b250", "syn_both_b250", "lebase_b500"]
rd = lambda t: CLS / t if t.startswith("syn_") else ROOT / "results" / t
def load(t, split):
    d = np.load(rd(t) / f"{split}_predictions.npz"); return d["y_true"][:, 0].astype(int), d["y_prob"][:, 0]
def f1_curve(y, p):
    pr = p[:, None] >= GRID[None, :]; tp = (pr & (y[:, None] == 1)).sum(0); fp = (pr & (y[:, None] == 0)).sum(0)
    fn = y.sum() - tp; return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
def f1_at(y, p, t):
    q = p >= t; tp = (q & (y == 1)).sum(); fp = (q & (y == 0)).sum(); fn = y.sum() - tp
    return 2 * tp / max(2 * tp + fp + fn, 1)
def thr(method, y, p):
    if method == "argmax": return GRID[int(np.argmax(f1_curve(y, p)))]
    if method == "prevalence": return float(np.quantile(p, 1 - y.mean()))
    raise ValueError(method)
vlab = pd.read_csv(ROOT / "data/finetune_joint/val/labels.csv"); vpid = vlab.file.astype(str).str.split(".").str[0].values
rng = np.random.default_rng(2); vp = np.unique(vpid)
splits = []
for _ in range(400):
    h = set(rng.permutation(vp)[: len(vp) // 2]); splits.append(np.array([p in h for p in vpid]))
def cv(y, p, method):
    s = []
    for m in splits:
        if y[m].sum() >= 3 and y[~m].sum() >= 3: s.append(f1_at(y[~m], p[~m], thr(method, y[m], p[m])))
    return float(np.mean(s))
methods = ["argmax", "prevalence"]
print("Held-out DR F1 inside validation (400 patient-split halves), mean over seeds; ensemble = 3-seed average")
print(f"{'arm':<15}" + "".join(f"{m:>12}" for m in methods) + "".join(f"{'ens+' + m:>16}" for m in methods))
cvres = {}
for a in ARMS:
    r = {m: np.mean([cv(*load(f"{a}_s{s}", "val"), m) for s in SEEDS]) for m in methods}
    yv = load(f"{a}_s0", "val")[0]; pe = np.mean([load(f"{a}_s{s}", "val")[1] for s in SEEDS], 0)
    for m in methods: r["ens+" + m] = cv(yv, pe, m)
    cvres[a] = r
    print(f"{a:<15}" + "".join(f"{r[m]:>12.4f}" for m in methods) + "".join(f"{r['ens+' + m]:>16.4f}" for m in methods))
allm = methods + ["ens+" + m for m in methods]
pooled = {m: float(np.mean([cvres[a][m] for a in ARMS])) for m in allm}
chosen = max(pooled, key=pooled.get)
print("\npooled held-out validation F1: " + ", ".join(f"{m} {v:.4f}" for m, v in pooled.items()))
print(f"CHOSEN on validation: {chosen}\n")

print("=" * 80 + f"\nTEST, read once with '{chosen}' for every arm (cut-off fitted on the FULL validation set)\n" + "=" * 80)
tlab = pd.read_csv(ROOT / "data/finetune_joint/test/labels.csv"); tpid = tlab.file.astype(str).str.split(".").str[0].values
out = {}
for a in ARMS:
    if chosen.startswith("ens+"):
        m = chosen[4:]; yv = load(f"{a}_s0", "val")[0]; pv = np.mean([load(f"{a}_s{s}", "val")[1] for s in SEEDS], 0)
        yt = load(f"{a}_s0", "test")[0]; pt = np.mean([load(f"{a}_s{s}", "test")[1] for s in SEEDS], 0)
        t = thr(m, yv, pv); q = pt >= t
        f = f1_at(yt, pt, t); out[a] = dict(f1=f, thr=float(t), sens=float(q[yt == 1].mean()), prec=float(yt[q].mean()))
    else:
        fs = []
        for s in SEEDS:
            yv, pv = load(f"{a}_s{s}", "val"); yt, pt = load(f"{a}_s{s}", "test"); fs.append(f1_at(yt, pt, thr(chosen, yv, pv)))
        out[a] = dict(f1=float(np.mean(fs)), seeds=[float(x) for x in fs])
    cur = np.mean([json.load(open(rd(f"{a}_s{s}") / "metrics_test.json"))["per_label"]["diabetic_retinopathy"]["f1"] for s in SEEDS])
    out[a]["trainer_f1"] = float(cur)
    print(f"  {a:<15} DR F1 {out[a]['f1']:.4f}   (trainer's cut-off, per seed mean: {cur:.4f})" +
          (f"   sens {out[a]['sens']:.3f} prec {out[a]['prec']:.3f}" if "sens" in out[a] else ""))
print(f"\n  synthetic − BRSET-only at 0 labels: syn_tgt {out['syn_tgt_b0']['f1'] - out['lebase_b0']['f1']:+.4f}, "
      f"syn_both {out['syn_both_b0']['f1'] - out['lebase_b0']['f1']:+.4f}")
(ROOT / "results/f1_methods.json").write_text(json.dumps(dict(cv=cvres, pooled=pooled, chosen=chosen, test=out), indent=2))
print("wrote results/f1_methods.json")
