"""Cut-off methods for DR F1, chosen on VALIDATION, then applied to test once.

Patient-split cross-validation inside the 725-image validation set: tune on half
the validation patients, score F1 on the other half (400 splits). Methods:
  argmax       F1-optimal cut-off on the tuning half
  boot_median  median of F1-optimal cut-offs over bootstrap resamples (the trainer's method)
  prevalence   flag the same fraction as the tuning half's positive rate
and the same three on the seed-ensemble (mean probability of the 3 seeds).
The single method with the best held-out F1 averaged over ALL arms is then
applied to test for every arm alike (cut-off tuned on the full validation set).
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/users/sthummala2/brset-convnextv2"); CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
SEEDS = (0, 1, 2); GRID = np.arange(0.05, 0.95, 0.01)
ARMS = ["lebase_b0", "syn_tgt_b0", "syn_both_b0", "lebase_b250", "syn_tgt_b250", "syn_both_b250", "lebase_b500"]
rd = lambda t: CLS / t if t.startswith("syn_") else ROOT / "results" / t
def load(t, split):
    d = np.load(rd(t) / f"{split}_predictions.npz"); return d["y_true"][:, 0].astype(bool), d["y_prob"][:, 0]

def f1_curve(y, p):
    P = p[None, :] >= GRID[:, None]
    tp = (P & y).sum(1); fp = (P & ~y).sum(1); fn = y.sum() - tp
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)
def f1_at(y, p, t):
    q = p >= t; tp = (q & y).sum(); fp = (q & ~y).sum(); fn = y.sum() - tp
    return 2 * tp / max(2 * tp + fp + fn, 1)
def thr(y, p, method, rng):
    if method == "argmax":
        return GRID[int(np.argmax(f1_curve(y, p)))]
    if method == "boot_median":
        ts = []
        for _ in range(100):
            i = rng.integers(0, len(y), len(y))
            if y[i].sum() >= 2: ts.append(GRID[int(np.argmax(f1_curve(y[i], p[i])))])
        return float(np.median(ts))
    if method == "prevalence":
        return float(np.quantile(p, 1 - y.mean()))

METHODS = ["argmax", "boot_median", "prevalence"]
vlab = pd.read_csv(ROOT / "data/finetune_joint/val/labels.csv"); vpid = vlab.file.astype(str).str.split(".").str[0].values
yv0, _ = load("lebase_b0_s0", "val"); assert (vlab.diabetic_retinopathy.values.astype(bool) == yv0).all()
vpats = np.unique(vpid); rng = np.random.default_rng(2); splits = []
for _ in range(400):
    h = set(rng.permutation(vpats)[: len(vpats) // 2]); splits.append(np.array([p in h for p in vpid]))
def cv(y, p, method):
    r2 = np.random.default_rng(3); sc = []
    for m in splits:
        if y[m].sum() < 3 or y[~m].sum() < 3: continue
        sc.append(f1_at(y[~m], p[~m], thr(y[m], p[m], method, r2)))
    return float(np.mean(sc))

print("=" * 92 + "\nA. Held-out DR F1 inside validation (patient-split CV, 400 splits)\n" + "=" * 92)
print(f"{'arm':<15}" + "".join(f"{m:>13}" for m in METHODS) + "".join(f"{'ens+' + m:>17}" for m in METHODS))
res = {}
for a in ARMS:
    V = [load(f"{a}_s{s}", "val") for s in SEEDS]; y = V[0][0]; pe = np.mean([v[1] for v in V], 0)
    r = {m: float(np.mean([cv(y, p, m) for _, p in V])) for m in METHODS}
    r.update({"ens+" + m: cv(y, pe, m) for m in METHODS}); res[a] = r
    print(f"{a:<15}" + "".join(f"{r[m]:>13.4f}" for m in METHODS) + "".join(f"{r['ens+' + m]:>17.4f}" for m in METHODS), flush=True)
allm = METHODS + ["ens+" + m for m in METHODS]
pooled = {m: float(np.mean([res[a][m] for a in ARMS])) for m in allm}
chosen = max(pooled, key=pooled.get)
print("\n  averaged over arms: " + ", ".join(f"{m} {v:.4f}" for m, v in pooled.items()))
print(f"  CHOSEN ON VALIDATION: {chosen}")

print("\n" + "=" * 92 + f"\nB. TEST DR F1 with the validation-chosen method ({chosen}), every arm alike — read once\n" + "=" * 92)
tlab = pd.read_csv(ROOT / "data/finetune_joint/test/labels.csv"); tpid = tlab.file.astype(str).str.split(".").str[0].values
ens = chosen.startswith("ens+"); base_m = chosen.replace("ens+", "")
test = {}
for a in ARMS:
    V = [load(f"{a}_s{s}", "val") for s in SEEDS]; T = [load(f"{a}_s{s}", "test") for s in SEEDS]
    yv, yt = V[0][0], T[0][0]
    if ens:
        pv, pt = np.mean([v[1] for v in V], 0), np.mean([t[1] for t in T], 0)
        t = thr(yv, pv, base_m, np.random.default_rng(0))
        bins = [(pt >= t)]; f = [f1_at(yt, pt, t)]; au = roc_auc_score(yt, pt); tt = [t]
    else:
        tt = [thr(v[0], v[1], base_m, np.random.default_rng(s)) for s, v in zip(SEEDS, V)]
        f = [f1_at(t_[0], t_[1], c) for t_, c in zip(T, tt)]; bins = [t_[1] >= c for t_, c in zip(T, tt)]
        au = float(np.mean([roc_auc_score(t_[0], t_[1]) for t_ in T]))
    rec = float(np.mean([(b & yt).sum() / yt.sum() for b in bins])); prec = float(np.mean([(b & yt).sum() / max(b.sum(), 1) for b in bins]))
    test[a] = dict(f1=f, f1_mean=float(np.mean(f)), auroc=float(au), thr=[float(x) for x in tt], recall=rec, precision=prec, bins=bins)
    print(f"  {a:<15} DR F1 {np.mean(f):.4f}  (seeds {', '.join(f'{x:.3f}' for x in f)})  AUROC {au:.4f}  sens {rec:.3f}  prec {prec:.3f}", flush=True)

print("\n  paired comparisons + patient-cluster bootstrap of the F1 difference (2,000 resamples):")
pats = np.unique(tpid); ix = {p: np.where(tpid == p)[0] for p in pats}; yt = load("lebase_b0_s0", "test")[0]
rngb = np.random.default_rng(1); out_pairs = {}
for a, b in [("syn_tgt_b0", "lebase_b0"), ("syn_both_b0", "lebase_b0"), ("syn_tgt_b250", "lebase_b250"), ("syn_both_b250", "lebase_b250"), ("syn_tgt_b250", "lebase_b500")]:
    d = []
    for _ in range(2000):
        s_ = np.concatenate([ix[p] for p in rngb.choice(pats, len(pats), replace=True)]); yy = yt[s_]
        fa = np.mean([2 * (q[s_] & yy).sum() / max(2 * (q[s_] & yy).sum() + (q[s_] & ~yy).sum() + (~q[s_] & yy).sum(), 1) for q in test[a]["bins"]])
        fb = np.mean([2 * (q[s_] & yy).sum() / max(2 * (q[s_] & yy).sum() + (q[s_] & ~yy).sum() + (~q[s_] & yy).sum(), 1) for q in test[b]["bins"]])
        d.append(fa - fb)
    d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5]); out_pairs[f"{a}-{b}"] = dict(mean=float(d.mean()), ci95=[float(lo), float(hi)], p_ge_0=float((d >= 0).mean()))
    print(f"    {a:<14} − {b:<12} F1 {test[a]['f1_mean'] - test[b]['f1_mean']:+.4f}   bootstrap {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  P(Δ≥0)={(d >= 0).mean():.3f}", flush=True)
for a in test: test[a].pop("bins")
(ROOT / "results/f1_thresholds.json").write_text(json.dumps(dict(cv=res, pooled=pooled, chosen=chosen, test=test, pairs=out_pairs), indent=2))
print("\nwrote results/f1_thresholds.json")
