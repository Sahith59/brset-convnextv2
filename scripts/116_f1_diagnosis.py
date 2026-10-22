"""Why does synthetic joint training raise DR AUROC but not DR F1?

Diagnostics only, then a thresholding comparison decided on VALIDATION by
patient-split cross-validation (tune the cut-off on one half of the validation
patients, score F1 on the other half, 400 repeats). Test is read once at the end
with the method validation picks, for every arm with the SAME method.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

ROOT = Path("/home/users/sthummala2/brset-convnextv2"); CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
SEEDS = (0, 1, 2); GRID = np.arange(0.05, 0.95, 0.01)
ARMS = ["lebase_b0", "syn_tgt_b0", "syn_both_b0", "lebase_b250", "syn_tgt_b250", "syn_both_b250", "lebase_b500"]
rd = lambda t: CLS / t if t.startswith("syn_") else ROOT / "results" / t
def load(t, split):
    d = np.load(rd(t) / f"{split}_predictions.npz"); return d["y_true"][:, 0].astype(int), d["y_prob"][:, 0]
def best_thr(y, p):
    f = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in GRID]; return GRID[int(np.argmax(f))], max(f)
def boot_thr(y, p, rng, n=200):
    ts = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() >= 2: ts.append(best_thr(y[i], p[i])[0])
    return float(np.median(ts))
def prev_thr(y, p):            # flag the same fraction as the tuning set's positive rate
    return float(np.quantile(p, 1 - y.mean()))
f1 = lambda y, p, t: f1_score(y, (p >= t).astype(int), zero_division=0)

vlab = pd.read_csv(ROOT / "data/finetune_joint/val/labels.csv"); vpid = vlab.file.astype(str).str.split(".").str[0].values
yv0, _ = load("lebase_b0_s0", "val"); yt0, _ = load("lebase_b0_s0", "test")
assert (vlab.diabetic_retinopathy.values == yv0).all()
print(f"validation {len(yv0)} images ({yv0.sum()} DR+), test {len(yt0)} images ({yt0.sum()} DR+)\n")

# ---------------------------------------------------------------- 1. where the F1 goes
print("=" * 96 + "\n1. Per arm (mean of 3 seeds): threshold, F1 on validation, F1 on test, and score distributions\n" + "=" * 96)
print(f"{'arm':<15}{'thr':>6}{'valF1':>7}{'testF1':>8}{'test oracle F1*':>16}{'test sens':>10}{'test prec':>10}"
      f"{'val neg':>9}{'val pos':>9}{'test neg':>9}{'test pos':>9}")
diag = {}
for a in ARMS:
    r = []
    for s in SEEDS:
        yv, pv = load(f"{a}_s{s}", "val"); yt, pt = load(f"{a}_s{s}", "test")
        thr = float(np.load(rd(f"{a}_s{s}") / "test_predictions.npz")["thresholds"][0])
        r.append(dict(thr=thr, vf=f1(yv, pv, thr), tf=f1(yt, pt, thr), of=best_thr(yt, pt)[1],
                      se=recall_score(yt, (pt >= thr).astype(int)), pr=precision_score(yt, (pt >= thr).astype(int), zero_division=0),
                      vn=np.median(pv[yv == 0]), vp=np.median(pv[yv == 1]), tn=np.median(pt[yt == 0]), tp=np.median(pt[yt == 1])))
    m = {k: float(np.mean([x[k] for x in r])) for k in r[0]}; diag[a] = m
    print(f"{a:<15}{m['thr']:>6.2f}{m['vf']:>7.3f}{m['tf']:>8.3f}{m['of']:>16.3f}{m['se']:>10.3f}{m['pr']:>10.3f}"
          f"{m['vn']:>9.3f}{m['vp']:>9.3f}{m['tn']:>9.3f}{m['tp']:>9.3f}")
print("  * oracle = best cut-off chosen ON test: a diagnostic of headroom, never a reportable result.")
print("  neg/pos = median predicted DR probability for healthy / diseased images.")

# ---------------------------------------------------------------- 2. is the F1 drop real?
print("\n" + "=" * 96 + "\n2. Patient-cluster bootstrap of the test DR F1 difference at the trainer's own cut-offs (2,000 resamples)\n" + "=" * 96)
tlab = pd.read_csv(ROOT / "data/finetune_joint/test/labels.csv"); tpid = tlab.file.astype(str).str.split(".").str[0].values
pats = np.unique(tpid); ix = {p: np.where(tpid == p)[0] for p in pats}
def preds(a): 
    out = []
    for s in SEEDS:
        y, p = load(f"{a}_s{s}", "test"); t = float(np.load(rd(f"{a}_s{s}") / "test_predictions.npz")["thresholds"][0])
        out.append((p >= t).astype(int))
    return out
rng = np.random.default_rng(1); B = {a: preds(a) for a in ("lebase_b0", "syn_tgt_b0", "syn_both_b0")}
for a in ("syn_tgt_b0", "syn_both_b0"):
    d = []
    for _ in range(2000):
        smp = np.concatenate([ix[p] for p in rng.choice(pats, len(pats), replace=True)]); yy = yt0[smp]
        d.append(np.mean([f1_score(yy, q[smp]) for q in B[a]]) - np.mean([f1_score(yy, q[smp]) for q in B["lebase_b0"]]))
    d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {a} − lebase_b0: DR F1 {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(Δ ≥ 0) = {(d >= 0).mean():.3f}")

# ---------------------------------------------------------------- 3. thresholding methods, judged on validation only
print("\n" + "=" * 96 + "\n3. Cut-off methods judged by patient-split cross-validation INSIDE validation (400 splits)\n" + "=" * 96)
vpats = np.unique(vpid); rng = np.random.default_rng(2)
splits = []
for _ in range(400):
    perm = rng.permutation(vpats); h = set(perm[:len(perm) // 2])
    m = np.array([p in h for p in vpid]); splits.append(m)
def cv_score(y, p, method):
    sc = []
    r2 = np.random.default_rng(3)
    for m in splits:
        a, b = m, ~m
        if y[a].sum() < 3 or y[b].sum() < 3: continue
        if method == "argmax": t = best_thr(y[a], p[a])[0]
        elif method == "boot_median": t = boot_thr(y[a], p[a], r2, n=40)
        elif method == "prevalence": t = prev_thr(y[a], p[a])
        sc.append(f1(y[b], p[b], t))
    return float(np.mean(sc))
methods = ["argmax", "boot_median", "prevalence"]
res = {}
print(f"{'arm':<15}" + "".join(f"{m:>14}" for m in methods) + f"{'seed-ensemble + best':>24}")
for a in ARMS:
    row = {m: np.mean([cv_score(*load(f"{a}_s{s}", "val"), m) for s in SEEDS]) for m in methods}
    yv, _ = load(f"{a}_s0", "val"); pe = np.mean([load(f"{a}_s{s}", "val")[1] for s in SEEDS], axis=0)
    best_m = max(methods, key=lambda m: row[m]); row["ensemble"] = cv_score(yv, pe, best_m)
    res[a] = row
    print(f"{a:<15}" + "".join(f"{row[m]:>14.4f}" for m in methods) + f"{row['ensemble']:>24.4f}")
pooled = {m: np.mean([res[a][m] for a in ARMS]) for m in methods}
chosen = max(pooled, key=pooled.get)
print(f"\n  method with the best held-out validation F1, averaged over all arms: {chosen}  "
      + ", ".join(f"{m} {v:.4f}" for m, v in pooled.items()))
(ROOT / "results/f1_diagnosis.json").write_text(json.dumps(dict(diag=diag, cv=res, chosen_method=chosen), indent=2))
print("wrote results/f1_diagnosis.json")
