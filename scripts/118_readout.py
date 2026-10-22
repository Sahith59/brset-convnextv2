"""Read-out for any set of joint-training arms: validation first, then test once.
Arms are templates with {s} for the seed. F1 is reported at the trainer's cut-off
and with the validation-chosen method from 117 (seed ensemble + bootstrap-median)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path("/home/users/sthummala2/brset-convnextv2"); CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
GRID = np.arange(0.05, 0.95, 0.01); S = (0, 1, 2)
ARMS = {"BRSET only (0 labels)": "lebase_b0_s{s}",
        "+synthetic gen_tgt 40k, p=0.25": "syn_tgt_b0_s{s}_p0.25",
        "+synthetic gen_tgt 40k, p=0.50": "syn_tgt_b0_s{s}",
        "+synthetic gen_tgt 40k, p=0.75": "syn_tgt_b0_s{s}_p0.75",
        "+synthetic gen_both 40k, p=0.50": "syn_both_b0_s{s}",
        "+synthetic gen_both 60k, p=0.50": "syn_bothld60_b0_s{s}",
        "250 real labels": "lebase_b250_s{s}",
        "250 real +gen_tgt 40k": "syn_tgt_b250_s{s}",
        "250 real +gen_both 60k": "syn_bothld60_b250_s{s}",
        "500 real labels": "lebase_b500_s{s}"}
rd = lambda t: (CLS if t.startswith("syn_") else ROOT / "results") / t
def ld(t, sp):
    d = np.load(rd(t) / f"{sp}_predictions.npz"); return d["y_true"][:, 0].astype(bool), d["y_prob"][:, 0]
def f1c(y, p):
    P = p[None] >= GRID[:, None]; tp = (P & y).sum(1); fp = (P & ~y).sum(1); fn = y.sum() - tp
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)
def f1(y, q): tp = (q & y).sum(); return 2 * tp / max(2 * tp + (q & ~y).sum() + (~q & y).sum(), 1)
def bmed(y, p, seed=0):
    r = np.random.default_rng(seed); ts = []
    for _ in range(200):
        i = r.integers(0, len(y), len(y))
        if y[i].sum() >= 2: ts.append(GRID[int(np.argmax(f1c(y[i], p[i])))])
    return float(np.median(ts))
R = {}
for name, tpl in ARMS.items():
    V = [ld(tpl.format(s=s), "val") for s in S]; T = [ld(tpl.format(s=s), "test") for s in S]
    yv, yt = V[0][0], T[0][0]
    thr = [float(np.load(rd(tpl.format(s=s)) / "test_predictions.npz")["thresholds"][0]) for s in S]
    pv, pt = np.mean([v[1] for v in V], 0), np.mean([t[1] for t in T], 0); te = bmed(yv, pv)
    R[name] = dict(val_auc=[roc_auc_score(yv, v[1]) for v in V], test_auc=[roc_auc_score(yt, t[1]) for t in T],
                   test_f1=[f1(yt, t[1] >= c) for t, c in zip(T, thr)], ens_f1=f1(yt, pt >= te), ens_auc=roc_auc_score(yt, pt),
                   sens=np.mean([((t[1] >= c) & yt).sum() / yt.sum() for t, c in zip(T, thr)]),
                   prec=np.mean([((t[1] >= c) & yt).sum() / max((t[1] >= c).sum(), 1) for t, c in zip(T, thr)]),
                   q=[t[1] >= c for t, c in zip(T, thr)], qe=pt >= te, P=[t[1] for t in T])
print(f"{'arm':<34}{'VAL AUROC':>10} | {'TEST AUROC':>11}{'F1':>7}{'sens':>6}{'prec':>6} |{'ens AUROC':>10}{'ens F1':>8}")
for n, r in R.items():
    print(f"{n:<34}{np.mean(r['val_auc']):>10.4f} | {np.mean(r['test_auc']):>11.4f}{np.mean(r['test_f1']):>7.4f}{r['sens']:>6.3f}{r['prec']:>6.3f} |{r['ens_auc']:>10.4f}{r['ens_f1']:>8.4f}")
tl = pd.read_csv(ROOT / "data/finetune_joint/test/labels.csv"); pid = tl.file.astype(str).str.split(".").str[0].values
pats = np.unique(pid); ix = {p: np.where(pid == p)[0] for p in pats}; yt = ld("lebase_b0_s0", "test")[0]
pairs = [("+synthetic gen_tgt 40k, p=0.25", "BRSET only (0 labels)"), ("+synthetic gen_tgt 40k, p=0.50", "BRSET only (0 labels)"),
         ("+synthetic gen_tgt 40k, p=0.75", "BRSET only (0 labels)"), ("+synthetic gen_both 60k, p=0.50", "BRSET only (0 labels)"),
         ("250 real +gen_both 60k", "250 real labels"), ("250 real +gen_both 60k", "500 real labels")]
print("\nPaired vs reference: mean of 3 seeds; patient-cluster bootstrap 95% CI (2,000 resamples)")
rng = np.random.default_rng(1); out = {}
for a, b in pairs:
    dA, dF, dE = [], [], []
    for _ in range(2000):
        s_ = np.concatenate([ix[p] for p in rng.choice(pats, len(pats), replace=True)]); yy = yt[s_]
        dA.append(np.mean([roc_auc_score(yy, p[s_]) for p in R[a]["P"]]) - np.mean([roc_auc_score(yy, p[s_]) for p in R[b]["P"]]))
        dF.append(np.mean([f1(yy, q[s_]) for q in R[a]["q"]]) - np.mean([f1(yy, q[s_]) for q in R[b]["q"]]))
        dE.append(f1(yy, R[a]["qe"][s_]) - f1(yy, R[b]["qe"][s_]))
    ci = lambda d: (np.mean(d), *np.percentile(d, [2.5, 97.5]))
    up = sum(x > y for x, y in zip(R[a]["test_auc"], R[b]["test_auc"]))
    print(f"  {a}  vs  {b}")
    for lab, d in (("AUROC", dA), ("F1", dF), ("ens F1", dE)):
        m, lo, hi = ci(d); print(f"      {lab:<7} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]" + (f"   seeds up {up}/3" if lab == "AUROC" else ""))
    out[f"{a} | {b}"] = {k: dict(zip(("mean", "lo", "hi"), map(float, ci(d)))) for k, d in (("auroc", dA), ("f1", dF), ("ens_f1", dE))}
for r in R.values():
    for k in ("q", "qe", "P"): r.pop(k)
(ROOT / "results/readout_2026-10-13_1610.json").write_text(json.dumps(dict(arms=R, pairs=out), indent=2, default=float))
print("\nwrote results/readout_2026-10-13_1610.json")
