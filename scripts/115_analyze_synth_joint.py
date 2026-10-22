"""Read-out of joint training with synthetic mBRSET (scripts/114_synth_joint.slurm).

Order fixed in advance (research/codex DECISIONS D068-D069):
 1. choose the generator on mBRSET VALIDATION DR AUROC (val_predictions.npz);
 2. report every arm on TEST once, paired by seed against the matching lebase run;
 3. patient-cluster bootstrap on test for (synthetic + B) minus (real B) and for
    (synthetic + 250) minus (real 500). Patients = mBRSET filename prefix.
Aggregates only are written.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
SEEDS = (0, 1, 2)


def run_dir(task):
    return CLS / task if task.startswith("syn_") else ROOT / "results" / task


def pred(task, split):
    d = np.load(run_dir(task) / f"{split}_predictions.npz")
    return d["y_true"], d["y_prob"]


def auc(task, split, col=0):
    y, p = pred(task, split)
    return roc_auc_score(y[:, col], p[:, col])


out = {}
# ---------------------------------------------------------------- 1. validation
print("=" * 78 + "\n1. mBRSET VALIDATION (725 images) — DR AUROC, used to choose the generator\n" + "=" * 78)
val = {}
for B in (0, 250):
    base = f"lebase_b{B}"
    vb = [auc(f"{base}_s{s}", "val") for s in SEEDS]
    val[base] = vb
    print(f"  {base:<16} {np.mean(vb):.4f}   seeds {', '.join(f'{x:.4f}' for x in vb)}")
    for G in ("both", "tgt"):
        t = f"syn_{G}_b{B}"
        v = [auc(f"{t}_s{s}", "val") for s in SEEDS]
        val[t] = v
        d = [a - b for a, b in zip(v, vb)]
        print(f"  {t:<16} {np.mean(v):.4f}   Δ vs {base} {np.mean(d):+.4f}  ({sum(x > 0 for x in d)}/3 seeds up)")
vb500 = [auc(f"lebase_b500_s{s}", "val") for s in SEEDS]
val["lebase_b500"] = vb500
print(f"  {'lebase_b500':<16} {np.mean(vb500):.4f}")
score = {G: np.mean(val[f"syn_{G}_b0"] + val[f"syn_{G}_b250"]) for G in ("both", "tgt")}
chosen = max(score, key=score.get)
print(f"\n  generator choice (mean validation DR AUROC over both budgets): "
      f"gen_both {score['both']:.4f} vs gen_tgt {score['tgt']:.4f} -> gen_{chosen}")
out["validation_dr_auroc"] = {k: [float(x) for x in v] for k, v in val.items()}
out["chosen_generator"] = f"gen_{chosen}"

# ---------------------------------------------------------------- 2. test
print("\n" + "=" * 78 + "\n2. mBRSET TEST (732 images, 193 patients) — read once\n" + "=" * 78)
test = {}
def met(task):
    return json.load(open(run_dir(task) / "metrics_test.json"))["per_label"]
for name in ["lebase_b0", "lebase_b250", "lebase_b500"] + [f"syn_{G}_b{B}" for G in ("both", "tgt") for B in (0, 250)]:
    m = [met(f"{name}_s{s}") for s in SEEDS]
    test[name] = dict(dr_auc=[x["diabetic_retinopathy"]["auc"] for x in m],
                      dr_f1=[x["diabetic_retinopathy"]["f1"] for x in m],
                      me_auc=[x["macular_edema"]["auc"] for x in m],
                      me_f1=[x["macular_edema"]["f1"] for x in m])
hdr = f"  {'arm':<16} {'DR AUROC':>9} {'DR F1':>7} {'ME AUROC':>9} {'ME F1':>7}"
print(hdr)
for k, v in test.items():
    print(f"  {k:<16} {np.mean(v['dr_auc']):>9.4f} {np.mean(v['dr_f1']):>7.4f} {np.mean(v['me_auc']):>9.4f} {np.mean(v['me_f1']):>7.4f}")
print("\n  paired by seed (DR AUROC):")
pairs = [("syn_both_b0", "lebase_b0"), ("syn_tgt_b0", "lebase_b0"),
         ("syn_both_b250", "lebase_b250"), ("syn_tgt_b250", "lebase_b250"),
         ("syn_both_b250", "lebase_b500"), ("syn_tgt_b250", "lebase_b500")]
for a, b in pairs:
    d = [x - y for x, y in zip(test[a]["dr_auc"], test[b]["dr_auc"])]
    f = [x - y for x, y in zip(test[a]["dr_f1"], test[b]["dr_f1"])]
    print(f"    {a:<14} − {b:<12}: DR AUROC {np.mean(d):+.4f} ({sum(x > 0 for x in d)}/3 up)   DR F1 {np.mean(f):+.4f} ({sum(x > 0 for x in f)}/3 up)")
out["test"] = {k: {kk: [float(x) for x in vv] for kk, vv in v.items()} for k, v in test.items()}

# ---------------------------------------------------------------- 3. bootstrap
print("\n" + "=" * 78 + "\n3. patient-cluster bootstrap on test, 2,000 resamples (seed-averaged DR AUROC)\n" + "=" * 78)
lab = pd.read_csv(ROOT / "data/finetune_joint/test/labels.csv")
pid = lab.file.astype(str).str.split(".").str[0].values
y0, _ = pred("lebase_b0_s0", "test")
assert len(pid) == len(y0) and (lab.diabetic_retinopathy.values == y0[:, 0]).all(), "test order mismatch"
pats = np.unique(pid)
idx = {p: np.where(pid == p)[0] for p in pats}
P = {}
for name in test:
    ps = []
    for s in SEEDS:
        y, p = pred(f"{name}_s{s}", "test")
        assert (y[:, 0] == y0[:, 0]).all(), f"{name} order mismatch"
        ps.append(p[:, 0])
    P[name] = ps
rng = np.random.default_rng(0)
boot = {f"{a}-{b}": [] for a, b in pairs}
for _ in range(2000):
    samp = np.concatenate([idx[p] for p in rng.choice(pats, size=len(pats), replace=True)])
    yy = y0[samp, 0]
    if yy.min() == yy.max():
        continue
    m = {k: np.mean([roc_auc_score(yy, ps[samp]) for ps in v]) for k, v in P.items()}
    for a, b in pairs:
        boot[f"{a}-{b}"].append(m[a] - m[b])
out["bootstrap"] = {}
for k, v in boot.items():
    v = np.array(v); lo, hi = np.percentile(v, [2.5, 97.5])
    out["bootstrap"][k] = dict(mean=float(v.mean()), ci95=[float(lo), float(hi)], p_le_0=float((v <= 0).mean()))
    print(f"  {k:<28} {v.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]   P(Δ ≤ 0) = {(v <= 0).mean():.3f}")
print("\n  The interval reflects patient sampling only, not training randomness (3 seeds).")
(ROOT / "results/synth_joint_readout.json").write_text(json.dumps(out, indent=2))
print("\nwrote results/synth_joint_readout.json")
