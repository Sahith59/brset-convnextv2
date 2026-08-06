"""Bootstrap 95% CIs for the strong-baseline runs, straight from saved test
predictions (no model reload needed), and compare against the prior ensemble.

Resamples the test set with replacement 2,000 times, recomputing each metric
at the already-fixed (validation-tuned) threshold, and reports the 2.5th /
97.5th percentiles. Same procedure as scripts/16, so the numbers are directly
comparable to the previously reported ensemble CIs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

RESULTS = Path("/home/users/sthummala2/brset-convnextv2/results")
LABELS = ["diabetic_retinopathy", "macular_edema"]
N_BOOT = 2000
SEED = 0


def metrics_at(y_true, y_prob, thr):
    pred = (y_prob >= thr).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }


def bootstrap(y_true, y_prob, thr, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc = {k: [] for k in ("auc", "f1", "precision", "recall")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() < 2 or y_true[idx].sum() == len(idx):
            continue
        m = metrics_at(y_true[idx], y_prob[idx], thr)
        for k, v in m.items():
            if not np.isnan(v):
                acc[k].append(v)
    point = metrics_at(y_true, y_prob, thr)
    out = {}
    for k, vals in acc.items():
        vals = np.array(vals)
        out[k] = {
            "point": float(point[k]),
            "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)),
        }
    return out


def report(task):
    d = RESULTS / task
    npz = np.load(d / "test_predictions.npz")
    y_true, y_prob, thr = npz["y_true"], npz["y_prob"], npz["thresholds"]
    print(f"\n{'='*78}\n{task}\n{'='*78}")
    out = {}
    for i, lab in enumerate(LABELS):
        r = bootstrap(y_true[:, i].astype(int), y_prob[:, i], float(thr[i]))
        out[lab] = r
        n_pos = int(y_true[:, i].sum())
        print(f"  {lab}  (threshold {thr[i]:.2f}, {n_pos} positives of {len(y_true)})")
        for k in ("auc", "f1", "precision", "recall"):
            v = r[k]
            print(f"    {k:10s} {v['point']:.4f}   95% CI [{v['lo']:.4f}, {v['hi']:.4f}]")
    with open(d / "bootstrap_ci.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> wrote {d/'bootstrap_ci.json'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=[
        "convnextv2_large_BRSET_strong_baseline",
        "convnextv2_large_BRSET_strong_baseline_focal",
    ])
    args = ap.parse_args()

    results = {t: report(t) for t in args.tasks}

    # ---- side-by-side against the previously reported ensemble ----
    prior_path = RESULTS / "convnextv2_large_BRSET_ensemble/ensemble_bootstrap_ci.json"
    prior = json.load(open(prior_path)) if prior_path.exists() else None

    print(f"\n{'='*78}\nCOMPARISON (point estimate, 95% CI)\n{'='*78}")
    hdr = f"{'':34s}" + "".join(f"{n:>26s}" for n in ["DR AUC", "DR F1"])
    print(hdr)
    rows = []
    if prior:
        rows.append(("prior ensemble (2 models)",
                     prior["diabetic_retinopathy"]["auc"],
                     prior["diabetic_retinopathy"]["f1"]))
    for t in args.tasks:
        short = "ASL (asymmetric)" if t.endswith("baseline") else "plain focal"
        r = results[t]["diabetic_retinopathy"]
        rows.append((f"{short} (1 model)",
                     [r["auc"]["point"], r["auc"]["lo"], r["auc"]["hi"]],
                     [r["f1"]["point"], r["f1"]["lo"], r["f1"]["hi"]]))
    for name, auc, f1 in rows:
        print(f"{name:34s}" +
              f"{auc[0]:.4f} [{auc[1]:.3f},{auc[2]:.3f}]".rjust(26) +
              f"{f1[0]:.4f} [{f1[1]:.3f},{f1[2]:.3f}]".rjust(26))

    print()
    hdr = f"{'':34s}" + "".join(f"{n:>26s}" for n in ["ME AUC", "ME F1"])
    print(hdr)
    rows = []
    if prior:
        rows.append(("prior ensemble (2 models)",
                     prior["macular_edema"]["auc"], prior["macular_edema"]["f1"]))
    for t in args.tasks:
        short = "ASL (asymmetric)" if t.endswith("baseline") else "plain focal"
        r = results[t]["macular_edema"]
        rows.append((f"{short} (1 model)",
                     [r["auc"]["point"], r["auc"]["lo"], r["auc"]["hi"]],
                     [r["f1"]["point"], r["f1"]["lo"], r["f1"]["hi"]]))
    for name, auc, f1 in rows:
        print(f"{name:34s}" +
              f"{auc[0]:.4f} [{auc[1]:.3f},{auc[2]:.3f}]".rjust(26) +
              f"{f1[0]:.4f} [{f1[1]:.3f},{f1[2]:.3f}]".rjust(26))


if __name__ == "__main__":
    main()
