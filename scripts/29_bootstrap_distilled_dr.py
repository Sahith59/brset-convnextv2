"""
Bootstrap 95% CIs for the Phase-2 distilled model's test-set results, so the
DR-improvement claim is validated the same rigorous way as every other
number in this project (2,000 resamples, same convention as script 16).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

RESULTS_DIR = Path("/home/users/sthummala2/brset-convnextv2/results")
LABEL_COLS = ["diabetic_retinopathy", "macular_edema"]
N_BOOTSTRAP = 2000
SEED = 42


def bootstrap_metric(y_true, y_prob, threshold, rng):
    n = len(y_true)
    aucs, f1s, precs, recs = [], [], [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        pred = (yp >= threshold).astype(int)
        if yt.sum() == 0 or yt.sum() == n:
            continue
        try:
            aucs.append(roc_auc_score(yt, yp))
        except ValueError:
            pass
        f1s.append(f1_score(yt, pred, zero_division=0))
        precs.append(precision_score(yt, pred, zero_division=0))
        recs.append(recall_score(yt, pred, zero_division=0))
    def ci(vals):
        vals = np.array(vals)
        return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    return {"auc": ci(aucs), "f1": ci(f1s), "precision": ci(precs), "recall": ci(recs)}


def main():
    d = np.load(RESULTS_DIR / "convnextv2_large_mBRSET_multilabel_512_distilled_dr/test_predictions.npz")
    y_true, y_prob, thresholds = d["y_true"], d["y_prob"], d["thresholds"]

    rng = np.random.default_rng(SEED)
    results = {}
    for i, col in enumerate(LABEL_COLS):
        ci = bootstrap_metric(y_true[:, i], y_prob[:, i], float(thresholds[i]), rng)
        print(f"{col} (threshold={thresholds[i]:.2f}):")
        for metric, (mean, lo, hi) in ci.items():
            print(f"  {metric:<10} {mean:.4f}  [{lo:.4f} - {hi:.4f}]")
        results[col] = ci

    with open(RESULTS_DIR / "convnextv2_large_mBRSET_multilabel_512_distilled_dr/bootstrap_ci.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote bootstrap_ci.json")


if __name__ == "__main__":
    main()
