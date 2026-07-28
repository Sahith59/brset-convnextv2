"""
Bootstrap confidence intervals for test-set AUC/F1/precision/recall, to
report each metric as a range rather than a single fragile point estimate -
especially important for macular_edema, whose test set has only ~60 positive
examples. Resamples test predictions (with replacement) 2000 times, applying
the already-tuned (on val) threshold each time, and reports the 2.5th/97.5th
percentile as the 95% CI. No retraining - reuses the trained checkpoint's
actual TTA test predictions.
"""
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "train_mod", str(Path(__file__).parent / "07_train_convnextv2_multilabel.py"))
train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_mod)

BRSETMultiLabel = train_mod.BRSETMultiLabel
build_transforms = train_mod.build_transforms
run_inference = train_mod.run_inference
tune_thresholds = train_mod.tune_thresholds
LABEL_COLS = train_mod.LABEL_COLS

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
            continue  # skip resamples with no variation (AUC undefined)
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
    ckpt_path = sys.argv[1]
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "dataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]

    model = timm.create_model(args["model"], pretrained=False, num_classes=args["nb_classes"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    class A: pass
    a = A()
    a.resize_size, a.input_size = args["resize_size"], args["input_size"]
    _, eval_tf = build_transforms(a)

    data_path = Path(args["data_path"])
    ds_val = BRSETMultiLabel(data_path / "val", eval_tf)
    ds_test = BRSETMultiLabel(data_path / "test", eval_tf)
    loader_val = DataLoader(ds_val, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    loader_test = DataLoader(ds_test, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    print(f"[{dataset_name}] running TTA inference (val for thresholds, test for CIs)...")
    y_val_true, y_val_prob = run_inference(model, loader_val, device, tta=True)
    thresholds = tune_thresholds(y_val_true, y_val_prob)
    y_test_true, y_test_prob = run_inference(model, loader_test, device, tta=True)

    np.savez(Path(ckpt_path).parent / "test_predictions.npz",
             y_true=y_test_true, y_prob=y_test_prob, thresholds=thresholds)

    rng = np.random.default_rng(SEED)
    print(f"\n{dataset_name}: bootstrap 95% CIs (n_test={len(y_test_true)}, {N_BOOTSTRAP} resamples)")
    results = {}
    for i, col in enumerate(LABEL_COLS):
        res = bootstrap_metric(y_test_true[:, i], y_test_prob[:, i], thresholds[i], rng)
        results[col] = res
        print(f"\n{col} (threshold={thresholds[i]:.2f}):")
        for metric, (mean, lo, hi) in res.items():
            print(f"  {metric:<10} {mean:.4f}  [95% CI: {lo:.4f} - {hi:.4f}]")

    import json
    out_path = Path(ckpt_path).parent / "bootstrap_ci.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
