"""
Ensemble two independently-trained checkpoints (the original gamma=2.0 model
and the regularized variant) by averaging their predicted probabilities, then
evaluate the ensemble the same rigorous way as every single model so far:
threshold tuned on val, TTA on test, bootstrap 95% CIs. No new training -
this is free, using checkpoints we already have.
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
compute_per_label_metrics = train_mod.compute_per_label_metrics
LABEL_COLS = train_mod.LABEL_COLS

N_BOOTSTRAP = 2000
SEED = 42


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    model = timm.create_model(args["model"], pretrained=False, num_classes=args["nb_classes"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, args


def get_loaders(args):
    class A: pass
    a = A()
    a.resize_size, a.input_size = args["resize_size"], args["input_size"]
    _, eval_tf = build_transforms(a)
    data_path = Path(args["data_path"])
    ds_val = BRSETMultiLabel(data_path / "val", eval_tf)
    ds_test = BRSETMultiLabel(data_path / "test", eval_tf)
    loader_val = DataLoader(ds_val, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    loader_test = DataLoader(ds_test, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    return loader_val, loader_test


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
    ckpt_a_path = sys.argv[1]
    ckpt_b_path = sys.argv[2]
    dataset_name = sys.argv[3]
    out_dir = Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a, args_a = load_model(ckpt_a_path, device)
    loader_val, loader_test = get_loaders(args_a)

    print(f"[{dataset_name}] model A: {ckpt_a_path}")
    y_val_true, y_val_prob_a = run_inference(model_a, loader_val, device, tta=True)
    y_test_true, y_test_prob_a = run_inference(model_a, loader_test, device, tta=True)
    del model_a
    torch.cuda.empty_cache()

    model_b, _ = load_model(ckpt_b_path, device)
    print(f"[{dataset_name}] model B: {ckpt_b_path}")
    _, y_val_prob_b = run_inference(model_b, loader_val, device, tta=True)
    _, y_test_prob_b = run_inference(model_b, loader_test, device, tta=True)
    del model_b
    torch.cuda.empty_cache()

    y_val_prob_ens = (y_val_prob_a + y_val_prob_b) / 2
    y_test_prob_ens = (y_test_prob_a + y_test_prob_b) / 2

    thresholds = tune_thresholds(y_val_true, y_val_prob_ens)
    per_label, macro_auc, macro_f1 = compute_per_label_metrics(y_test_true, y_test_prob_ens, thresholds)

    print(f"\n{dataset_name} ENSEMBLE (average of models A+B) test results:")
    for col in LABEL_COLS:
        m = per_label[col]
        print(f"  {col}: AUC={m['auc']:.4f} F1={m['f1']:.4f} precision={m['precision']:.4f} "
              f"recall={m['recall']:.4f} accuracy={m['accuracy']:.4f} threshold={m['threshold']:.2f}")
    print(f"  macro_auc={macro_auc:.4f} macro_f1={macro_f1:.4f}")

    import json
    with open(out_dir / "ensemble_metrics_test.json", "w") as f:
        json.dump({"per_label": per_label, "macro_auc": macro_auc, "macro_f1": macro_f1}, f, indent=2)

    rng = np.random.default_rng(SEED)
    print(f"\n{dataset_name} ENSEMBLE bootstrap 95% CIs ({N_BOOTSTRAP} resamples):")
    ci_results = {}
    for i, col in enumerate(LABEL_COLS):
        res = bootstrap_metric(y_test_true[:, i], y_test_prob_ens[:, i], thresholds[i], rng)
        ci_results[col] = res
        print(f"\n{col} (threshold={thresholds[i]:.2f}):")
        for metric, (mean, lo, hi) in res.items():
            print(f"  {metric:<10} {mean:.4f}  [95% CI: {lo:.4f} - {hi:.4f}]")

    with open(out_dir / "ensemble_bootstrap_ci.json", "w") as f:
        json.dump(ci_results, f, indent=2)
    print(f"\nwrote results to {out_dir}")


if __name__ == "__main__":
    main()
