"""
Follow-up to the cross-dataset test: is the recall collapse a genuine
generalization failure, or just a threshold/calibration shift between
BRSET's and mBRSET's probability distributions? The model weights are NOT
retouched here - only the decision threshold is re-tuned, using mBRSET's own
validation split run through the same frozen BRSET-trained ensemble, then
applied to mBRSET's test set. This isolates calibration from raw
discriminative ability (AUC is unaffected either way).
"""
import json
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

import sys
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

RESULTS_DIR = Path("/home/users/sthummala2/brset-convnextv2/results")
MBRSET_DATA_PATH = "/home/users/sthummala2/brset-convnextv2/data/finetune_mbrset_multilabel"
N_BOOTSTRAP = 2000
SEED = 42


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    model = timm.create_model(args["model"], pretrained=False, num_classes=args["nb_classes"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, args


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_a_path = RESULTS_DIR / "convnextv2_large_BRSET_multilabel_512/checkpoint-best.pth"
    ckpt_b_path = RESULTS_DIR / "convnextv2_large_BRSET_multilabel_512_regularized/checkpoint-best.pth"

    model_a, args_a = load_model(ckpt_a_path, device)
    class A: pass
    a = A()
    a.resize_size, a.input_size = args_a["resize_size"], args_a["input_size"]
    _, eval_tf = build_transforms(a)

    dataset_mbrset_val = BRSETMultiLabel(Path(MBRSET_DATA_PATH) / "val", eval_tf)
    loader_val = DataLoader(dataset_mbrset_val, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    print(f"mBRSET val set: n={len(dataset_mbrset_val)}")

    print("Running BRSET checkpoints on mBRSET VAL set (to re-tune threshold only)...")
    y_val_true, y_val_prob_a = run_inference(model_a, loader_val, device, tta=True)
    del model_a; torch.cuda.empty_cache()
    model_b, _ = load_model(ckpt_b_path, device)
    _, y_val_prob_b = run_inference(model_b, loader_val, device, tta=True)
    del model_b; torch.cuda.empty_cache()
    y_val_prob_ensemble = (y_val_prob_a + y_val_prob_b) / 2

    recalibrated_thresholds = tune_thresholds(y_val_true, y_val_prob_ensemble)
    print(f"recalibrated thresholds (tuned on mBRSET val, model weights untouched): "
          f"{dict(zip(LABEL_COLS, recalibrated_thresholds))}")

    # reuse the already-computed test-set probabilities from script 24
    d_test = np.load(RESULTS_DIR / "cross_dataset_BRSET_on_mBRSET/predictions.npz")
    y_test_true, y_test_prob = d_test["y_true"], d_test["y_prob"]

    rng = np.random.default_rng(SEED)
    print(f"\n{'='*70}\nRecalibrated cross-dataset result (threshold re-tuned, weights frozen)\n{'='*70}")
    results = {}
    for i, col in enumerate(LABEL_COLS):
        yt, yp = y_test_true[:, i], y_test_prob[:, i]
        t = recalibrated_thresholds[i]
        pred = (yp >= t).astype(int)
        auc = roc_auc_score(yt, yp)
        f1 = f1_score(yt, pred, zero_division=0)
        precision = precision_score(yt, pred, zero_division=0)
        recall = recall_score(yt, pred, zero_division=0)
        print(f"\n{col} (recalibrated threshold={t:.2f}):")
        print(f"  AUC={auc:.4f}  F1={f1:.4f}  precision={precision:.4f}  recall={recall:.4f}")
        ci = bootstrap_metric(yt, yp, t, rng)
        for metric, (mean, lo, hi) in ci.items():
            print(f"    {metric:<10} {mean:.4f}  [{lo:.4f} - {hi:.4f}]")
        results[col] = {"threshold": float(t), "auc": auc, "f1": f1, "precision": precision,
                         "recall": recall, "bootstrap_ci": ci}

    with open(RESULTS_DIR / "cross_dataset_BRSET_on_mBRSET/recalibrated_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote recalibrated_metrics.json")


if __name__ == "__main__":
    main()
