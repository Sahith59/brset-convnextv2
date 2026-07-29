"""
Recall-focused (F2) threshold tuning for the two final models.

Correctly tunes both F1-optimal and F2-optimal thresholds on the VALIDATION
set (never the test set - tuning on test would be leakage/peeking, the same
mistake to avoid everywhere else in this project), then applies those fixed
thresholds to the held-out test set for final metrics + bootstrap CI.

  - BRSET final model = ensemble of original + regularized checkpoints,
    averaging both checkpoints' probabilities on val (for threshold tuning)
    and on the cached test predictions (for final evaluation).
  - mBRSET final model = regularized checkpoint alone.
"""
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score
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
LABEL_COLS = train_mod.LABEL_COLS

N_BOOTSTRAP = 2000
SEED = 42
RESULTS_DIR = Path("/home/users/sthummala2/brset-convnextv2/results")


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    model = timm.create_model(args["model"], pretrained=False, num_classes=args["nb_classes"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, args


def get_val_loader(args):
    class A: pass
    a = A()
    a.resize_size, a.input_size = args["resize_size"], args["input_size"]
    _, eval_tf = build_transforms(a)
    ds_val = BRSETMultiLabel(Path(args["data_path"]) / "val", eval_tf)
    return DataLoader(ds_val, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)


def tune_threshold_fbeta(y_true, y_prob, beta):
    best_t, best_score = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.02):
        pred = (y_prob >= t).astype(int)
        score = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        if score > best_score:
            best_score, best_t = score, t
    return best_t


def bootstrap_metric(y_true, y_prob, threshold, rng):
    n = len(y_true)
    aucs, f1s, f2s, precs, recs = [], [], [], [], []
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
        f1s.append(fbeta_score(yt, pred, beta=1, zero_division=0))
        f2s.append(fbeta_score(yt, pred, beta=2, zero_division=0))
        precs.append(precision_score(yt, pred, zero_division=0))
        recs.append(recall_score(yt, pred, zero_division=0))
    def ci(vals):
        vals = np.array(vals)
        return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    return {"auc": ci(aucs), "f1": ci(f1s), "f2": ci(f2s), "precision": ci(precs), "recall": ci(recs)}


def process(dataset_name, y_val_true, y_val_prob, y_test_true, y_test_prob, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print(f"\n{'='*70}\n{dataset_name}\n{'='*70}")
    results = {}
    for i, col in enumerate(LABEL_COLS):
        yv, pv = y_val_true[:, i], y_val_prob[:, i]
        yt, pt = y_test_true[:, i], y_test_prob[:, i]

        t_f1 = tune_threshold_fbeta(yv, pv, beta=1)   # tuned on VAL
        t_f2 = tune_threshold_fbeta(yv, pv, beta=2)   # tuned on VAL

        pred_f1 = (pt >= t_f1).astype(int)   # applied to TEST
        pred_f2 = (pt >= t_f2).astype(int)   # applied to TEST
        print(f"\n{col}:")
        print(f"  F1-optimal (tuned on val)  threshold={t_f1:.2f}  test F1={fbeta_score(yt,pred_f1,beta=1,zero_division=0):.4f}  "
              f"precision={precision_score(yt,pred_f1,zero_division=0):.4f}  recall={recall_score(yt,pred_f1,zero_division=0):.4f}")
        print(f"  F2-optimal (tuned on val)  threshold={t_f2:.2f}  test F1={fbeta_score(yt,pred_f2,beta=1,zero_division=0):.4f}  "
              f"precision={precision_score(yt,pred_f2,zero_division=0):.4f}  recall={recall_score(yt,pred_f2,zero_division=0):.4f}")

        ci = bootstrap_metric(yt, pt, t_f2, rng)
        print(f"  F2-threshold bootstrap 95% CI (test set):")
        for metric, (mean, lo, hi) in ci.items():
            print(f"    {metric:<10} {mean:.4f}  [{lo:.4f} - {hi:.4f}]")

        results[col] = {
            "f1_optimal_threshold_tuned_on_val": float(t_f1),
            "f2_optimal_threshold_tuned_on_val": float(t_f2),
            "test_at_f2_threshold": {
                "f1": float(fbeta_score(yt, pred_f2, beta=1, zero_division=0)),
                "f2": float(fbeta_score(yt, pred_f2, beta=2, zero_division=0)),
                "precision": float(precision_score(yt, pred_f2, zero_division=0)),
                "recall": float(recall_score(yt, pred_f2, zero_division=0)),
                "auc": float(roc_auc_score(yt, pt)),
            },
            "bootstrap_ci_at_f2_threshold": ci,
        }

    with open(out_dir / "recall_focused_thresholds.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_dir / 'recall_focused_thresholds.json'}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- BRSET final model: ensemble ----
    d_orig_test = np.load(RESULTS_DIR / "convnextv2_large_BRSET_multilabel_512/test_predictions.npz")
    d_reg_test = np.load(RESULTS_DIR / "convnextv2_large_BRSET_multilabel_512_regularized/test_predictions.npz")
    assert np.array_equal(d_orig_test["y_true"], d_reg_test["y_true"])
    y_test_true_brset = d_orig_test["y_true"]
    y_test_prob_brset_ensemble = (d_orig_test["y_prob"] + d_reg_test["y_prob"]) / 2

    model_a, args_a = load_model(RESULTS_DIR / "convnextv2_large_BRSET_multilabel_512/checkpoint-best.pth", device)
    loader_val = get_val_loader(args_a)
    y_val_true_brset, y_val_prob_a = run_inference(model_a, loader_val, device, tta=True)
    del model_a; torch.cuda.empty_cache()

    model_b, _ = load_model(RESULTS_DIR / "convnextv2_large_BRSET_multilabel_512_regularized/checkpoint-best.pth", device)
    _, y_val_prob_b = run_inference(model_b, loader_val, device, tta=True)
    del model_b; torch.cuda.empty_cache()

    y_val_prob_brset_ensemble = (y_val_prob_a + y_val_prob_b) / 2
    process("BRSET (final model = ensemble)", y_val_true_brset, y_val_prob_brset_ensemble,
            y_test_true_brset, y_test_prob_brset_ensemble,
            RESULTS_DIR / "convnextv2_large_BRSET_ensemble")

    # ---- mBRSET final model: regularized checkpoint alone ----
    d_mbrset_test = np.load(RESULTS_DIR / "convnextv2_large_mBRSET_multilabel_512_regularized/test_predictions.npz")
    model_c, args_c = load_model(RESULTS_DIR / "convnextv2_large_mBRSET_multilabel_512_regularized/checkpoint-best.pth", device)
    loader_val_m = get_val_loader(args_c)
    y_val_true_m, y_val_prob_m = run_inference(model_c, loader_val_m, device, tta=True)
    del model_c; torch.cuda.empty_cache()

    process("mBRSET (final model = regularized)", y_val_true_m, y_val_prob_m,
            d_mbrset_test["y_true"], d_mbrset_test["y_prob"],
            RESULTS_DIR / "convnextv2_large_mBRSET_multilabel_512_regularized")


if __name__ == "__main__":
    main()
