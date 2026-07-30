"""
Cross-dataset generalization test, per Dr. Ye's direction: evaluate the
BRSET-trained model (the ensemble of the original + regularized checkpoints,
BRSET's official final model) directly on mBRSET - no retraining, no
threshold re-tuning on mBRSET data. This tests whether what the model
learned is genuine retinal disease signal or something specific to BRSET's
own cameras/population.

Evaluated on mBRSET's test split (732 images) specifically, the same
held-out set used to report mBRSET's own regularized model's numbers, for a
clean, direct, same-images comparison between in-domain and cross-domain
performance. BRSET's own val-tuned F1-optimal thresholds (0.61 DR, 0.39 ME)
are applied as-is.
"""
import json
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
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
LABEL_COLS = train_mod.LABEL_COLS

RESULTS_DIR = Path("/home/users/sthummala2/brset-convnextv2/results")
MBRSET_DATA_PATH = "/home/users/sthummala2/brset-convnextv2/data/finetune_mbrset_multilabel"
BRSET_THRESHOLDS = {"diabetic_retinopathy": 0.61, "macular_edema": 0.39}
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

    # Load the BRSET ensemble's two checkpoints, but point their eval transforms
    # and dataset at mBRSET's test split instead of BRSET's.
    ckpt_a_path = RESULTS_DIR / "convnextv2_large_BRSET_multilabel_512/checkpoint-best.pth"
    ckpt_b_path = RESULTS_DIR / "convnextv2_large_BRSET_multilabel_512_regularized/checkpoint-best.pth"

    model_a, args_a = load_model(ckpt_a_path, device)
    class A: pass
    a = A()
    a.resize_size, a.input_size = args_a["resize_size"], args_a["input_size"]
    _, eval_tf = build_transforms(a)

    dataset_mbrset_test = BRSETMultiLabel(Path(MBRSET_DATA_PATH) / "test", eval_tf)
    loader = DataLoader(dataset_mbrset_test, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    print(f"mBRSET test set: n={len(dataset_mbrset_test)}")

    print("Running BRSET original checkpoint on mBRSET test set (TTA)...")
    y_true, y_prob_a = run_inference(model_a, loader, device, tta=True)
    del model_a; torch.cuda.empty_cache()

    model_b, _ = load_model(ckpt_b_path, device)
    print("Running BRSET regularized checkpoint on mBRSET test set (TTA)...")
    _, y_prob_b = run_inference(model_b, loader, device, tta=True)
    del model_b; torch.cuda.empty_cache()

    y_prob_ensemble = (y_prob_a + y_prob_b) / 2

    out_dir = RESULTS_DIR / "cross_dataset_BRSET_on_mBRSET"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "predictions.npz", y_true=y_true, y_prob=y_prob_ensemble)

    print(f"\n{'='*70}\nBRSET-trained model (ensemble) evaluated on mBRSET test set\n{'='*70}")
    rng = np.random.default_rng(SEED)
    results = {}
    cm_data = {}
    for i, col in enumerate(LABEL_COLS):
        yt, yp = y_true[:, i], y_prob_ensemble[:, i]
        threshold = BRSET_THRESHOLDS[col]
        pred = (yp >= threshold).astype(int)

        auc = roc_auc_score(yt, yp)
        f1 = f1_score(yt, pred, zero_division=0)
        precision = precision_score(yt, pred, zero_division=0)
        recall = recall_score(yt, pred, zero_division=0)
        accuracy = accuracy_score(yt, pred)
        cm = confusion_matrix(yt, pred, labels=[0, 1])
        cm_data[col] = cm.tolist()

        print(f"\n{col} (BRSET's own threshold={threshold}, applied as-is):")
        print(f"  AUC={auc:.4f}  F1={f1:.4f}  precision={precision:.4f}  recall={recall:.4f}  accuracy={accuracy:.4f}")
        print(f"  confusion matrix [[TN,FP],[FN,TP]]: {cm.tolist()}")

        ci = bootstrap_metric(yt, yp, threshold, rng)
        print(f"  bootstrap 95% CI:")
        for metric, (mean, lo, hi) in ci.items():
            print(f"    {metric:<10} {mean:.4f}  [{lo:.4f} - {hi:.4f}]")

        results[col] = {
            "threshold": threshold, "auc": auc, "f1": f1, "precision": precision,
            "recall": recall, "accuracy": accuracy, "confusion_matrix": cm.tolist(),
            "bootstrap_ci": ci,
        }

    macro_auc = np.mean([results[c]["auc"] for c in LABEL_COLS])
    macro_f1 = np.mean([results[c]["f1"] for c in LABEL_COLS])
    print(f"\nmacro_auc={macro_auc:.4f} macro_f1={macro_f1:.4f}")
    results["macro_auc"] = float(macro_auc)
    results["macro_f1"] = float(macro_f1)

    with open(out_dir / "cross_dataset_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_dir / 'cross_dataset_metrics.json'}")


if __name__ == "__main__":
    main()
