"""
Phase 1 of the MoE step Dr. Ye asked for: a simple gated Mixture-of-Experts
combining the BRSET-trained expert and the mBRSET-trained expert, evaluated
on mBRSET's test set (the same Exp-3 setting).

The RUS/PID decomposition (script 26) found close to zero unique information
in the BRSET-expert's decisions once you know the mBRSET-expert's decision,
but real (if modest) synergy - 9.6% of the joint information for DR, 1.5%
for ME - that only appears when both experts' outputs are combined. This
script tests whether a learned gate can actually capture that synergy and
beat mBRSET's own dedicated model, or whether the synergy is too small/noisy
to realize in practice.

Gate = per-label logistic regression on [p_BRSET_expert, p_mBRSET_expert,
p_BRSET_expert * p_mBRSET_expert] (the interaction term is what lets the gate
express synergy - a plain average of the two probabilities cannot).
Trained on mBRSET's VALIDATION split (never test), threshold tuned on
validation (F1-optimal), then applied once to mBRSET's held-out test set.
"""
import json
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.linear_model import LogisticRegression
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


def get_loader(args, split):
    class A: pass
    a = A()
    a.resize_size, a.input_size = args["resize_size"], args["input_size"]
    _, eval_tf = build_transforms(a)
    ds = BRSETMultiLabel(Path(MBRSET_DATA_PATH) / split, eval_tf)
    return DataLoader(ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)


def tune_threshold_f1(y_true, y_prob):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.02, 0.98, 0.01):
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


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

    # ---- Run BRSET ensemble (2 checkpoints) + mBRSET expert on mBRSET's VAL split ----
    ckpt_a_path = RESULTS_DIR / "convnextv2_large_BRSET_multilabel_512/checkpoint-best.pth"
    ckpt_b_path = RESULTS_DIR / "convnextv2_large_BRSET_multilabel_512_regularized/checkpoint-best.pth"
    ckpt_m_path = RESULTS_DIR / "convnextv2_large_mBRSET_multilabel_512_regularized/checkpoint-best.pth"

    model_a, args_a = load_model(ckpt_a_path, device)
    loader_val = get_loader(args_a, "val")
    print("Running BRSET original checkpoint on mBRSET VAL set...")
    y_val_true, prob_a_val = run_inference(model_a, loader_val, device, tta=True)
    del model_a; torch.cuda.empty_cache()

    model_b, _ = load_model(ckpt_b_path, device)
    print("Running BRSET regularized checkpoint on mBRSET VAL set...")
    _, prob_b_val = run_inference(model_b, loader_val, device, tta=True)
    del model_b; torch.cuda.empty_cache()

    prob_brset_val = (prob_a_val + prob_b_val) / 2

    model_m, args_m = load_model(ckpt_m_path, device)
    loader_val_m = get_loader(args_m, "val")
    print("Running mBRSET expert on mBRSET VAL set...")
    y_val_true_m, prob_mbrset_val = run_inference(model_m, loader_val_m, device, tta=True)
    del model_m; torch.cuda.empty_cache()
    assert np.array_equal(y_val_true, y_val_true_m)

    np.savez(RESULTS_DIR / "moe_val_predictions.npz",
             y_true=y_val_true, prob_brset=prob_brset_val, prob_mbrset=prob_mbrset_val)

    # ---- Load cached TEST predictions for both experts (already verified aligned) ----
    d_brset_test = np.load(RESULTS_DIR / "cross_dataset_BRSET_on_mBRSET/predictions.npz")
    d_mbrset_test = np.load(RESULTS_DIR / "convnextv2_large_mBRSET_multilabel_512_regularized/test_predictions.npz")
    y_test_true = d_mbrset_test["y_true"]
    prob_brset_test = d_brset_test["y_prob"]
    prob_mbrset_test = d_mbrset_test["y_prob"]

    print(f"\n{'='*70}\nGated MoE (logistic gate, trained on mBRSET val, evaluated on mBRSET test)\n{'='*70}")
    rng = np.random.default_rng(SEED)
    results = {}
    for i, col in enumerate(LABEL_COLS):
        pv_b, pv_m = prob_brset_val[:, i], prob_mbrset_val[:, i]
        pt_b, pt_m = prob_brset_test[:, i], prob_mbrset_test[:, i]
        yv, yt = y_val_true[:, i], y_test_true[:, i]

        X_val = np.stack([pv_b, pv_m, pv_b * pv_m], axis=1)
        X_test = np.stack([pt_b, pt_m, pt_b * pt_m], axis=1)

        gate = LogisticRegression(max_iter=1000)
        gate.fit(X_val, yv)
        gate_prob_val = gate.predict_proba(X_val)[:, 1]
        gate_prob_test = gate.predict_proba(X_test)[:, 1]

        threshold = tune_threshold_f1(yv, gate_prob_val)
        pred_test = (gate_prob_test >= threshold).astype(int)

        auc = roc_auc_score(yt, gate_prob_test)
        f1 = f1_score(yt, pred_test, zero_division=0)
        precision = precision_score(yt, pred_test, zero_division=0)
        recall = recall_score(yt, pred_test, zero_division=0)

        # mBRSET-alone baseline on the same test set, for direct comparison
        mbrset_alone_pred = (pt_m >= float(d_mbrset_test["thresholds"][i])).astype(int)
        f1_alone = f1_score(yt, mbrset_alone_pred, zero_division=0)

        print(f"\n{col}:")
        print(f"  gate coefficients [p_BRSET, p_mBRSET, interaction] = {gate.coef_[0]}, intercept={gate.intercept_[0]:.3f}")
        print(f"  gate threshold (tuned on val) = {threshold:.2f}")
        print(f"  MoE gate  -> AUC={auc:.4f} F1={f1:.4f} precision={precision:.4f} recall={recall:.4f}")
        print(f"  mBRSET-alone (for comparison) -> F1={f1_alone:.4f}")

        ci = bootstrap_metric(yt, gate_prob_test, threshold, rng)
        print(f"  bootstrap 95% CI:")
        for metric, (mean, lo, hi) in ci.items():
            print(f"    {metric:<10} {mean:.4f}  [{lo:.4f} - {hi:.4f}]")

        results[col] = {
            "gate_coefficients": gate.coef_[0].tolist(), "gate_intercept": float(gate.intercept_[0]),
            "threshold": float(threshold), "auc": float(auc), "f1": float(f1),
            "precision": float(precision), "recall": float(recall),
            "mbrset_alone_f1": float(f1_alone),
            "bootstrap_ci": ci,
        }

    with open(RESULTS_DIR / "moe_gate_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {RESULTS_DIR / 'moe_gate_results.json'}")


if __name__ == "__main__":
    main()
