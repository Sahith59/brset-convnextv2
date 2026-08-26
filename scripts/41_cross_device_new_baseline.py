"""Cross-device evaluation of the NEW Part-1 baseline on mBRSET.

The earlier cross-device numbers (script 24) were produced with the two-model
BRSET ensemble. Part 1 has since selected a better single model
(focal loss, oversampling OFF, 40 epochs), which had never been tested on
mBRSET. This script repeats the transfer test with that model.

It reports the first three points of the gap decomposition:

  1. transferred as is   - BRSET's own validation-chosen cutoff, applied unchanged
  2. cutoff retuned      - cutoff chosen on the mBRSET VALIDATION split, applied to test
                           (never chosen on test, so this number is honest)
  3. best at any cutoff  - exhaustive sweep on test. This is an oracle upper bound
                           for every method that only rescales scores.

The fourth point (a model trained on mBRSET) comes from scripts 42 and 43.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader

_spec = importlib.util.spec_from_file_location(
    "train30", str(Path(__file__).parent / "30_train_strong_baseline.py"))
t30 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t30)

RESULTS = Path("/home/users/sthummala2/brset-convnextv2/results")
MBRSET = Path("/home/users/sthummala2/brset-convnextv2/data/finetune_mbrset_multilabel")
SRC_RUN = "afl40_focal_control"           # the selected Part-1 model
OUT_DIR = RESULTS / "cross_device_newbaseline_BRSET_on_mBRSET"
LABELS = t30.LABEL_COLS
N_BOOT = 2000
SEED = 42
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


class _A:
    pass


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = timm.create_model(a["model"], pretrained=False, num_classes=a["nb_classes"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"loaded {ckpt_path}  (epoch {ck['epoch']}, variant {ck['variant']})", flush=True)
    return model, a


def infer(model, split_dir, eval_tf, device, tta):
    ds = t30.BRSETMultiLabel(split_dir, eval_tf)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    y, p = t30.run_inference(model, dl, device, tta=tta)
    print(f"  {split_dir.name}: n={len(ds)}", flush=True)
    return y, p


def boot_ci(yt, yp, thr, rng):
    out = {k: [] for k in ("auc", "f1", "precision", "recall")}
    n = len(yt)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        a, b = yt[idx], yp[idx]
        if a.sum() == 0 or a.sum() == n:
            continue
        pred = (b >= thr).astype(int)
        out["auc"].append(roc_auc_score(a, b))
        out["f1"].append(f1_score(a, pred, zero_division=0))
        out["precision"].append(precision_score(a, pred, zero_division=0))
        out["recall"].append(recall_score(a, pred, zero_division=0))
    return {k: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in out.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src_metrics = json.load(open(RESULTS / SRC_RUN / "metrics_test.json"))
    src_thr = {c: src_metrics["per_label"][c]["threshold"] for c in LABELS}
    print(f"source model      : {SRC_RUN}")
    print(f"source cutoffs    : {src_thr}   (chosen on BRSET validation)\n", flush=True)

    model, a = load_model(RESULTS / SRC_RUN / "checkpoint-best.pth", device)
    cfg = _A()
    cfg.resize_size, cfg.input_size = a["resize_size"], a["input_size"]
    _, eval_tf = t30.build_transforms(cfg)
    tta = a.get("tta", "flip4")

    print("running inference on mBRSET (tta=%s)..." % tta, flush=True)
    y_val, p_val = infer(model, MBRSET / "val", eval_tf, device, tta)
    y_te, p_te = infer(model, MBRSET / "test", eval_tf, device, tta)
    np.savez(OUT_DIR / "predictions.npz", y_true=y_te, y_prob=p_te,
             y_true_val=y_val, y_prob_val=p_val)

    rng = np.random.default_rng(SEED)
    results = {}
    print(f"\n{'='*78}")
    print(f"NEW Part-1 baseline ({SRC_RUN}) evaluated on mBRSET test")
    print(f"{'='*78}")

    for i, c in enumerate(LABELS):
        yt, yp = y_te[:, i].astype(int), p_te[:, i]
        yv, pv = y_val[:, i].astype(int), p_val[:, i]

        thr_src = src_thr[c]
        thr_ret = float(max(((f1_score(yv, (pv >= t).astype(int), zero_division=0)), t)
                            for t in GRID)[1])
        f1_orc, thr_orc = max(((f1_score(yt, (yp >= t).astype(int), zero_division=0)), t)
                              for t in GRID)

        auc = roc_auc_score(yt, yp)
        row = {"auc": float(auc), "n": int(len(yt)), "n_pos": int(yt.sum())}
        print(f"\n{c}   (n={len(yt)}, positives={yt.sum()})")
        print(f"  AUC = {auc:.4f}")

        for tag, thr in (("1_transferred_as_is", thr_src),
                         ("2_cutoff_retuned_on_mbrset_val", thr_ret),
                         ("3_best_at_any_cutoff_oracle", float(thr_orc))):
            pred = (yp >= thr).astype(int)
            cm = confusion_matrix(yt, pred, labels=[0, 1])
            m = {"threshold": float(thr),
                 "f1": float(f1_score(yt, pred, zero_division=0)),
                 "precision": float(precision_score(yt, pred, zero_division=0)),
                 "recall": float(recall_score(yt, pred, zero_division=0)),
                 "accuracy": float(accuracy_score(yt, pred)),
                 "confusion_matrix": cm.tolist()}
            row[tag] = m
            print(f"  {tag:34s} thr={thr:.2f}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  missed={cm[1][0]}/{cm[1].sum()}")

        row["bootstrap_ci_at_source_threshold"] = boot_ci(yt, yp, thr_src, rng)
        ci = row["bootstrap_ci_at_source_threshold"]
        print(f"  bootstrap 95% CI at source cutoff:")
        for k, (mn, lo, hi) in ci.items():
            print(f"    {k:<10} {mn:.4f}  [{lo:.4f}, {hi:.4f}]")
        results[c] = row

    json.dump(results, open(OUT_DIR / "cross_device_metrics.json", "w"), indent=2)
    print(f"\nwrote {OUT_DIR}/cross_device_metrics.json", flush=True)


if __name__ == "__main__":
    main()
