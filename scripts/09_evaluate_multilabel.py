"""
Standalone final evaluation for the multi-label BRSET model, for use when a
training run is stopped before its natural completion (so the training
script's own end-of-run test block never executes). Loads a checkpoint,
re-tunes per-label thresholds on the validation set (with TTA), then reports
full test-set metrics (with TTA) the same way the training script would have.
"""
import json
import sys
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader

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


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/users/sthummala2/brset-convnextv2/results/convnextv2_large_BRSET_multilabel_512/checkpoint-best.pth"
    out_path = sys.argv[2] if len(sys.argv) > 2 else \
        "/home/users/sthummala2/brset-convnextv2/results/multilabel_classification_report.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, args: {args}")

    model = timm.create_model(args["model"], pretrained=False, num_classes=args["nb_classes"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    class A: pass
    a = A()
    a.resize_size, a.input_size = args["resize_size"], args["input_size"]
    _, eval_tf = build_transforms(a)

    data_path = Path(args["data_path"])
    dataset_val = BRSETMultiLabel(data_path / "val", eval_tf)
    dataset_test = BRSETMultiLabel(data_path / "test", eval_tf)
    loader_val = DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    loader_test = DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    print(f"val/test sizes: {len(dataset_val)}/{len(dataset_test)}")

    print("Running val inference with TTA to tune thresholds...")
    y_val_true, y_val_prob = run_inference(model, loader_val, device, tta=True)
    thresholds = tune_thresholds(y_val_true, y_val_prob)
    print(f"tuned thresholds: {dict(zip(LABEL_COLS, thresholds))}")

    print("Running test inference with TTA...")
    y_test_true, y_test_prob = run_inference(model, loader_test, device, tta=True)
    per_label, macro_auc, macro_f1 = compute_per_label_metrics(y_test_true, y_test_prob, thresholds)

    lines = [f"BRSET ConvNeXt V2 Large @ 512, multi-label classification report",
             f"Checkpoint epoch: {ckpt['epoch']}, n_test={len(dataset_test)}", ""]
    for col in LABEL_COLS:
        m = per_label[col]
        lines.append(f"{col}:")
        lines.append(f"  threshold={m['threshold']:.2f}  AUC={m['auc']:.4f}  F1={m['f1']:.4f}  "
                      f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  accuracy={m['accuracy']:.4f}")
        lines.append(f"  confusion matrix [[TN,FP],[FN,TP]]: {m['confusion_matrix']}")
        lines.append("")
    lines.append(f"MACRO: AUC={macro_auc:.4f}  F1={macro_f1:.4f}")

    report = "\n".join(lines)
    print(report)
    with open(out_path, "w") as f:
        f.write(report)
    with open(out_path.replace(".txt", ".json"), "w") as f:
        json.dump({"epoch": ckpt["epoch"], "per_label": per_label, "macro_auc": macro_auc, "macro_f1": macro_f1}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
