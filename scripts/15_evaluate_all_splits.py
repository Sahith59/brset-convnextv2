"""
Evaluate a trained multi-label checkpoint on ALL THREE splits (train, val,
test) rather than just test - train/val accuracy were never computed during
training (only macro AUC/F1 were logged per epoch), so this fills that gap
and doubles as a direct overfitting check (train accuracy vs test accuracy).

Thresholds are tuned once on val (no TTA, for speed and consistency across
all three splits) and applied identically to train/val/test.
"""
import sys
from pathlib import Path

import timm
import torch
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


def main():
    ckpt_path = sys.argv[1]
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "dataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    print(f"[{dataset_name}] checkpoint epoch {ckpt['epoch']}, data_path={args['data_path']}")

    model = timm.create_model(args["model"], pretrained=False, num_classes=args["nb_classes"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    class A: pass
    a = A()
    a.resize_size, a.input_size = args["resize_size"], args["input_size"]
    _, eval_tf = build_transforms(a)

    data_path = Path(args["data_path"])
    results = {}
    probs_cache = {}
    for split in ("train", "val", "test"):
        ds = BRSETMultiLabel(data_path / split, eval_tf)
        loader = DataLoader(ds, batch_size=24, shuffle=False, num_workers=8, pin_memory=True)
        y_true, y_prob = run_inference(model, loader, device, tta=False)
        probs_cache[split] = (y_true, y_prob)
        print(f"  {split}: n={len(ds)}")

    thresholds = tune_thresholds(*probs_cache["val"])
    print(f"  thresholds (tuned on val): {dict(zip(LABEL_COLS, thresholds))}")

    print(f"\n{'Split':<8}{'Label':<22}{'AUC':<8}{'F1':<8}{'Precision':<11}{'Recall':<9}{'Accuracy':<9}")
    for split in ("train", "val", "test"):
        y_true, y_prob = probs_cache[split]
        per_label, macro_auc, macro_f1 = compute_per_label_metrics(y_true, y_prob, thresholds)
        for col in LABEL_COLS:
            m = per_label[col]
            print(f"{split:<8}{col:<22}{m['auc']:<8.4f}{m['f1']:<8.4f}{m['precision']:<11.4f}"
                  f"{m['recall']:<9.4f}{m['accuracy']:<9.4f}")
        print(f"{split:<8}{'MACRO':<22}{macro_auc:<8.4f}{macro_f1:<8.4f}")
        results[split] = {"per_label": per_label, "macro_auc": macro_auc, "macro_f1": macro_f1}

    out_path = Path(ckpt_path).parent / "metrics_all_splits.json"
    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
