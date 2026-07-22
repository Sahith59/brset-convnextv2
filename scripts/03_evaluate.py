"""
Full multi-class evaluation of a trained ConvNeXt V2 checkpoint on the held-out
BRSET test set: per-class precision/recall/F1/support, macro & weighted
averages, overall accuracy, confusion matrix, and one-vs-rest macro AUC.

Mirrors ../mbrset-retfound/scripts/05_evaluate_icdr5.py exactly, for direct
side-by-side comparison with the mBRSET/RETFound results.
"""
import sys

import numpy as np
import timm
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_CKPT = "/home/users/sthummala2/brset-convnextv2/results/convnextv2_base_BRSET_icdr5_finetune/checkpoint-best.pth"
DEFAULT_OUT = "/home/users/sthummala2/brset-convnextv2/results/icdr5_classification_report.txt"
DATA_PATH = "/home/users/sthummala2/brset-convnextv2/data/finetune_icdr5"


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]

    model = timm.create_model(args["model"], pretrained=False, num_classes=args["nb_classes"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    eval_tf = transforms.Compose([
        transforms.Resize((args["resize_size"], args["resize_size"])),
        transforms.CenterCrop(args["input_size"]),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset_test = datasets.ImageFolder(f"{DATA_PATH}/test", transform=eval_tf)
    class_to_idx = dataset_test.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    ordered_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    loader = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=4)

    all_true, all_pred, all_probs = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)
            preds = probs.argmax(dim=1)
            all_true.extend(targets.numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_probs = np.array(all_probs)

    accuracy = (all_true == all_pred).mean()
    report = classification_report(
        all_true, all_pred, labels=list(range(len(ordered_names))),
        target_names=ordered_names, digits=4, zero_division=0,
    )
    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(ordered_names))))

    try:
        macro_auc_ovr = roc_auc_score(all_true, all_probs, multi_class="ovr", average="macro")
    except ValueError as e:
        macro_auc_ovr = float("nan")
        print(f"WARNING: could not compute macro AUC-OVR ({e})")

    cm_lines = ["Confusion matrix (rows=actual, cols=predicted):", "cols: " + ", ".join(ordered_names)]
    for name, row in zip(ordered_names, cm):
        cm_lines.append(f"{name:>20s}: {row.tolist()}")

    out = (
        f"BRSET ConvNeXt V2 ICDR-5 classification report on held-out test set\n"
        f"n_test = {len(all_true)}  class_to_idx = {class_to_idx}\n\n"
        f"Overall accuracy: {accuracy:.4f}\n"
        f"Macro AUC (one-vs-rest): {macro_auc_ovr:.4f}\n\n"
        f"Per-class report:\n{report}\n"
        + "\n".join(cm_lines) + "\n"
    )
    print(out)
    with open(out_path, "w") as f:
        f.write(out)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
