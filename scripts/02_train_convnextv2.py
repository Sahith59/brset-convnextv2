"""
Fine-tune ConvNeXt V2 (timm: convnextv2_base.fcmae_ft_in22k_in1k) on full BRSET
for 5-class ICDR diabetic retinopathy grading.

Reference: Nakayama et al., "A Brazilian multilabel ophthalmological dataset
(BRSET)", PLOS Digital Health 2024. The paper's own recipe for this task was
Adam (lr=1e-5), weighted cross-entropy, 50 epochs with early-stopping patience
7, raw 0-1 pixel normalization, 256->224 resize/crop, evaluated via AUC-ROC and
macro F1. We deliberately diverge on several of these (see README/report for
the full comparison table): AdamW/1e-4 + ImageNet mean/std normalization
(correct for our ImageNet-pretrained checkpoint, vs. the paper's own
from-scratch-style normalization), unweighted loss for this first baseline
run, 100 epochs with best-checkpoint selection instead of patience-based early
stopping (for consistency with the mBRSET/RETFound pipeline), and a wider
metric set (accuracy, hamming loss, macro jaccard/average-precision/kappa, in
addition to the paper's AUC/F1). Input resolution (224 via 256 resize) and the
core task/label definition (DR_ICDR, 0-4) match the paper exactly.

Validation/test metrics mirror the RETFound/mBRSET pipeline
(../mbrset-retfound/RETFound_MAE/engine_finetune.py) for direct comparability
between the two projects: accuracy, hamming loss, macro jaccard, macro average
precision, kappa, macro F1, macro AUC (one-vs-rest), macro precision/recall,
and a composite score = (macro_f1 + macro_auc + kappa) / 3 used for
best-checkpoint selection.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, average_precision_score, cohen_kappa_score,
    confusion_matrix, f1_score, hamming_loss, jaccard_score,
    precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="convnextv2_base.fcmae_ft_in22k_in1k")
    p.add_argument("--data_path", default="/home/users/sthummala2/brset-convnextv2/data/finetune_icdr5")
    p.add_argument("--nb_classes", default=5, type=int)
    p.add_argument("--input_size", default=224, type=int)
    p.add_argument("--resize_size", default=256, type=int)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--warmup_epochs", default=5, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--weight_decay", default=0.05, type=float)
    p.add_argument("--label_smoothing", default=0.1, type=float)
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--ckpt_freq", default=10, type=int)
    p.add_argument("--class_weighted_loss", action="store_true", default=False)
    p.add_argument("--use_weighted_sampler", action="store_true", default=False,
                    help="Oversample rare classes at the data-loading level (inverse-frequency "
                         "WeightedRandomSampler) instead of only reweighting the loss.")
    p.add_argument("--focal_gamma", default=0.0, type=float,
                    help="If >0, use focal loss with this focusing exponent instead of plain "
                         "cross-entropy. Combines with --class_weighted_loss as the alpha term.")
    p.add_argument("--task", default="convnextv2_base_BRSET_icdr5_finetune")
    p.add_argument("--output_dir", default="/home/users/sthummala2/brset-convnextv2/results")
    p.add_argument("--seed", default=0, type=int)
    return p.parse_args()


def build_transforms(args):
    train_tf = transforms.Compose([
        transforms.Resize((args.resize_size, args.resize_size)),
        transforms.RandomCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.resize_size, args.resize_size)),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017): down-weights easy/well-classified examples and
    concentrates gradient signal on whatever the model is currently getting wrong,
    rather than a static per-class weight. Optionally combined with class weights
    (alpha) on top of the focusing term."""

    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                              label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def compute_metrics(y_true, y_pred, y_prob, num_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    true_onehot = np.eye(num_classes)[y_true]

    accuracy = accuracy_score(y_true, y_pred)
    hamming = hamming_loss(true_onehot, np.eye(num_classes)[y_pred])
    jaccard = jaccard_score(y_true, y_pred, average="macro", zero_division=0)
    avg_prec = average_precision_score(true_onehot, y_prob, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(true_onehot, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    score = (f1 + auc + kappa) / 3
    return {
        "accuracy": accuracy, "hamming": hamming, "jaccard": jaccard,
        "average_precision": avg_prec, "kappa": kappa, "f1": f1,
        "roc_auc": auc, "precision": precision, "recall": recall, "score": score,
    }


@torch.no_grad()
def run_eval(model, loader, device, num_classes):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)
        y_true.extend(targets.numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
    return compute_metrics(y_true, y_pred, y_prob, num_classes), (y_true, y_pred)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, epoch, print_freq=20):
    model.train()
    running_loss, n_batches = 0.0, 0
    t0 = time.time()
    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        n_batches += 1
        if i % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch: [{epoch}]  [{i}/{len(loader)}]  lr: {lr:.6f}  loss: {loss.item():.4f}  "
                  f"avg_loss: {running_loss / n_batches:.4f}  elapsed: {time.time() - t0:.1f}s", flush=True)
    scheduler.step()
    return running_loss / max(n_batches, 1)


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xlabel("Predicted Classes")
    ax.set_ylabel("Actual Classes")
    ax.set_title("Confusion Matrix (Normalized)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.4f}", ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"

    print(f"Namespace: {vars(args)}", flush=True)

    train_tf, eval_tf = build_transforms(args)
    data_path = Path(args.data_path)
    dataset_train = datasets.ImageFolder(data_path / "train", transform=train_tf)
    dataset_val = datasets.ImageFolder(data_path / "val", transform=eval_tf)
    dataset_test = datasets.ImageFolder(data_path / "test", transform=eval_tf)
    class_names = dataset_train.classes
    print(f"class_to_idx: {dataset_train.class_to_idx}", flush=True)

    if args.use_weighted_sampler:
        counts = np.bincount([label for _, label in dataset_train.samples], minlength=args.nb_classes)
        inv_freq = 1.0 / np.maximum(counts, 1)
        sample_weights = [inv_freq[label] for _, label in dataset_train.samples]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True)
        print(f"use_weighted_sampler: train counts={counts.tolist()}, "
              f"per-class inverse-freq weight={inv_freq.tolist()}", flush=True)
        loader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=True)
    else:
        loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=True)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"train/val/test sizes: {len(dataset_train)}/{len(dataset_val)}/{len(dataset_test)}", flush=True)

    model = timm.create_model(args.model, pretrained=True, num_classes=args.nb_classes)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_parameters: {n_params}", flush=True)

    class_weights = None
    if args.class_weighted_loss:
        counts = np.bincount([label for _, label in dataset_train.samples], minlength=args.nb_classes)
        class_weights = len(dataset_train.samples) / (args.nb_classes * np.maximum(counts, 1))
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print(f"class_weighted_loss: train counts={counts.tolist()}, weights={class_weights.tolist()}", flush=True)

    if args.focal_gamma > 0:
        print(f"using focal loss, gamma={args.focal_gamma}, alpha_weighted={args.class_weighted_loss}", flush=True)
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
    scaler = torch.amp.GradScaler("cuda")

    best_score, best_epoch = -1.0, -1
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, loader_train, optimizer, scheduler, scaler, criterion, device, epoch)
        val_metrics, _ = run_eval(model, loader_val, device, args.nb_classes)
        print(f"val: accuracy={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f} "
              f"roc_auc={val_metrics['roc_auc']:.4f} kappa={val_metrics['kappa']:.4f} "
              f"score={val_metrics['score']:.4f}", flush=True)

        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": train_loss, **val_metrics}) + "\n")

        if val_metrics["score"] > best_score:
            best_score, best_epoch = val_metrics["score"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                       out_dir / "checkpoint-best.pth")
        print(f"Best epoch = {best_epoch}, Best score = {best_score:.4f}", flush=True)

        if args.ckpt_freq and ((epoch + 1) % args.ckpt_freq == 0 or epoch == args.epochs - 1):
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                       out_dir / f"checkpoint-epoch{epoch + 1}.pth")
            print(f"Saved periodic checkpoint: checkpoint-epoch{epoch + 1}.pth", flush=True)

    # ---- Final test with best checkpoint ----
    best_ckpt = torch.load(out_dir / "checkpoint-best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    model.to(device)
    print(f"Test with the best model, epoch = {best_ckpt['epoch']}:", flush=True)
    test_metrics, (y_true, y_pred) = run_eval(model, loader_test, device, args.nb_classes)
    print(f"TEST: accuracy={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f} "
          f"roc_auc={test_metrics['roc_auc']:.4f} precision={test_metrics['precision']:.4f} "
          f"recall={test_metrics['recall']:.4f} kappa={test_metrics['kappa']:.4f} "
          f"score={test_metrics['score']:.4f}", flush=True)

    import csv
    with open(out_dir / "metrics_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(test_metrics.keys()))
        w.writerow(list(test_metrics.values()))

    plot_confusion_matrix(y_true, y_pred, class_names, out_dir / "confusion_matrix_test.jpg")

    total_time = time.time() - start_time
    print(f"Training time {total_time / 3600:.2f} hours", flush=True)


if __name__ == "__main__":
    main()
