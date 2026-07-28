"""
Fine-tune ConvNeXt V2 Large on full BRSET for multi-label binary classification
of diabetic_retinopathy and macular_edema (independent 0/1 findings per image,
not the DR_ICDR severity scale used in the earlier 5-class work).

Per Dr. Ye's direction, this targets a strong, non-baseline model:
  - convnextv2_large.fcmae_ft_in22k_in1k_384 backbone, fine-tuned at a HIGHER
    input resolution (default 512) than its 384px pretraining. ConvNeXt is
    fully convolutional (global-average-pooled head), so it accepts a
    resolution mismatch from pretraining more gracefully than a ViT would;
    this is not a published, proven combination the way 384px is, so it is
    being tried deliberately as an experiment, not assumed to work.
  - Weighted oversampling + multi-label focal loss, the same combination
    already validated to help on this exact dataset's rare-class problem in
    the 5-class ICDR work (scripts/02_train_convnextv2.py), adapted here to
    per-label binary weighting/focusing instead of a single softmax class.
  - Per-label threshold tuning on the validation set (instead of a blind 0.5
    cutoff) and test-time augmentation (horizontal-flip averaging) at final
    test evaluation.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LABEL_COLS = ["diabetic_retinopathy", "macular_edema"]


class BRSETMultiLabel(Dataset):
    def __init__(self, split_dir, transform):
        self.split_dir = Path(split_dir)
        self.labels_df = pd.read_csv(self.split_dir / "labels.csv")
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img = Image.open(self.split_dir / row["file"]).convert("RGB")
        img = self.transform(img)
        target = torch.tensor([row[c] for c in LABEL_COLS], dtype=torch.float32)
        return img, target


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="convnextv2_large.fcmae_ft_in22k_in1k_384")
    p.add_argument("--data_path", default="/home/users/sthummala2/brset-convnextv2/data/finetune_multilabel")
    p.add_argument("--nb_classes", default=2, type=int)
    p.add_argument("--input_size", default=512, type=int)
    p.add_argument("--resize_size", default=560, type=int)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--accum_iter", default=4, type=int, help="Gradient accumulation steps (effective batch = batch_size * accum_iter)")
    p.add_argument("--epochs", default=40, type=int)
    p.add_argument("--warmup_epochs", default=3, type=int)
    p.add_argument("--lr", default=5e-5, type=float)
    p.add_argument("--weight_decay", default=0.05, type=float)
    p.add_argument("--drop_path", default=0.0, type=float,
                    help="Stochastic depth rate. Was never actually set before (defaulted to "
                         "~0), a real gap given the confirmed train-vs-test overfitting.")
    p.add_argument("--mixup_alpha", default=0.0, type=float,
                    help="If >0, multi-label mixup: blend pairs of images and their multi-hot "
                         "label vectors with lambda~Beta(alpha,alpha) each batch.")
    p.add_argument("--label_smoothing", default=0.0, type=float,
                    help="Soften hard 0/1 BCE targets: y -> y*(1-eps) + 0.5*eps.")
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--ckpt_freq", default=5, type=int)
    p.add_argument("--focal_gamma", default=2.0, type=float)
    p.add_argument("--focal_alpha_weighted", action="store_true", default=False,
                    help="Add per-label pos_weight (alpha) to the focal loss, computed from "
                         "training label frequency. Combined with a lower --focal_gamma, this "
                         "trades some of the pure focal margin-sharpening for better-calibrated "
                         "probabilities (targets the AUC-high/F1-lower pattern seen without it).")
    p.add_argument("--task", default="convnextv2_large_BRSET_multilabel_512")
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


class MultiLabelFocalLoss(nn.Module):
    """Per-label alpha-weighted binary focal loss (Lin et al., 2017 full formulation:
    focusing term gamma AND a per-class alpha/pos_weight term), averaged across labels
    then across the batch. The earlier version of this loss only had the focusing term;
    alpha was missing entirely, which combined with a high gamma (2.0) sharpened decision
    margins at the cost of probability calibration (high AUC, comparatively lower F1)."""

    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none")
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()


def build_sample_weights(labels_df):
    """Per-sample weight = the highest inverse-frequency among the positive
    labels present on that image (negative-only images get a baseline weight
    of 1). This oversamples images carrying a rare positive finding."""
    counts = {c: labels_df[c].sum() for c in LABEL_COLS}
    n = len(labels_df)
    inv_freq = {c: n / max(counts[c], 1) for c in LABEL_COLS}
    weights = []
    for _, row in labels_df.iterrows():
        w = 1.0
        for c in LABEL_COLS:
            if row[c] == 1:
                w = max(w, inv_freq[c])
        weights.append(w)
    return weights, counts


@torch.no_grad()
def run_inference(model, loader, device, tta=False):
    model.eval()
    all_targets, all_probs = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            probs = torch.sigmoid(logits.float())
            if tta:
                flipped = torch.flip(images, dims=[3])
                logits_f = model(flipped)
                probs_f = torch.sigmoid(logits_f.float())
                probs = (probs + probs_f) / 2
        all_targets.append(targets.numpy())
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_targets), np.concatenate(all_probs)


def tune_thresholds(y_true, y_prob):
    """Per-label threshold that maximizes F1 on the given (validation) set."""
    thresholds = []
    for i in range(y_true.shape[1]):
        best_t, best_f1 = 0.5, -1
        for t in np.arange(0.05, 0.95, 0.02):
            pred = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return thresholds


def compute_per_label_metrics(y_true, y_prob, thresholds):
    results = {}
    for i, col in enumerate(LABEL_COLS):
        pred = (y_prob[:, i] >= thresholds[i]).astype(int)
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            auc = float("nan")
        results[col] = {
            "threshold": thresholds[i],
            "auc": auc,
            "f1": f1_score(y_true[:, i], pred, zero_division=0),
            "precision": precision_score(y_true[:, i], pred, zero_division=0),
            "recall": recall_score(y_true[:, i], pred, zero_division=0),
            "accuracy": accuracy_score(y_true[:, i], pred),
            "confusion_matrix": confusion_matrix(y_true[:, i], pred, labels=[0, 1]).tolist(),
        }
    macro_auc = np.mean([results[c]["auc"] for c in LABEL_COLS])
    macro_f1 = np.mean([results[c]["f1"] for c in LABEL_COLS])
    return results, macro_auc, macro_f1


def mixup_batch(images, targets, alpha):
    """Multi-label mixup: blend a batch with a shuffled copy of itself, images
    and their multi-hot label vectors both linearly interpolated by the same
    lambda. Works directly with BCE-family losses since they accept soft/
    continuous targets in [0,1], no special-casing needed downstream."""
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[perm]
    mixed_targets = lam * targets + (1 - lam) * targets[perm]
    return mixed_images, mixed_targets


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, epoch, accum_iter,
                     mixup_alpha=0.0, label_smoothing=0.0, print_freq=20):
    model.train()
    running_loss, n_batches = 0.0, 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_alpha > 0:
            images, targets = mixup_batch(images, targets, mixup_alpha)
        if label_smoothing > 0:
            targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, targets) / accum_iter
        scaler.scale(loss).backward()
        if (i + 1) % accum_iter == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item() * accum_iter
        n_batches += 1
        if i % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch: [{epoch}]  [{i}/{len(loader)}]  lr: {lr:.7f}  loss: {loss.item()*accum_iter:.4f}  "
                  f"avg_loss: {running_loss/n_batches:.4f}  elapsed: {time.time()-t0:.1f}s", flush=True)
    scheduler.step()
    return running_loss / max(n_batches, 1)


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
    dataset_train = BRSETMultiLabel(data_path / "train", train_tf)
    dataset_val = BRSETMultiLabel(data_path / "val", eval_tf)
    dataset_test = BRSETMultiLabel(data_path / "test", eval_tf)
    print(f"train/val/test sizes: {len(dataset_train)}/{len(dataset_val)}/{len(dataset_test)}", flush=True)

    sample_weights, counts = build_sample_weights(dataset_train.labels_df)
    print(f"train label counts: {counts}", flush=True)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = timm.create_model(args.model, pretrained=True, num_classes=args.nb_classes,
                               drop_path_rate=args.drop_path)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_parameters: {n_params}", flush=True)

    pos_weight = None
    if args.focal_alpha_weighted:
        pos_weight_vals = []
        for col in LABEL_COLS:
            n_pos = counts[col]
            n_neg = len(dataset_train) - n_pos
            pos_weight_vals.append(n_neg / max(n_pos, 1))
        pos_weight = torch.tensor(pos_weight_vals, dtype=torch.float32, device=device)
        print(f"focal_alpha_weighted: pos_weight={dict(zip(LABEL_COLS, pos_weight_vals))}", flush=True)
    criterion = MultiLabelFocalLoss(gamma=args.focal_gamma, pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
    scaler = torch.amp.GradScaler("cuda")

    best_score, best_epoch = -1.0, -1
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, loader_train, optimizer, scheduler, scaler, criterion, device, epoch,
                                      args.accum_iter, mixup_alpha=args.mixup_alpha, label_smoothing=args.label_smoothing)
        y_true, y_prob = run_inference(model, loader_val, device, tta=False)
        thresholds = tune_thresholds(y_true, y_prob)
        per_label, macro_auc, macro_f1 = compute_per_label_metrics(y_true, y_prob, thresholds)
        score = (macro_auc + macro_f1) / 2
        print(f"val: macro_auc={macro_auc:.4f} macro_f1={macro_f1:.4f} score={score:.4f} "
              f"dr_auc={per_label['diabetic_retinopathy']['auc']:.4f} "
              f"me_auc={per_label['macular_edema']['auc']:.4f}", flush=True)

        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "macro_auc": macro_auc,
                                 "macro_f1": macro_f1, "score": score}) + "\n")

        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args), "thresholds": thresholds},
                       out_dir / "checkpoint-best.pth")
        print(f"Best epoch = {best_epoch}, Best score = {best_score:.4f}", flush=True)

        if args.ckpt_freq and ((epoch + 1) % args.ckpt_freq == 0 or epoch == args.epochs - 1):
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args), "thresholds": thresholds},
                       out_dir / f"checkpoint-epoch{epoch+1}.pth")
            print(f"Saved periodic checkpoint: checkpoint-epoch{epoch+1}.pth", flush=True)

    # ---- Final test with best checkpoint, threshold re-tuned on val, TTA at test time ----
    best_ckpt = torch.load(out_dir / "checkpoint-best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    model.to(device)
    print(f"Test with the best model, epoch = {best_ckpt['epoch']}:", flush=True)

    y_val_true, y_val_prob = run_inference(model, loader_val, device, tta=True)
    final_thresholds = tune_thresholds(y_val_true, y_val_prob)
    print(f"final per-label thresholds (tuned on val, TTA): {dict(zip(LABEL_COLS, final_thresholds))}", flush=True)

    y_test_true, y_test_prob = run_inference(model, loader_test, device, tta=True)
    per_label, macro_auc, macro_f1 = compute_per_label_metrics(y_test_true, y_test_prob, final_thresholds)

    print("TEST RESULTS (with TTA, tuned thresholds):", flush=True)
    for col in LABEL_COLS:
        m = per_label[col]
        print(f"  {col}: AUC={m['auc']:.4f} F1={m['f1']:.4f} precision={m['precision']:.4f} "
              f"recall={m['recall']:.4f} accuracy={m['accuracy']:.4f} threshold={m['threshold']:.2f} "
              f"cm={m['confusion_matrix']}", flush=True)
    print(f"  macro_auc={macro_auc:.4f} macro_f1={macro_f1:.4f}", flush=True)

    with open(out_dir / "metrics_test.json", "w") as f:
        json.dump({"per_label": per_label, "macro_auc": macro_auc, "macro_f1": macro_f1}, f, indent=2)

    with open(out_dir / "metrics_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "auc", "f1", "precision", "recall", "accuracy", "threshold"])
        for col in LABEL_COLS:
            m = per_label[col]
            w.writerow([col, m["auc"], m["f1"], m["precision"], m["recall"], m["accuracy"], m["threshold"]])
        w.writerow(["MACRO", macro_auc, macro_f1, "", "", "", ""])

    total_time = time.time() - start_time
    print(f"Training time {total_time/3600:.2f} hours", flush=True)


if __name__ == "__main__":
    main()
