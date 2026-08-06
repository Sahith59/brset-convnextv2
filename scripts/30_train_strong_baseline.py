"""
Single strong ConvNeXt V2 baseline on full BRSET: one model, two independent
binary outputs (diabetic_retinopathy, macular_edema).

Per Dr. Ye's feedback, this replaces the earlier multi-approach comparison with
ONE model, and targets the F1 shortfall directly. BRSET is severely imbalanced
(train: 6.6% DR-positive, 2.4% ME-positive), which is what holds F1 down while
AUC stays high, so the imbalance handling is the substance of this run:

  1. Asymmetric Loss (Ridnik et al., ICCV 2021) instead of symmetric focal loss.
     ASL is built for exactly this multi-label rare-positive regime: it decouples
     the focusing exponent for positives vs. negatives (gamma_neg > gamma_pos, so
     easy negatives are discounted hard while rare positives keep full gradient)
     and adds a probability margin that discards near-certain negatives entirely.
     Symmetric focal loss (the previous gamma=2.0 for both) down-weights rare
     positives just as aggressively as the abundant negatives -- the opposite of
     what this label distribution needs.
  2. Weighted oversampling of images carrying a rare positive finding (kept).
  3. EMA of model weights -- one deployable model, no ensembling.
  4. 4-way flip TTA at evaluation.
  5. Bootstrap-smoothed per-label thresholds. With only 159 DR / 65 ME positives
     in validation, a single argmax-F1 threshold is fit to noise; averaging the
     F1-optimal threshold over resamples gives an operating point that holds up
     on test.
"""
import argparse
import copy
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


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification (Ridnik et al., ICCV 2021).

    Two mechanisms, both aimed at the rare-positive problem:
      - Decoupled focusing: (1-p)^gamma_pos on positives, p^gamma_neg on
        negatives, with gamma_neg > gamma_pos. Symmetric focal loss uses one
        gamma for both and therefore suppresses the rare positives it should be
        protecting.
      - Probability shifting: negatives with p < clip contribute exactly zero
        loss, fully discarding the easiest negatives rather than merely
        down-weighting them.
    """

    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        pt = xs_pos * targets + xs_neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        loss = loss * torch.pow(1.0 - pt, gamma)
        return -loss.mean()


class MultiLabelFocalLoss(nn.Module):
    """Symmetric focal loss, kept so --loss focal reproduces the previous runs."""

    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none")
        pt = torch.exp(-bce)
        return (((1 - pt) ** self.gamma) * bce).mean()


class ModelEma:
    """Exponential moving average of weights. Yields one model, not an ensemble."""

    def __init__(self, model, decay=0.9998):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_v, model_v in zip(self.module.state_dict().values(),
                                   model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v.detach(), alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="convnextv2_large.fcmae_ft_in22k_in1k_384")
    p.add_argument("--data_path", default="/home/users/sthummala2/brset-convnextv2/data/finetune_multilabel")
    p.add_argument("--nb_classes", default=2, type=int)
    p.add_argument("--input_size", default=512, type=int)
    p.add_argument("--resize_size", default=560, type=int)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--accum_iter", default=4, type=int)
    p.add_argument("--epochs", default=40, type=int)
    p.add_argument("--warmup_epochs", default=3, type=int)
    p.add_argument("--lr", default=5e-5, type=float)
    p.add_argument("--weight_decay", default=0.1, type=float)
    p.add_argument("--drop_path", default=0.3, type=float)
    p.add_argument("--mixup_alpha", default=0.2, type=float)
    p.add_argument("--label_smoothing", default=0.1, type=float)
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--ckpt_freq", default=10, type=int)
    p.add_argument("--loss", default="asl", choices=["asl", "focal"])
    p.add_argument("--asl_gamma_neg", default=4.0, type=float)
    p.add_argument("--asl_gamma_pos", default=0.0, type=float)
    p.add_argument("--asl_clip", default=0.05, type=float)
    p.add_argument("--focal_gamma", default=2.0, type=float)
    p.add_argument("--ema_decay", default=0.9998, type=float)
    p.add_argument("--tta", default="flip4", choices=["none", "hflip", "flip4"])
    p.add_argument("--thr_bootstrap", default=200, type=int,
                    help="Resamples for bootstrap-smoothed threshold selection (0 = plain argmax-F1)")
    p.add_argument("--task", default="convnextv2_large_BRSET_strong_baseline")
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


def build_sample_weights(labels_df):
    counts = {c: int(labels_df[c].sum()) for c in LABEL_COLS}
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
def run_inference(model, loader, device, tta="none"):
    model.eval()
    all_targets, all_probs = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            views = [images]
            if tta in ("hflip", "flip4"):
                views.append(torch.flip(images, dims=[3]))
            if tta == "flip4":
                views.append(torch.flip(images, dims=[2]))
                views.append(torch.flip(images, dims=[2, 3]))
            probs = torch.stack([torch.sigmoid(model(v).float()) for v in views]).mean(0)
        all_targets.append(targets.numpy())
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_targets), np.concatenate(all_probs)


def _best_f1_threshold(y_true_col, y_prob_col, grid):
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true_col, (y_prob_col >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def tune_thresholds(y_true, y_prob, n_boot=0, seed=0):
    """Per-label F1-optimal threshold. With n_boot>0, average the optimal
    threshold across bootstrap resamples so it is not fit to the particular
    noise of a validation set with very few positives."""
    grid = np.arange(0.05, 0.95, 0.01)
    rng = np.random.default_rng(seed)
    thresholds = []
    for i in range(y_true.shape[1]):
        if n_boot and n_boot > 0:
            ts = []
            n = len(y_true)
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                if y_true[idx, i].sum() < 2:
                    continue
                ts.append(_best_f1_threshold(y_true[idx, i], y_prob[idx, i], grid))
            thresholds.append(float(np.median(ts)) if ts else
                              float(_best_f1_threshold(y_true[:, i], y_prob[:, i], grid)))
        else:
            thresholds.append(float(_best_f1_threshold(y_true[:, i], y_prob[:, i], grid)))
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
    macro_auc = float(np.mean([results[c]["auc"] for c in LABEL_COLS]))
    macro_f1 = float(np.mean([results[c]["f1"] for c in LABEL_COLS]))
    return results, macro_auc, macro_f1


def mixup_batch(images, targets, alpha):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1 - lam) * images[perm], lam * targets + (1 - lam) * targets[perm]


def train_one_epoch(model, ema, loader, optimizer, scheduler, scaler, criterion, device,
                     epoch, args, print_freq=40):
    model.train()
    running_loss, n_batches = 0.0, 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if args.mixup_alpha > 0:
            images, targets = mixup_batch(images, targets, args.mixup_alpha)
        if args.label_smoothing > 0:
            targets = targets * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(images), targets) / args.accum_iter
        scaler.scale(loss).backward()
        if (i + 1) % args.accum_iter == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        running_loss += loss.item() * args.accum_iter
        n_batches += 1
        if i % print_freq == 0:
            print(f"Epoch: [{epoch}]  [{i}/{len(loader)}]  lr: {optimizer.param_groups[0]['lr']:.7f}  "
                  f"loss: {loss.item()*args.accum_iter:.4f}  avg: {running_loss/n_batches:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    scheduler.step()
    return running_loss / max(n_batches, 1)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"
    print(f"Namespace: {vars(args)}", flush=True)

    train_tf, eval_tf = build_transforms(args)
    data_path = Path(args.data_path)
    ds_train = BRSETMultiLabel(data_path / "train", train_tf)
    ds_val = BRSETMultiLabel(data_path / "val", eval_tf)
    ds_test = BRSETMultiLabel(data_path / "test", eval_tf)
    print(f"train/val/test sizes: {len(ds_train)}/{len(ds_val)}/{len(ds_test)}", flush=True)

    sample_weights, counts = build_sample_weights(ds_train.labels_df)
    print(f"train label counts: {counts} (of {len(ds_train)})", flush=True)
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True)

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = timm.create_model(args.model, pretrained=True, num_classes=args.nb_classes,
                               drop_path_rate=args.drop_path).to(device)
    print(f"n_parameters: {sum(p.numel() for p in model.parameters())}", flush=True)

    if args.loss == "asl":
        criterion = AsymmetricLoss(gamma_neg=args.asl_gamma_neg,
                                    gamma_pos=args.asl_gamma_pos, clip=args.asl_clip)
        print(f"criterion = AsymmetricLoss(gamma_neg={args.asl_gamma_neg}, "
              f"gamma_pos={args.asl_gamma_pos}, clip={args.asl_clip})", flush=True)
    else:
        criterion = MultiLabelFocalLoss(gamma=args.focal_gamma)
        print(f"criterion = MultiLabelFocalLoss(gamma={args.focal_gamma})", flush=True)

    ema = ModelEma(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine],
                                                       milestones=[args.warmup_epochs])
    scaler = torch.amp.GradScaler("cuda")

    best_score, best_epoch, best_variant = -1.0, -1, "raw"
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, ema, loader_train, optimizer, scheduler,
                                      scaler, criterion, device, epoch, args)

        epoch_best = None
        for variant, net in (("raw", model), ("ema", ema.module if ema else None)):
            if net is None:
                continue
            y_true, y_prob = run_inference(net, loader_val, device, tta="none")
            thr = tune_thresholds(y_true, y_prob, n_boot=0)
            _, macro_auc, macro_f1 = compute_per_label_metrics(y_true, y_prob, thr)
            score = (macro_auc + macro_f1) / 2
            print(f"val[{variant}]: macro_auc={macro_auc:.4f} macro_f1={macro_f1:.4f} "
                  f"score={score:.4f}", flush=True)
            if epoch_best is None or score > epoch_best[0]:
                epoch_best = (score, variant, macro_auc, macro_f1, thr)

        score, variant, macro_auc, macro_f1, thr = epoch_best
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "variant": variant,
                                 "macro_auc": macro_auc, "macro_f1": macro_f1, "score": score}) + "\n")

        if score > best_score:
            best_score, best_epoch, best_variant = score, epoch, variant
            net = ema.module if (variant == "ema" and ema) else model
            torch.save({"model": net.state_dict(), "epoch": epoch, "variant": variant,
                        "args": vars(args), "thresholds": thr}, out_dir / "checkpoint-best.pth")
        print(f"Best epoch = {best_epoch} ({best_variant}), Best score = {best_score:.4f}", flush=True)

        if args.ckpt_freq and ((epoch + 1) % args.ckpt_freq == 0 or epoch == args.epochs - 1):
            net = ema.module if (variant == "ema" and ema) else model
            torch.save({"model": net.state_dict(), "epoch": epoch, "variant": variant,
                        "args": vars(args), "thresholds": thr}, out_dir / f"checkpoint-epoch{epoch+1}.pth")

    # ---- Final test: best checkpoint, TTA, bootstrap-smoothed thresholds from val ----
    ckpt = torch.load(out_dir / "checkpoint-best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"\nTest with best model: epoch {ckpt['epoch']} (variant={ckpt['variant']}), tta={args.tta}", flush=True)

    y_val_true, y_val_prob = run_inference(model, loader_val, device, tta=args.tta)
    final_thr = tune_thresholds(y_val_true, y_val_prob, n_boot=args.thr_bootstrap, seed=args.seed)
    print(f"thresholds (val, bootstrap={args.thr_bootstrap}): {dict(zip(LABEL_COLS, final_thr))}", flush=True)

    y_test_true, y_test_prob = run_inference(model, loader_test, device, tta=args.tta)
    per_label, macro_auc, macro_f1 = compute_per_label_metrics(y_test_true, y_test_prob, final_thr)

    print("TEST RESULTS:", flush=True)
    for col in LABEL_COLS:
        m = per_label[col]
        print(f"  {col}: AUC={m['auc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} thr={m['threshold']:.2f} cm={m['confusion_matrix']}", flush=True)
    print(f"  macro_auc={macro_auc:.4f} macro_f1={macro_f1:.4f}", flush=True)

    np.savez(out_dir / "test_predictions.npz", y_true=y_test_true, y_prob=y_test_prob,
             thresholds=np.array(final_thr))
    np.savez(out_dir / "val_predictions.npz", y_true=y_val_true, y_prob=y_val_prob)
    with open(out_dir / "metrics_test.json", "w") as f:
        json.dump({"per_label": per_label, "macro_auc": macro_auc, "macro_f1": macro_f1,
                   "best_epoch": ckpt["epoch"], "variant": ckpt["variant"]}, f, indent=2)
    with open(out_dir / "metrics_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "auc", "f1", "precision", "recall", "accuracy", "threshold"])
        for col in LABEL_COLS:
            m = per_label[col]
            w.writerow([col, m["auc"], m["f1"], m["precision"], m["recall"], m["accuracy"], m["threshold"]])
        w.writerow(["MACRO", macro_auc, macro_f1, "", "", "", ""])

    print(f"Training time {(time.time()-start_time)/3600:.2f} hours", flush=True)


if __name__ == "__main__":
    main()
