"""
Phase 2: distill the DR-specific MoE gate into a single deployable model.

Context (see scripts 24-27): the BRSET-trained ensemble and the mBRSET-trained
model were combined via a small logistic gate (script 27). The RUS/PID
decomposition (script 26) predicted, and the gate confirmed, that combining
both experts genuinely helps for diabetic_retinopathy (real synergy, F1
0.815 -> 0.822) but not for macular_edema (near-zero synergy, gate F1 0.807
-> 0.727, worse). Running three networks at inference to get that DR gain is
wasteful, so this step folds the gate's DR behavior into one model via
knowledge distillation, instead of shipping the gate itself.

Design:
  - Warm-start from the mBRSET regularized checkpoint (already a strong,
    properly-regularized model) and continue fine-tuning at a low LR, rather
    than training from scratch.
  - Loss = the same multi-label focal loss on the true hard labels (both
    labels, as before) + a distillation term that pulls ONLY the DR logit
    toward the frozen gate's DR probability (computed once, offline, from the
    two frozen experts - not learned jointly here).
  - macular_edema gets no distillation term at all - the evidence says
    combining experts doesn't help it, so this run should not touch it.
  - No mixup here: mixup blends images/labels pairwise, which would also
    have to blend the precomputed per-image teacher targets in a way that's
    easy to get subtly wrong; skipped deliberately for this first attempt to
    keep the new distillation term easy to isolate and interpret.

The DR teacher probability for an image = sigmoid(w . [p_BRSET, p_mBRSET,
p_BRSET*p_mBRSET] + b) using the exact coefficients already fit in script 27
(results/moe_gate_results.json) - frozen constants, not re-fit here.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "train_mod", str(Path(__file__).parent / "07_train_convnextv2_multilabel.py"))
train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_mod)

BRSETMultiLabel = train_mod.BRSETMultiLabel
build_transforms = train_mod.build_transforms
MultiLabelFocalLoss = train_mod.MultiLabelFocalLoss
build_sample_weights = train_mod.build_sample_weights
run_inference = train_mod.run_inference
tune_thresholds = train_mod.tune_thresholds
compute_per_label_metrics = train_mod.compute_per_label_metrics
LABEL_COLS = train_mod.LABEL_COLS

RESULTS_DIR = Path("/home/users/sthummala2/brset-convnextv2/results")
MBRSET_DATA_PATH = "/home/users/sthummala2/brset-convnextv2/data/finetune_mbrset_multilabel"

with open(RESULTS_DIR / "moe_gate_results.json") as f:
    GATE = json.load(f)["diabetic_retinopathy"]
GATE_W = np.array(GATE["gate_coefficients"], dtype=np.float64)
GATE_B = float(GATE["gate_intercept"])


def gate_dr_prob(p_brset, p_mbrset):
    z = GATE_W[0] * p_brset + GATE_W[1] * p_mbrset + GATE_W[2] * (p_brset * p_mbrset) + GATE_B
    return 1.0 / (1.0 + np.exp(-z))


def load_model(ckpt_path, device, drop_path=0.0):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    model = timm.create_model(args["model"], pretrained=False, num_classes=args["nb_classes"],
                               drop_path_rate=drop_path)
    model.load_state_dict(ckpt["model"])
    return model, args


class DistillWrapper(Dataset):
    """Wraps BRSETMultiLabel, adding a precomputed DR teacher probability per index
    (same row order as the underlying labels.csv, so indices line up exactly)."""

    def __init__(self, base_dataset, teacher_dr_prob):
        assert len(base_dataset) == len(teacher_dr_prob)
        self.base = base_dataset
        self.teacher_dr_prob = teacher_dr_prob.astype(np.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]
        return img, target, self.teacher_dr_prob[idx]


def compute_teacher_probs(split, device):
    """Fresh TTA inference of the frozen BRSET ensemble + frozen mBRSET expert
    on the given mBRSET split, combined via the frozen DR gate. Cached to disk
    since this is expensive GPU inference that never needs to be redone unless
    the underlying experts change."""
    cache_path = RESULTS_DIR / f"distill_teacher_probs_{split}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        print(f"loaded cached teacher probs for {split} from {cache_path}", flush=True)
        return d["y_true"], d["teacher_dr_prob"], d["mbrset_prob"]

    model_a, args_a = load_model(RESULTS_DIR / "convnextv2_large_BRSET_multilabel_512/checkpoint-best.pth", device)
    model_a.to(device).eval()
    class A: pass
    a = A(); a.resize_size, a.input_size = args_a["resize_size"], args_a["input_size"]
    _, eval_tf = build_transforms(a)
    ds = BRSETMultiLabel(Path(MBRSET_DATA_PATH) / split, eval_tf)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    print(f"[{split}] BRSET original checkpoint TTA inference...", flush=True)
    y_true, prob_a = run_inference(model_a, loader, device, tta=True)
    del model_a; torch.cuda.empty_cache()

    model_b, _ = load_model(RESULTS_DIR / "convnextv2_large_BRSET_multilabel_512_regularized/checkpoint-best.pth", device)
    model_b.to(device).eval()
    print(f"[{split}] BRSET regularized checkpoint TTA inference...", flush=True)
    _, prob_b = run_inference(model_b, loader, device, tta=True)
    del model_b; torch.cuda.empty_cache()
    prob_brset = (prob_a + prob_b) / 2

    model_m, args_m = load_model(RESULTS_DIR / "convnextv2_large_mBRSET_multilabel_512_regularized/checkpoint-best.pth", device)
    model_m.to(device).eval()
    class M: pass
    m = M(); m.resize_size, m.input_size = args_m["resize_size"], args_m["input_size"]
    _, eval_tf_m = build_transforms(m)
    ds_m = BRSETMultiLabel(Path(MBRSET_DATA_PATH) / split, eval_tf_m)
    loader_m = DataLoader(ds_m, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    print(f"[{split}] mBRSET regularized (expert) TTA inference...", flush=True)
    y_true_m, prob_mbrset = run_inference(model_m, loader_m, device, tta=True)
    del model_m; torch.cuda.empty_cache()
    assert np.array_equal(y_true, y_true_m)

    dr_col = LABEL_COLS.index("diabetic_retinopathy")
    teacher_dr_prob = gate_dr_prob(prob_brset[:, dr_col], prob_mbrset[:, dr_col])

    np.savez(cache_path, y_true=y_true, teacher_dr_prob=teacher_dr_prob, mbrset_prob=prob_mbrset)
    print(f"wrote {cache_path}", flush=True)
    return y_true, teacher_dr_prob, prob_mbrset


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="convnextv2_large.fcmae_ft_in22k_in1k_384")
    p.add_argument("--warm_start_ckpt", default=str(RESULTS_DIR / "convnextv2_large_mBRSET_multilabel_512_regularized/checkpoint-best.pth"))
    p.add_argument("--input_size", default=512, type=int)
    p.add_argument("--resize_size", default=560, type=int)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--accum_iter", default=4, type=int)
    p.add_argument("--epochs", default=15, type=int)
    p.add_argument("--warmup_epochs", default=1, type=int)
    p.add_argument("--lr", default=1e-5, type=float)
    p.add_argument("--weight_decay", default=0.1, type=float)
    p.add_argument("--drop_path", default=0.3, type=float)
    p.add_argument("--label_smoothing", default=0.1, type=float)
    p.add_argument("--focal_gamma", default=2.0, type=float)
    p.add_argument("--kd_weight", default=1.0, type=float, help="Weight on the DR-only distillation term.")
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--ckpt_freq", default=5, type=int)
    p.add_argument("--debug_max_train", default=0, type=int,
                    help="If >0, truncate the training set to this many images - smoke-test only.")
    p.add_argument("--task", default="convnextv2_large_mBRSET_multilabel_512_distilled_dr")
    p.add_argument("--output_dir", default=str(RESULTS_DIR))
    p.add_argument("--seed", default=0, type=int)
    return p.parse_args()


def train_one_epoch_distill(model, loader, optimizer, scheduler, scaler, criterion, device, epoch,
                             accum_iter, kd_weight, label_smoothing, print_freq=20):
    model.train()
    running_loss, running_kd, n_batches = 0.0, 0.0, 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    dr_idx = LABEL_COLS.index("diabetic_retinopathy")
    for i, (images, targets, teacher_dr) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        teacher_dr = teacher_dr.to(device, non_blocking=True)
        if label_smoothing > 0:
            targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            hard_loss = criterion(logits, targets)
            kd_loss = nn.functional.binary_cross_entropy_with_logits(
                logits[:, dr_idx].float(), teacher_dr.float())
            loss = (hard_loss + kd_weight * kd_loss) / accum_iter
        scaler.scale(loss).backward()
        if (i + 1) % accum_iter == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item() * accum_iter
        running_kd += kd_loss.item()
        n_batches += 1
        if i % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch: [{epoch}]  [{i}/{len(loader)}]  lr: {lr:.7f}  loss: {loss.item()*accum_iter:.4f}  "
                  f"kd_loss: {kd_loss.item():.4f}  avg_loss: {running_loss/n_batches:.4f}  "
                  f"elapsed: {time.time()-t0:.1f}s", flush=True)
    scheduler.step()
    return running_loss / max(n_batches, 1), running_kd / max(n_batches, 1)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"
    print(f"Namespace: {vars(args)}", flush=True)

    y_train_true, teacher_dr_train, _ = compute_teacher_probs("train", device)
    y_val_true, teacher_dr_val, _ = compute_teacher_probs("val", device)

    class TA: pass
    ta = TA(); ta.resize_size, ta.input_size = args.resize_size, args.input_size
    train_tf, eval_tf = build_transforms(ta)

    base_train = BRSETMultiLabel(Path(MBRSET_DATA_PATH) / "train", train_tf)
    base_val = BRSETMultiLabel(Path(MBRSET_DATA_PATH) / "val", eval_tf)
    base_test = BRSETMultiLabel(Path(MBRSET_DATA_PATH) / "test", eval_tf)
    assert np.array_equal(base_train.labels_df[LABEL_COLS].values, y_train_true)
    assert np.array_equal(base_val.labels_df[LABEL_COLS].values, y_val_true)

    if args.debug_max_train > 0:
        base_train.labels_df = base_train.labels_df.iloc[:args.debug_max_train].reset_index(drop=True)
        teacher_dr_train = teacher_dr_train[:args.debug_max_train]
        print(f"DEBUG: truncated train set to {len(base_train)} images", flush=True)

    dataset_train = DistillWrapper(base_train, teacher_dr_train)
    dataset_val = base_val
    dataset_test = base_test
    print(f"train/val/test sizes: {len(dataset_train)}/{len(dataset_val)}/{len(dataset_test)}", flush=True)

    sample_weights, counts = build_sample_weights(base_train.labels_df)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model, _ = load_model(args.warm_start_ckpt, device, drop_path=args.drop_path)
    model.to(device)
    print(f"warm-started from {args.warm_start_ckpt}", flush=True)

    criterion = MultiLabelFocalLoss(gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
    scaler = torch.amp.GradScaler("cuda")

    best_score, best_epoch = -1.0, -1
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, kd_loss = train_one_epoch_distill(
            model, loader_train, optimizer, scheduler, scaler, criterion, device, epoch,
            args.accum_iter, args.kd_weight, args.label_smoothing)
        y_true, y_prob = run_inference(model, loader_val, device, tta=False)
        thresholds = tune_thresholds(y_true, y_prob)
        per_label, macro_auc, macro_f1 = compute_per_label_metrics(y_true, y_prob, thresholds)
        score = (macro_auc + macro_f1) / 2
        print(f"val: macro_auc={macro_auc:.4f} macro_f1={macro_f1:.4f} score={score:.4f} "
              f"dr_f1={per_label['diabetic_retinopathy']['f1']:.4f} "
              f"me_f1={per_label['macular_edema']['f1']:.4f} kd_loss={kd_loss:.4f}", flush=True)

        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "kd_loss": kd_loss,
                                 "macro_auc": macro_auc, "macro_f1": macro_f1, "score": score}) + "\n")

        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args), "thresholds": thresholds},
                       out_dir / "checkpoint-best.pth")
        print(f"Best epoch = {best_epoch}, Best score = {best_score:.4f}", flush=True)

        if args.ckpt_freq and ((epoch + 1) % args.ckpt_freq == 0 or epoch == args.epochs - 1):
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args), "thresholds": thresholds},
                       out_dir / f"checkpoint-epoch{epoch+1}.pth")

    if args.debug_max_train > 0:
        print("DEBUG smoke test complete, skipping final test evaluation.", flush=True)
        return

    best_ckpt = torch.load(out_dir / "checkpoint-best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    model.to(device)
    print(f"Test with the best model, epoch = {best_ckpt['epoch']}:", flush=True)

    y_val_true2, y_val_prob = run_inference(model, loader_val, device, tta=True)
    final_thresholds = tune_thresholds(y_val_true2, y_val_prob)
    print(f"final per-label thresholds (tuned on val, TTA): {dict(zip(LABEL_COLS, final_thresholds))}", flush=True)

    y_test_true, y_test_prob = run_inference(model, loader_test, device, tta=True)
    per_label, macro_auc, macro_f1 = compute_per_label_metrics(y_test_true, y_test_prob, final_thresholds)

    print("TEST RESULTS (distilled student, with TTA, tuned thresholds):", flush=True)
    for col in LABEL_COLS:
        m = per_label[col]
        print(f"  {col}: AUC={m['auc']:.4f} F1={m['f1']:.4f} precision={m['precision']:.4f} "
              f"recall={m['recall']:.4f} accuracy={m['accuracy']:.4f} threshold={m['threshold']:.2f} "
              f"cm={m['confusion_matrix']}", flush=True)
    print(f"  macro_auc={macro_auc:.4f} macro_f1={macro_f1:.4f}", flush=True)

    with open(out_dir / "metrics_test.json", "w") as f:
        json.dump({"per_label": per_label, "macro_auc": macro_auc, "macro_f1": macro_f1}, f, indent=2)
    np.savez(out_dir / "test_predictions.npz", y_true=y_test_true, y_prob=y_test_prob, thresholds=final_thresholds)

    total_time = time.time() - start_time
    print(f"Training time {total_time/3600:.2f} hours", flush=True)


if __name__ == "__main__":
    main()
