#!/usr/bin/env python
"""Unsupervised domain adaptation on UNLABELED mBRSET images.

WHAT THIS IS FOR
----------------
Every arm tried so far either ignores the 3,402 real mBRSET training images
entirely (zero-shot: DR AUC 0.9104, F1 0.7774) or consumes them WITH labels
(pooling: DR AUC 0.9463, F1 0.8365). Nobody has used them WITHOUT labels. In
the label-cost framing of the paper that still counts as zero target TRAINING
labels, so the headline survives, and the arm sits on the label-efficiency
curve at budget 0.

The method is the oldest, least exotic thing in the UDA literature: train
supervised on source, and in the same step push the pooled feature
distributions of a labelled source batch and an UNLABELED target batch
together.

  L = L_task(source, source labels) + lambda * L_align(f_source, f_target)

  --uda_loss coral   Deep CORAL (Sun & Saenko, ECCV 2016 workshops):
                     squared Frobenius distance between the two batches'
                     feature COVARIANCE matrices, scaled 1/(4 d^2).
  --uda_loss mmd     Maximum Mean Discrepancy with a multi-scale RBF kernel
                     (Gretton et al. 2012; the DAN form, Long et al. ICML 2015),
                     biased V-statistic so identical batches give exactly 0.

No discriminator, no gradient-reversal layer, no adversarial stability tricks.
Note that the adversarial form of this idea was already run on this project in
scripts/46_domain_separation.py, but on the AGGREGATED LABELLED set, where the
target labels dominate the objective. The untried regime is this one: target
images present, target labels absent.

Why CORAL and MMD can disagree here. The measured gap (research memory,
3 Sep 2026) is that mBRSET is noisier, higher-contrast and DIMMER -- a
first-order shift. Deep CORAL as published matches second-order statistics
ONLY and is blind to a pure mean shift; RBF-kernel MMD is sensitive to all
moments. --uda_coral_mean adds an explicit first-order term to CORAL if the
distinction has to be isolated.

A second practical problem the two criteria share: their raw scales are not
comparable (CORAL carries a 1/(4 d^2) factor; MMD is O(1)) and at batch 16 the
sample covariance of 1536-dimensional features is dominated by estimator
noise, so there is no way to guess a literal lambda without spending GPU
hours. --uda_scale_mode relative measures sup/align once, at the first step,
and folds it into the weight, so --uda_weight reads as "the alignment term
starts at this fraction of the supervised loss" for either criterion. The
measured scale is printed and logged so the run can be quoted as a literal
weight afterwards.

SECONDARY, off by default: confidence-thresholded self-training (--st_weight).
Pseudo-labels come from the EMA teacher's own predictions on unlabeled target
images, thresholded at --st_conf, ramped in from --st_start_epoch.

NO-LABEL-LEAK GUARANTEE (the single most important property here)
-----------------------------------------------------------------
1. The target CSV is read with pandas usecols=["file"]. The label columns are
   never parsed into memory, and the constructor asserts the loaded column
   list is exactly ["file"].
2. UnlabeledTargetImages.__getitem__ returns a bare image tensor, not a
   (image, target) tuple. There is no target to misuse downstream.
3. The target sampler is a uniform shuffle. The baseline's label-driven
   WeightedRandomSampler is NOT used for the target stream, because sampling
   frequencies derived from labels would themselves be a leak.
4. coral_loss / mmd_rbf_loss / pseudo_labels take NO label argument. Check
   their signatures: they cannot see a label even by accident.
5. The target split directory is refused if it is named val/ or test/, and
   the target file list is intersected against the val and test file lists
   (those CSVs are also read with usecols=["file"]); any overlap aborts the
   run. This stops alignment to the evaluation set, which would be far worse
   than a training-label leak.
6. Target-domain labels CAN still enter training through the pre-existing
   --target_budget flag, which is a different experiment (pooling). The banner
   prints the budget and warns loudly if it is not 0.
7. Honest disclosure, unchanged from every other arm in this project: the
   725-image mBRSET validation set IS labelled and IS used for checkpoint
   selection and threshold fitting. That is the protocol for the zero-shot,
   pooling and synthesis arms too, so the comparison is fair, but "zero target
   labels" means zero target TRAINING labels.

BASELINE PRESERVATION
---------------------
scripts/30_train_strong_baseline.py is NOT edited. This file imports it as a
module, reuses its dataset, losses, EMA, transforms, sampler, evaluation,
threshold fitting and checkpointing, and swaps in its own epoch loop only.
When the new flags are off, train_one_epoch() below immediately delegates to
the original t30.train_one_epoch with the identical argument list -- the same
code object the baseline arms execute. Nothing else runs: no target loader is
built, no hook is registered, and no additional draw is taken from the torch
or numpy RNG, so a seeded run is bitwise identical to the baseline.
"""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "t30", str(_HERE / "30_train_strong_baseline.py"))
t30 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t30)

LABEL_COLS = t30.LABEL_COLS
AMP_DTYPE = t30.AMP_DTYPE
_ORIG_TRAIN_ONE_EPOCH = t30.train_one_epoch


# --------------------------------------------------------------------------
# unlabeled target stream
# --------------------------------------------------------------------------
class UnlabeledTargetImages(Dataset):
    """Target-domain images with NO labels, by construction.

    Only the 'file' column of labels.csv is parsed (usecols=["file"]), so the
    label columns never reach memory, and __getitem__ returns a bare tensor so
    there is no second element that could be mistaken for a target.
    """

    def __init__(self, split_dir, transform, forbidden_files=()):
        self.split_dir = Path(split_dir)
        if self.split_dir.name in ("val", "test"):
            raise SystemExit(
                f"refusing to use {self.split_dir} as the unlabeled target set: "
                f"aligning to the evaluation split is a test-set leak")
        csv_path = self.split_dir / "labels.csv"
        if not csv_path.exists():
            raise SystemExit(f"no labels.csv under {self.split_dir}")
        df = pd.read_csv(csv_path, usecols=["file"])
        self.columns_loaded = list(df.columns)
        if self.columns_loaded != ["file"]:
            raise SystemExit(f"label-leak guard: loaded columns {self.columns_loaded}, "
                             f"expected exactly ['file']")
        for c in LABEL_COLS:
            if c in self.columns_loaded:
                raise SystemExit(f"label-leak guard: {c} was loaded")
        self.files = [str(x) for x in df["file"].tolist()]
        forbidden = set(forbidden_files)
        if forbidden:
            overlap = sorted(forbidden.intersection(self.files))
            if overlap:
                raise SystemExit(
                    f"refusing to continue: {len(overlap)} unlabeled target images also "
                    f"appear in the val/test splits, e.g. {overlap[:5]}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.split_dir / self.files[idx]
        # Same NFS retry as t30.BRSETMultiLabel: transient read failures killed
        # a run at epoch 6 once.
        last = None
        for attempt in range(5):
            try:
                return self.transform(Image.open(path).convert("RGB"))
            except (OSError, IOError) as e:
                last = e
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"failed to read {path} after 5 attempts: {last}")


def eval_split_filenames(data_path):
    """File names in val/ and test/, for the overlap guard. Labels not read."""
    names = []
    for split in ("val", "test"):
        csv_path = Path(data_path) / split / "labels.csv"
        if csv_path.exists():
            names.extend(str(x) for x in
                         pd.read_csv(csv_path, usecols=["file"])["file"].tolist())
    return names


class _Cycler:
    """Endless batches from a finite loader; re-iterates when exhausted."""

    def __init__(self, loader):
        self.loader = loader
        self._it = iter(loader)
        self.epochs = 0

    def next(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            self.epochs += 1
            return next(self._it)


# --------------------------------------------------------------------------
# alignment losses -- note: NO label argument anywhere
# --------------------------------------------------------------------------
def _cov(x):
    """Unbiased feature covariance of a (n, d) batch."""
    n = x.size(0)
    if n < 2:
        raise ValueError(f"covariance needs at least 2 samples, got {n}")
    xc = x - x.mean(dim=0, keepdim=True)
    return (xc.t() @ xc) / (n - 1)


def coral_loss(f_source, f_target, mean_weight=0.0):
    """Deep CORAL (Sun & Saenko, ECCV 2016 workshops).

    ||C_s - C_t||_F^2 / (4 d^2). mean_weight > 0 adds the first-order term
    ||mu_s - mu_t||^2 / d, which published Deep CORAL does NOT have; it is
    exposed because the measured BRSET/mBRSET gap includes a brightness shift
    that a covariance-only criterion is blind to.
    """
    d = f_source.size(1)
    diff = _cov(f_source) - _cov(f_target)
    loss = (diff * diff).sum() / (4.0 * d * d)
    if mean_weight:
        md = f_source.mean(dim=0) - f_target.mean(dim=0)
        loss = loss + mean_weight * (md * md).sum() / d
    return loss


def mmd_rbf_loss(f_source, f_target, n_kernels=5, kernel_mul=2.0, bandwidth=None):
    """MMD^2 with a multi-scale RBF kernel (the DAN form, Long et al. ICML 2015).

    Biased V-statistic (diagonals included), which is what makes the value
    exactly zero for two identical batches. The bandwidth is the mean
    off-diagonal squared distance of the pooled batch and is DETACHED, so the
    kernel scale is treated as a hyper-parameter of the step rather than
    something the encoder can game by shrinking its features.
    """
    n_s = f_source.size(0)
    total = torch.cat([f_source, f_target], dim=0)
    n = total.size(0)
    sq = _sq_dists(total, total)
    if bandwidth is None:
        bw = sq.detach().sum() / max(n * n - n, 1)
    else:
        bw = torch.as_tensor(float(bandwidth), dtype=sq.dtype, device=sq.device)
    bw = bw.clamp_min(1e-8) / (kernel_mul ** (n_kernels // 2))
    kernel = 0.0
    for i in range(n_kernels):
        kernel = kernel + torch.exp(-sq / (bw * (kernel_mul ** i)))
    xx = kernel[:n_s, :n_s].mean()
    yy = kernel[n_s:, n_s:].mean()
    xy = kernel[:n_s, n_s:].mean()
    return xx + yy - 2.0 * xy


def _sq_dists(a, b):
    aa = (a * a).sum(dim=1).unsqueeze(1)
    bb = (b * b).sum(dim=1).unsqueeze(0)
    return (aa + bb - 2.0 * (a @ b.t())).clamp_min(0.0)


def _maybe_norm(f, mode):
    if mode == "l2":
        return f / f.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return f


def alignment_loss(feats_source, feats_target, args):
    """Mean alignment loss over the tapped feature sites. Computed in fp32."""
    if args.uda_loss == "none" or not feats_source:
        return None
    terms = []
    for fs, ft in zip(feats_source, feats_target):
        fs = _maybe_norm(fs.float(), args.uda_feat_norm)
        ft = _maybe_norm(ft.float(), args.uda_feat_norm)
        if args.uda_loss == "coral":
            terms.append(coral_loss(fs, ft, mean_weight=args.uda_coral_mean))
        elif args.uda_loss == "mmd":
            terms.append(mmd_rbf_loss(fs, ft, n_kernels=args.uda_mmd_kernels,
                                      kernel_mul=args.uda_mmd_mul))
        else:
            raise SystemExit(f"unknown --uda_loss {args.uda_loss}")
    return terms[0] if len(terms) == 1 else sum(terms) / len(terms)


def uda_step_loss(sup_loss, align_loss, weight):
    """Combine the two losses. Returns the supervised loss OBJECT ITSELF when
    the weight is zero, so the autograd graph is provably the baseline's."""
    if align_loss is None or weight == 0.0:
        return sup_loss
    return sup_loss + weight * align_loss


# --------------------------------------------------------------------------
# non-invasive feature taps
# --------------------------------------------------------------------------
class _FeatureTap:
    """Captures a module's input (or pooled output) with a forward hook.

    A forward hook that returns None cannot alter the value the network
    computes, so the supervised forward is untouched: model(images) runs
    exactly the baseline's code and we read the pooled features off the side.
    """

    def __init__(self, module, mode):
        self.mode = mode
        self.buf = None
        if mode == "input":
            self.handle = module.register_forward_pre_hook(self._pre)
        else:
            self.handle = module.register_forward_hook(self._post)

    def _pre(self, module, inputs):
        self.buf = inputs[0]
        return None

    def _post(self, module, inputs, output):
        self.buf = output.mean(dim=(2, 3)) if output.dim() == 4 else output
        return None

    def take(self):
        buf, self.buf = self.buf, None
        if buf is None:
            raise RuntimeError("feature tap did not fire; the forward did not reach it")
        if buf.dim() != 2:
            raise RuntimeError(f"tapped features have shape {tuple(buf.shape)}, expected (B, D)")
        return buf

    def clear(self):
        self.buf = None

    def remove(self):
        self.handle.remove()


class AlignmentTaps:
    def __init__(self, model, spec):
        self.taps, self.names = [], []
        for token in [t.strip() for t in spec.split(",") if t.strip()]:
            if token == "prelogits":
                head = model.get_classifier()
                if not hasattr(head, "weight"):
                    raise SystemExit("model.get_classifier() is not a linear layer; "
                                     "cannot tap pre-logits features")
                self.taps.append(_FeatureTap(head, "input"))
            elif token.startswith("stage"):
                idx = int(token[len("stage"):])
                if not hasattr(model, "stages"):
                    raise SystemExit(f"model has no .stages, cannot tap {token}")
                self.taps.append(_FeatureTap(model.stages[idx], "output"))
            else:
                raise SystemExit(f"unknown --uda_taps token {token!r} "
                                 f"(expected prelogits or stageN)")
            self.names.append(token)
        if not self.taps:
            raise SystemExit("--uda_taps selected no feature site")

    def take(self):
        return [t.take() for t in self.taps]

    def clear(self):
        for t in self.taps:
            t.clear()

    def remove(self):
        for t in self.taps:
            t.remove()


# --------------------------------------------------------------------------
# self-training (secondary, OFF by default)
# --------------------------------------------------------------------------
@torch.no_grad()
def pseudo_labels(teacher_logits, conf):
    """Confidence-thresholded pseudo-labels from the model's OWN predictions.

    Takes logits and nothing else: there is no label argument, so no ground
    truth can enter. Returns (hard labels, keep mask) per element.
    """
    p = torch.sigmoid(teacher_logits.float())
    mask = (p >= conf) | (p <= 1.0 - conf)
    return (p >= 0.5).float(), mask


def self_training_loss(criterion, logits, hard_labels, keep):
    """Supervised loss on the confident pseudo-labels only.

    Both tensors are cast to fp32 on purpose. The target logits come out of a
    bf16 autocast region while the pseudo-labels are built in fp32, and
    binary_cross_entropy_with_logits refuses two different float dtypes in its
    backward ("Found dtype Float but expected BFloat16"). Without the cast this
    path would run fine in every fp32 CPU test and then die on the GPU at
    --st_start_epoch, eight hours into a run.
    """
    if not bool(keep.any()):
        return None
    return criterion(logits[keep].float(), hard_labels[keep].float())


def align_weight_at(args, epoch):
    w = float(args.uda_weight)
    r = int(args.uda_rampup_epochs)
    if r > 0 and epoch < r:
        w *= float(epoch + 1) / float(r)
    return w


def calibrate_scale(sup_loss, align_loss, args):
    """One-off calibration of lambda so it reads as a FRACTION of the supervised loss.

    The raw scale of these criteria is not comparable: Deep CORAL's published
    1/(4 d^2) normalisation makes its value tiny and d-dependent, while MMD is
    O(1) by construction. Worse, at batch 16 on 1536-dimensional features the
    sample covariance is mostly estimator noise, so the raw CORAL value is not
    even a stable quantity to pick a weight against. In 'relative' mode the
    ratio sup/align is measured ONCE, at the first step, and folded into the
    weight for the rest of training, so --uda_weight 0.05 means "the alignment
    term starts at 5 percent of the supervised loss" for either criterion.
    The measured ratio is printed and logged, so the run is reproducible as a
    literal weight afterwards.
    """
    if args.uda_scale_mode != "relative":
        return 1.0
    if "scale" not in _STATE:
        a = float(align_loss.detach().item())
        s = float(sup_loss.detach().item())
        ok = a > 0.0 and math.isfinite(a) and math.isfinite(s) and s > 0.0
        _STATE["scale"] = (s / a) if ok else 1.0
        print(f"[uda] scale calibration: sup={s:.6g} align={a:.6g} -> "
              f"scale={_STATE['scale']:.6g}; effective lambda = --uda_weight * scale "
              f"(* ramp)", flush=True)
        if not ok:
            print("[uda] WARNING: calibration failed (align <= 0 or non-finite); "
                  "scale left at 1.0", flush=True)
    return _STATE["scale"]


def st_weight_at(args, epoch):
    w = float(args.st_weight)
    if w <= 0.0 or epoch < int(args.st_start_epoch):
        return 0.0
    r = int(args.st_rampup_epochs)
    if r > 0:
        w *= min(1.0, float(epoch - int(args.st_start_epoch) + 1) / float(r))
    return w


# --------------------------------------------------------------------------
# epoch loop
# --------------------------------------------------------------------------
def uda_enabled(args):
    align_on = (float(getattr(args, "uda_weight", 0.0)) > 0.0
                and getattr(args, "uda_loss", "none") != "none")
    return align_on or float(getattr(args, "st_weight", 0.0)) > 0.0


_STATE = {}


def _build_target_stream(args, device):
    train_tf, eval_tf = t30.build_transforms(args)
    tf = train_tf if args.uda_target_tf == "train" else eval_tf
    if getattr(args, "translator_ckpt", ""):
        raise SystemExit(
            "--translator_ckpt with the UDA flags is not supported: with a translator "
            "attached t30 defers normalisation to the training loop, so the two streams "
            "would be normalised differently. Run them as separate arms.")
    forbidden = eval_split_filenames(args.data_path)
    ds = UnlabeledTargetImages(args.uda_target_path, tf, forbidden_files=forbidden)
    bs = int(args.uda_target_batch) or int(args.batch_size)
    loader = DataLoader(ds, batch_size=bs, shuffle=True,
                        num_workers=int(args.uda_num_workers), pin_memory=True,
                        drop_last=True)
    print(f"[uda] unlabeled target stream: {len(ds)} images from {args.uda_target_path}", flush=True)
    print(f"[uda]   columns parsed from labels.csv: {ds.columns_loaded}  "
          f"(labels never read)", flush=True)
    print(f"[uda]   transform={args.uda_target_tf}  batch={bs}  "
          f"uniform shuffle (no label-driven sampler)", flush=True)
    print(f"[uda]   checked against {len(forbidden)} val/test file names: no overlap", flush=True)
    return _Cycler(loader)


def reset_state():
    """Drop the cached target stream and remove the feature hooks.

    Only used by the unit tests, which run several configurations in one
    process. Training never calls it.
    """
    taps = _STATE.pop("taps", None)
    if taps is not None:
        taps.remove()
    _STATE.clear()


def train_one_epoch(model, ema, loader, optimizer, scheduler, scaler, criterion, device,
                    epoch, args, print_freq=40, translator=None):
    """Baseline epoch plus the unlabeled-target terms.

    With the UDA flags off this is the baseline's own function, called with the
    identical arguments. Nothing in this file executes beyond the guard.
    """
    if not uda_enabled(args):
        return _ORIG_TRAIN_ONE_EPOCH(model, ema, loader, optimizer, scheduler, scaler,
                                     criterion, device, epoch, args, print_freq=print_freq,
                                     translator=translator)
    if translator is not None:
        raise SystemExit("--translator_ckpt with the UDA flags is not supported")

    if "target" not in _STATE:
        _STATE["target"] = _build_target_stream(args, device)
        _STATE["taps"] = AlignmentTaps(model, args.uda_taps)
        print(f"[uda] taps: {_STATE['taps'].names}", flush=True)
    target_stream, taps = _STATE["target"], _STATE["taps"]

    w_align = align_weight_at(args, epoch)
    w_st = st_weight_at(args, epoch)
    print(f"[uda] epoch {epoch}: loss={args.uda_loss} lambda={w_align:.6g} "
          f"self_training_weight={w_st:.6g}", flush=True)

    model.train()
    running_loss, running_sup, running_align, running_st = 0.0, 0.0, 0.0, 0.0
    n_batches, n_bad, n_st_batches = 0, 0, 0
    conf_kept, conf_total, ps_pos = 0.0, 0.0, 0.0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        if len(batch) == 3:
            raise SystemExit("the source loader yielded translate flags; "
                             "translate_mode=union is not supported with the UDA flags")
        images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        t_images = target_stream.next().to(device, non_blocking=True)

        if args.mixup_alpha > 0:
            images, targets = t30.mixup_batch(images, targets, args.mixup_alpha)
        if args.label_smoothing > 0:
            targets = targets * (1 - args.label_smoothing) + 0.5 * args.label_smoothing

        taps.clear()
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE[args.amp_dtype],
                            enabled=args.amp_dtype != "fp32"):
            logits = model(images)
            f_src = taps.take()
            sup_loss = criterion(logits, targets)
            t_logits = model(t_images)
            f_tgt = taps.take()
            teacher_logits = None
            if w_st > 0.0:
                if args.st_teacher == "ema" and ema is not None:
                    with torch.no_grad():
                        teacher_logits = ema.module(t_images)
                else:
                    teacher_logits = t_logits.detach()

        # alignment in fp32, outside autocast: covariances and kernels in bf16
        # are not trustworthy
        align = alignment_loss(f_src, f_tgt, args) if w_align > 0.0 else None
        w_eff = w_align * (calibrate_scale(sup_loss, align, args)
                           if align is not None else 1.0)
        loss = uda_step_loss(sup_loss, align, w_eff)

        st_loss = None
        if teacher_logits is not None:
            y_ps, keep = pseudo_labels(teacher_logits, args.st_conf)
            conf_total += keep.numel()
            conf_kept += float(keep.sum().item())
            st_loss = self_training_loss(criterion, t_logits, y_ps, keep)
            if st_loss is not None:
                ps_pos += float(y_ps[keep].sum().item())
                loss = loss + w_st * st_loss
                n_st_batches += 1

        loss = loss / args.accum_iter
        if not torch.isfinite(loss):
            n_bad += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss).backward()
        if (i + 1) % args.accum_iter == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        running_loss += loss.item() * args.accum_iter
        running_sup += float(sup_loss.item())
        if align is not None:
            running_align += float(align.item())
        if st_loss is not None:
            running_st += float(st_loss.item())
        n_batches += 1
        if i % print_freq == 0:
            print(f"Epoch: [{epoch}]  [{i}/{len(loader)}]  lr: {optimizer.param_groups[0]['lr']:.7f}  "
                  f"loss: {loss.item()*args.accum_iter:.4f}  avg: {running_loss/n_batches:.4f}  "
                  f"sup: {running_sup/n_batches:.4f}  "
                  f"align: {running_align/max(n_batches,1):.6f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    scheduler.step()
    if n_bad:
        print(f"WARNING epoch {epoch}: skipped {n_bad} non-finite batches", flush=True)

    diag = {
        "epoch": epoch, "uda_loss": args.uda_loss, "lambda": w_align,
        "scale": _STATE.get("scale", 1.0),
        "lambda_effective": w_align * _STATE.get("scale", 1.0),
        "st_weight": w_st,
        "mean_total": running_loss / max(n_batches, 1),
        "mean_sup": running_sup / max(n_batches, 1),
        "mean_align": running_align / max(n_batches, 1),
        "mean_align_times_lambda": (w_align * _STATE.get("scale", 1.0)
                                    * running_align / max(n_batches, 1)),
        "mean_st": running_st / max(n_st_batches, 1),
        "st_confident_fraction": (conf_kept / conf_total) if conf_total else 0.0,
        "st_pseudo_positive_rate": (ps_pos / conf_kept) if conf_kept else 0.0,
        "target_passes": target_stream.epochs,
    }
    if torch.cuda.is_available():
        diag["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    print("[uda] " + json.dumps(diag), flush=True)
    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "uda_log.jsonl", "a") as f:
        f.write(json.dumps(diag) + "\n")
    return running_loss / max(n_batches, 1)


# --------------------------------------------------------------------------
# args: our flags, then everything else straight to the baseline parser
# --------------------------------------------------------------------------
def uda_parser():
    p = argparse.ArgumentParser(add_help=False,
                                description="UDA flags added on top of scripts/30")
    p.add_argument("--uda_target_path",
                   default="/home/users/sthummala2/brset-convnextv2/data/finetune_mbrset_multilabel/train",
                   help="Directory of UNLABELED target images. Only the 'file' column of its "
                        "labels.csv is read. val/ and test/ are refused.")
    p.add_argument("--uda_loss", default="none", choices=["none", "coral", "mmd"],
                   help="Feature-alignment criterion. 'none' plus --st_weight 0 reproduces the "
                        "baseline exactly.")
    p.add_argument("--uda_weight", default=0.0, type=float,
                   help="lambda on the alignment loss. CORAL's 1/(4 d^2) scaling makes its raw "
                        "value tiny (order 1e-4 on 1536-d features), so lambda is large for "
                        "CORAL and order 1 for MMD. The per-epoch log prints the raw mean and "
                        "lambda*mean so the ratio against the supervised loss is visible.")
    p.add_argument("--uda_scale_mode", default="absolute", choices=["absolute", "relative"],
                   help="'absolute' uses --uda_weight literally, as the published methods do. "
                        "'relative' multiplies it by sup/align measured at the first step, so "
                        "the weight reads as a fraction of the supervised loss and is "
                        "comparable between CORAL and MMD. The measured scale is logged.")
    p.add_argument("--uda_taps", default="prelogits",
                   help="Comma-separated feature sites: 'prelogits' (input of the classifier, the "
                        "Deep CORAL/DAN choice) and/or 'stageN' (global-average-pooled output of "
                        "ConvNeXt stage N, 0-3). Multiple sites are averaged. The measured gap is "
                        "low-level, so stage2,stage3 is a defensible alternative.")
    p.add_argument("--uda_target_batch", default=0, type=int,
                   help="Unlabeled target batch size. 0 means match --batch_size, which is what "
                        "balanced CORAL/MMD estimators want. Drop to 8 if the second forward OOMs.")
    p.add_argument("--uda_target_tf", default="train", choices=["train", "eval"],
                   help="Preprocessing for target images. 'train' uses the same augmentation as "
                        "the source stream; a mismatch would itself create a feature gap that "
                        "the alignment loss would then chase.")
    p.add_argument("--uda_num_workers", default=4, type=int)
    p.add_argument("--uda_rampup_epochs", default=0, type=int,
                   help="Linearly ramp lambda over the first N epochs (0 = constant).")
    p.add_argument("--uda_feat_norm", default="none", choices=["none", "l2"],
                   help="Optional L2 normalisation of pooled features before the loss.")
    p.add_argument("--uda_coral_mean", default=0.0, type=float,
                   help="Weight of an extra first-order (mean-matching) term in CORAL. 0.0 is "
                        "published Deep CORAL, which is blind to a pure mean shift.")
    p.add_argument("--uda_mmd_kernels", default=5, type=int)
    p.add_argument("--uda_mmd_mul", default=2.0, type=float)
    p.add_argument("--st_weight", default=0.0, type=float,
                   help="SECONDARY: weight on confidence-thresholded self-training. 0 = off.")
    p.add_argument("--st_conf", default=0.9, type=float,
                   help="Keep a pseudo-label when max(p, 1-p) >= this.")
    p.add_argument("--st_start_epoch", default=5, type=int)
    p.add_argument("--st_rampup_epochs", default=5, type=int)
    p.add_argument("--st_teacher", default="ema", choices=["ema", "raw"])
    p.add_argument("--uda_parse_only", action="store_true", default=False,
                   help="PREFLIGHT. Parse the whole command line, check the split layout and "
                        "build the unlabeled target index (CSV only, no image decoding, no "
                        "model, no GPU), print what would run and exit. Lets a launch line be "
                        "validated on a CPU allocation before a 12-hour GPU job is spent on a "
                        "typo. Prints DRY RUN so a log cannot be mistaken for a trained arm.")
    return p


def get_args():
    p = uda_parser()
    if "-h" in sys.argv or "--help" in sys.argv:
        p.print_help()
        print("\n--- flags below are passed straight to "
              "scripts/30_train_strong_baseline.py ---\n")
        t30.get_args()   # prints the baseline help and exits
    uda_args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = t30.get_args()
    for k, v in vars(uda_args).items():
        if hasattr(args, k):
            raise SystemExit(f"flag collision with the baseline parser: {k}")
        setattr(args, k, v)
    return args


def main():
    args = get_args()
    print("=" * 78, flush=True)
    print("UDA on UNLABELED target images (scripts/94_uda_align.py)", flush=True)
    print(f"  alignment      : {args.uda_loss} lambda={args.uda_weight} "
          f"({args.uda_scale_mode}) taps={args.uda_taps}", flush=True)
    print(f"  self-training  : weight={args.st_weight} conf={args.st_conf} "
          f"start_epoch={args.st_start_epoch} teacher={args.st_teacher}", flush=True)
    print(f"  uda active     : {uda_enabled(args)}", flush=True)
    if not uda_enabled(args):
        print("  -> flags off: delegating to the unmodified baseline epoch loop", flush=True)
    print(f"  LABELLED target images in train (--target_budget): {args.target_budget}", flush=True)
    if args.target_budget != 0:
        print("  WARNING: --target_budget != 0, so LABELLED target images are in the training "
              "set. This is no longer a zero-target-label arm; the unlabeled stream will also "
              "contain those same images.", flush=True)
    print("  mBRSET val (725 labelled images) is used for checkpoint selection and threshold "
          "fitting, as in every other arm.", flush=True)
    print("=" * 78, flush=True)

    if args.uda_parse_only:
        print("DRY RUN (--uda_parse_only): nothing is trained.", flush=True)
        print(f"Namespace: {vars(args)}", flush=True)
        for split in ("train", "val", "test"):
            csv_path = Path(args.data_path) / split / "labels.csv"
            n = len(pd.read_csv(csv_path, usecols=["file"])) if csv_path.exists() else -1
            print(f"  {csv_path}: {'MISSING' if n < 0 else str(n) + ' rows'}", flush=True)
            if n < 0:
                raise SystemExit(f"missing split: {csv_path}")
        if uda_enabled(args):
            tf, _ = t30.build_transforms(args)
            ds = UnlabeledTargetImages(
                args.uda_target_path, tf,
                forbidden_files=eval_split_filenames(args.data_path))
            print(f"  unlabeled target: {len(ds)} images, columns parsed {ds.columns_loaded}, "
                  f"no overlap with val/test", flush=True)
            bs = int(args.uda_target_batch) or int(args.batch_size)
            print(f"  target batch {bs}; one source epoch of 11372/{args.batch_size} = "
                  f"{11372 // args.batch_size} steps draws "
                  f"{(11372 // args.batch_size) * bs / len(ds):.2f} passes over it", flush=True)
        print("DRY RUN complete: command line and data layout are valid.", flush=True)
        return

    t30.get_args = lambda: args
    t30.train_one_epoch = train_one_epoch
    t30.main()


if __name__ == "__main__":
    main()
