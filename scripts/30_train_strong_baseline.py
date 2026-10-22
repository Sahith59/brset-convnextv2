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
import torch.nn.functional as F
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

# Ordinal auxiliary head. The binary DR label collapses the ICDR 0-4 scale, so a
# grade-1 eye and a grade-4 eye carry an identical target and the model learns a
# single "positive" prototype -- built, in BRSET, mostly from florid disease
# (43.8% of its positives are grade 3-4, against 25.9% in mBRSET). Measured
# consequence: zero-shot AUC 0.7735 at grade 1 versus 0.9963 at grade 4, with
# every missed patient at grade 1 or 2.
#
# These four extra outputs are the cumulative decomposition P(grade >= k), which
# keeps a monotone severity axis in the representation instead of letting it
# collapse. This fixes the REPRESENTATION; --severity_match fixes the
# DISTRIBUTION. They are independent and combinable.
#
# _ordm is a per-image mask: 1 where a grade is known, 0 where it is not.
ORD_COLS = ["_ord1", "_ord2", "_ord3", "_ord4", "_ordm"]
_EXTRA_COLS = []    # appended to the target tensor when the aux head is on
_MV = [None, 0.0]  # (cycling iterator over same-eye photo pairs, running consistency loss)
_NB_OUT = [None]    # eval slices logits and targets back to the scored columns
# --synth_dir: per-image probability of reading the SYNTHETIC mBRSET twin of a
# BRSET training image instead of the original. Set only on the training set.
_SYNTH_P = [0.0]
AMP_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_EVAL_DTYPE = [torch.float16]   # set from args in main()


class BRSETMultiLabel(Dataset):
    def __init__(self, split_dir, transform):
        self.split_dir = Path(split_dir)
        self.labels_df = pd.read_csv(self.split_dir / "labels.csv")
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        flag = float(row["_translate"]) if "_translate" in self.labels_df.columns else -1.0
        path = self.split_dir / row["file"]
        sp = row["_synth"] if "_synth" in self.labels_df.columns else ""
        if isinstance(sp, str) and sp and _SYNTH_P[0] > 0 and float(torch.rand(())) < _SYNTH_P[0]:
            opts = sp.split("|")
            path = Path(opts[int(torch.randint(len(opts), ()))] if len(opts) > 1 else sp)   # same eye, same labels, generator-rendered
        # The images live on NFS and reads occasionally fail transiently under
        # load -- a run died at epoch 6 with FileNotFoundError on a file that
        # was present and whose symlink was intact. Retry rather than lose
        # hours of training to a momentary blip.
        last = None
        for attempt in range(5):
            try:
                img = self.transform(Image.open(path).convert("RGB"))
                target = torch.tensor([row[c] for c in LABEL_COLS + _EXTRA_COLS],
                                       dtype=torch.float32)
                if flag >= 0.0:
                    return img, target, torch.tensor(flag, dtype=torch.float32)
                return img, target
            except (OSError, IOError) as e:
                last = e
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"failed to read {path} after 5 attempts: {last}")


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


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Loss for multi-label sigmoid outputs.

    Origin: Li, Kamnitsas & Glocker, "Overfitting of neural nets under class
    imbalance", MICCAI 2019 -> IEEE TMI. Their softmax form is

        L = sum_j [ -r_j y_ij log(p_ij) - (1-r_j)(1-p_ij)^gamma y_ij log(p_ij) ]

    where r_j flags the RARE class. The defining idea: the rare class gets
    plain cross-entropy with NO focal modulation, so its gradient is never
    suppressed; only the abundant class is down-weighted. Yeung et al.
    (Computerized Medical Imaging and Graphics, 2022) add a delta term
    balancing the two, recommending delta = 0.6.

    Written per-label for independent sigmoids, positives = the rare class:

        L = -delta * y * log(p)                       (no modulation on positives)
            -(1-delta) * (1-y) * p^gamma * log(1-p)   (focal modulation on negatives)

    Contrast with Asymmetric Loss (Ridnik et al., ICCV 2021), which has a
    tunable focusing exponent on positives as well and adds a probability
    margin. Asymmetric Focal Loss is the stricter statement: never discount a
    positive.
    """

    def __init__(self, gamma=2.0, delta=0.6, eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.delta = delta
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        pos = -targets * torch.log(p)                                   # plain CE
        neg = -(1.0 - targets) * (p ** self.gamma) * torch.log(1.0 - p)  # focal
        return (self.delta * pos + (1.0 - self.delta) * neg).mean()


class MultiLabelFocalLoss(nn.Module):
    """Symmetric focal loss, kept so --loss focal reproduces the previous runs."""

    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets, weight=None):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none")
        pt = torch.exp(-bce)
        l = ((1 - pt) ** self.gamma) * bce
        if weight is None:
            return l.mean()
        # per-label validity mask: an invalid label contributes no gradient
        return (l * weight).sum() / weight.sum().clamp(min=1.0)


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
    p.add_argument("--amp_dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"],
                    help="Autocast dtype. 'auto' picks bf16 on GPUs that support it. The focal "
                         "run afl40_focal_control diverged to NaN at epoch 26 under fp16: ConvNeXt "
                         "V2's GRN takes an L2 norm over a 512x512 map, which can exceed the fp16 "
                         "max of 65504. bf16 has fp32's exponent range and does not overflow.")
    p.add_argument("--resize_mode", default="squash", choices=["squash", "preserve"],
                    help="How images are brought to resize_size. 'squash' is Resize((S,S)), which "
                         "forces both dimensions and IGNORES aspect ratio. Measured 3 Sep 2026: every "
                         "BRSET image is non-square (mean aspect 1.287) and every mBRSET image is "
                         "exactly square, so squash compresses BRSET horizontally by about 23 percent "
                         "while leaving mBRSET untouched, creating a geometric domain gap in "
                         "preprocessing. 'preserve' resizes the short side and centre crops, treating "
                         "both datasets identically.")
    p.add_argument("--translator_ckpt", default="",
                    help="Path to a trained two-stage translator (scripts/83). When set, every "
                         "TRAINING image is passed through it on the GPU before normalisation, "
                         "turning BRSET images into synthetic mBRSET-like ones. Validation and "
                         "test are NEVER translated; they are real target images. Translation is "
                         "done on the fly rather than from a saved corpus because JPEG and "
                         "resampling destroy the sensor-noise component that carries most of the "
                         "per-channel sharpness match, and because the acquisition model's "
                         "stochastic terms then redraw every epoch instead of mapping the whole "
                         "source set onto a single point in appearance space.")
    p.add_argument("--translate_mode", default="prob", choices=["prob", "union"],
                    help="How synthesis is combined with real data. 'prob' translates each image "
                         "with probability --translate_p, so an epoch still contains one copy of "
                         "each image. 'union' is the advisor's request literally: the training "
                         "set becomes real BRSET PLUS synthetic mBRSET, every image present both "
                         "ways in every epoch, doubling the epoch. Measured 10 Oct, full "
                         "REPLACEMENT (prob at p=1.0) costs 0.0216 DR AUC against the "
                         "untranslated baseline, so the distinction is not cosmetic.")
    p.add_argument("--translate_p", default=1.0, type=float,
                    help="Probability that a given training image is translated. 1.0 REPLACES "
                         "every source image with its synthetic version; measured 10 Oct, that "
                         "costs 0.0216 DR AUC and 0.0496 F1 against the untranslated baseline, "
                         "2.5x the noise floor. Replacement means the model never sees a real "
                         "image and simply trades the BRSET-vs-mBRSET gap for a "
                         "synthetic-vs-real-mBRSET one. Values below 1.0 use synthesis as "
                         "AUGMENTATION alongside real images, which is what the prior work that "
                         "reports gains actually does.")
    p.add_argument("--severity_match", action="store_true", default=False,
                    help="Reweight SOURCE-domain positives so their ICDR severity mix matches the "
                         "target cohort's. Measured 10 Oct: among positives, BRSET is 14.6 percent "
                         "mild and 36.6 percent proliferative while mBRSET is 24.0 and 18.7. "
                         "Zero-shot AUC by grade is 0.7735 / 0.9390 / 0.9973 / 0.9963 for grades "
                         "1 to 4, and ALL 42 missed patients are grade 1 or 2 with none above. "
                         "The transfer gap therefore sits on a hidden ordinal axis that the binary "
                         "label collapses, which is orthogonal to image appearance (all 9 "
                         "appearance statistics correlate with grade at |rho| <= 0.11). Matching "
                         "the marginal prevalence, which the earlier control did, cannot fix this; "
                         "the conditional severity distribution is what differs.")
    p.add_argument("--severity_csv",
                    default="/data/users4/nshaik3/Datasets/BRSET/physionet.org/files/"
                            "brazilian-ophthalmological/1.0.1/labels_brset.csv")
    p.add_argument("--severity_target", default="0.240,0.501,0.072,0.187",
                    help="target ICDR mix for grades 1,2,3,4 among positives; default is mBRSET's")
    p.add_argument("--target_pos_rate", default=-1.0, type=float,
                    help="Resample so the EXPECTED fraction of DR-positive images drawn per epoch "
                         "equals this rate. -1 disables, reproducing every earlier run. This "
                         "isolates Dr. Ye's label-distribution hypothesis: BRSET train is 6.6 "
                         "percent DR-positive and mBRSET is 23.4 percent, so pooling more than "
                         "doubles the absolute positive count (748 to 1545). Synthetic target "
                         "images inherit BRSET's labels and therefore CANNOT reproduce that, so if "
                         "the pooling gain is mostly class balance the synthesis claim fails. "
                         "Setting this with --target_budget 0 is the decisive control.")
    p.add_argument("--target_budget", default=-1, type=int,
                    help="Number of REAL target-domain (domain==1) training images to keep. -1 "
                         "keeps all of them, which reproduces every earlier run. This is the "
                         "x-axis of the label-efficiency study: how many real handheld-camera "
                         "labels are needed, with and without synthesised ones. Stratified on DR "
                         "and nested across budgets for a fixed --budget_seed.")
    p.add_argument("--budget_seed", default=0, type=int,
                    help="Which subsample of the target domain to draw. Separate from --seed so "
                         "the effect of WHICH images were labelled can be separated from the "
                         "effect of training randomness.")
    p.add_argument("--extra_manifest", default="",
                    help="CSV of EXTERNAL graded images (file,grade) from "
                         "102_build_extra_manifest.py. Imports real mild disease "
                         "that BRSET lacks: ~115 grade-1 training images is the "
                         "measured bottleneck, and reweighting cannot create "
                         "diversity that is not there.")
    p.add_argument("--pseudo_manifest", default="",
                    help="CSV (file, diabetic_retinopathy, macular_edema, _mdr, _med) of UNLABELLED "
                         "target images with TEACHER pseudo-labels and per-label validity masks, "
                         "from scripts/121_pseudo_manifest.py. Real target labels are never read: "
                         "the rows come from a teacher's scores, not from labels.csv.")
    p.add_argument("--synth_all", action="store_true",
                    help="Swap EVERY training row (both domains) for its synthetic twin, not only BRSET rows "
                         "(D083: train on synthetic images only, from BRSET and mBRSET).")
    p.add_argument("--mv_weight", default=0.0, type=float,
                    help="Same-eye consistency on UNLABELLED target photos (research/codex D080): two photos "
                         "of one mBRSET eye, augmented independently, must get the same predicted "
                         "probabilities (mean squared difference). 0 = off. No target label is read.")
    p.add_argument("--mv_batch", default=4, type=int, help="eyes (photo pairs) per micro-step")
    p.add_argument("--extra_n", default=1000, type=int)
    p.add_argument("--extra_severity", default="mild",
                    choices=["mild", "severe", "mixed"],
                    help="Which external POSITIVES to import. All arms add the "
                         "same count, so severity is the only thing that varies. "
                         "'severe' is the control; 'mixed' matters because the "
                         "oracle test found a random severity mix beat both "
                         "extremes at a 250-label budget.")
    p.add_argument("--extra_seed", default=0, type=int)
    p.add_argument("--synth_dir", default="",
                    help="Directory of synthetic mBRSET PNGs named <BRSET stem>.png "
                         "(scripts/111_translate.py). Each BRSET TRAINING image is replaced "
                         "by its synthetic twin with probability --synth_p, redrawn every "
                         "time it is loaded. Step count and eye identities are unchanged, "
                         "so the comparison with the same run without --synth_dir isolates "
                         "the camera rendering. Val/test are never touched.")
    p.add_argument("--synth_p", default=0.5, type=float)
    p.add_argument("--budget_severity", default="random",
                    choices=["random", "mild", "severe"],
                    help="WHICH target labels to spend the budget on, holding the "
                         "budget and the positive rate fixed. 'mild' draws the "
                         "positives from ICDR 1-2 only, 'severe' from 3-4 only. "
                         "The severe arm is the control: without it, a win for "
                         "'mild' is only a better subset, not evidence that "
                         "severity is the mechanism.")
    p.add_argument("--warm_start_ckpt", default="",
                    help="Initialize weights from this checkpoint instead of ImageNet. Used to "
                         "fine-tune a BRSET-trained model on mBRSET. The classifier head is kept "
                         "since both datasets share the same two labels.")
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
    p.add_argument("--no_oversample", action="store_true", default=False,
                    help="Disable weighted oversampling. The sampler lifts the effective positive "
                         "rate the loss sees from 6.6%% to 61%% (DR) and 2.4%% to 38.5%% (ME), so an "
                         "asymmetric loss tuned for a skewed distribution is applied to data that "
                         "has already been rebalanced past parity. See Bag of Tricks (AAAI 2021) and "
                         "Distribution-Balanced Loss (ECCV 2020) on combining re-sampling with "
                         "re-weighting.")
    p.add_argument("--loss", default="afl", choices=["afl", "asl", "focal"])
    p.add_argument("--afl_gamma", default=2.0, type=float,
                    help="Asymmetric Focal Loss: focal exponent applied to NEGATIVES only.")
    p.add_argument("--afl_delta", default=0.6, type=float,
                    help="Asymmetric Focal Loss: weight on the positive term "
                         "(Yeung et al. recommend 0.6).")
    p.add_argument("--asl_gamma_neg", default=4.0, type=float)
    p.add_argument("--asl_gamma_pos", default=0.0, type=float)
    p.add_argument("--asl_clip", default=0.05, type=float)
    p.add_argument("--focal_gamma", default=2.0, type=float)
    p.add_argument("--ema_decay", default=0.9998, type=float)
    p.add_argument("--tta", default="flip4", choices=["none", "hflip", "flip4"])
    p.add_argument("--thr_bootstrap", default=200, type=int,
                    help="Resamples for bootstrap-smoothed threshold selection (0 = plain argmax-F1)")
    p.add_argument("--ordinal_aux", action="store_true", default=False,
                    help="Add a cumulative-link (CORAL-style) ordinal head on the "
                         "ICDR grade alongside the binary outputs. Fixes the "
                         "representation; --severity_match fixes the distribution.")
    p.add_argument("--ordinal_weight", default=0.3, type=float,
                    help="Weight on the ordinal term. The binary objective stays "
                         "primary because it is what we score.")
    p.add_argument("--ordinal_mbrset_csv",
                    default="/data/users4/nshaik3/Datasets/mBRSET/physionet.org/"
                            "files/mbrset/1.0/labels_mbrset.csv",
                    help="mBRSET grades (final_icdr), same 0-4 ICDR scale as "
                         "BRSET's DR_ICDR -- which is what makes one shared head "
                         "legitimate across the two cohorts.")
    p.add_argument("--task", default="convnextv2_large_BRSET_strong_baseline")
    p.add_argument("--output_dir", default="/home/users/sthummala2/brset-convnextv2/results")
    p.add_argument("--seed", default=0, type=int)
    return p.parse_args()



class MaskedLabelLoss(nn.Module):
    """Per-label validity mask for rows whose labels are only partly known.

    DDR/Aptos/Messidor2 carry an ICDR grade but no macular-oedema label. Writing
    a 0 there would teach the model that every imported eye is oedema-free, which
    is exactly the kind of silent label corruption that made the translator arms
    worse. The mask zeroes that term instead.
    """

    def __init__(self, base, nb, off):
        super().__init__()
        self.base, self.nb, self.off = base, nb, off

    def forward(self, logits, targets):
        w = targets[:, self.off:self.off + self.nb].clamp(0.0, 1.0)
        return self.base(logits[:, :self.nb], targets[:, :self.nb], weight=w)


def load_extra_images(ds, args):
    """Append external graded POSITIVES to the training set.

    Positives only, matched count across arms: the arms then differ in the
    severity of what was added and in nothing else. Returns the number added.
    """
    meta = pd.read_csv(args.extra_manifest)
    meta = meta[meta.grade >= 1]          # positives only
    if args.extra_severity == "mild":
        pool = meta[meta.grade.isin([1, 2])]
    elif args.extra_severity == "severe":
        pool = meta[meta.grade.isin([3, 4])]
    else:
        pool = meta
    if len(pool) < args.extra_n:
        raise SystemExit(f"extra_severity={args.extra_severity}: asked for "
                         f"{args.extra_n} but only {len(pool)} available")
    rng = np.random.RandomState(args.extra_seed)
    sel = pool.iloc[rng.permutation(len(pool))[:args.extra_n]]
    add = pd.DataFrame(dict(
        file=sel.file.values,                      # absolute paths
        diabetic_retinopathy=np.ones(len(sel), dtype=int),
        macular_edema=np.zeros(len(sel), dtype=int),   # masked, never learned
        domain=np.full(len(sel), 2)))
    base = ds.labels_df
    for c in base.columns:
        if c not in add.columns:
            add[c] = 0
    ds.labels_df = pd.concat([base, add[base.columns]], ignore_index=True)
    hist = {int(k): int(v) for k, v in sel.grade.value_counts().sort_index().items()}
    print(f"extra images: +{len(sel)} external positives "
          f"(severity={args.extra_severity}) grade mix {hist}", flush=True)
    print(f"  training set {len(base)} -> {len(ds.labels_df)}", flush=True)
    return len(sel)


def load_pseudo_images(ds, args):
    """Append unlabelled target images with teacher pseudo-labels (domain 3).

    Only images the teacher was confident about are in the manifest; their DR label
    is the teacher's, the oedema term is masked unless the manifest says otherwise.
    Returns the manifest so the caller can copy its validity masks.
    """
    m = pd.read_csv(args.pseudo_manifest)
    need = ["file", "diabetic_retinopathy", "macular_edema", "_mdr", "_med"]
    if list(m.columns[:5]) != need:
        raise SystemExit(f"pseudo manifest columns {list(m.columns)}; need {need}")
    add = m[need[:3]].copy()
    add["domain"] = 3
    base = ds.labels_df
    for c in base.columns:
        if c not in add.columns:
            add[c] = 0
    ds.labels_df = pd.concat([base, add[base.columns]], ignore_index=True)
    print(f"pseudo images: +{len(m)} unlabelled target images with teacher labels "
          f"(DR+ {int(m.diabetic_retinopathy.sum())}, DR- {int((m.diabetic_retinopathy == 0).sum())}; "
          f"oedema valid on {int(m._med.sum())}); training set {len(base)} -> {len(ds.labels_df)}", flush=True)
    return m


class EyePairs(torch.utils.data.Dataset):
    """Two different photos of the same UNLABELLED target eye, each with its own augmentation.

    Eyes come from the training split's target rows (file and domain columns only) joined with
    acquisition metadata (patient, laterality) -- no diagnosis column is ever parsed."""

    MB = "/data/users4/nshaik3/Datasets/mBRSET/physionet.org/files/mbrset/1.0/labels_mbrset.csv"

    def __init__(self, split_dir, tf):
        tr = pd.read_csv(Path(split_dir) / "labels.csv", usecols=["file", "domain"])
        assert sorted(tr.columns) == ["domain", "file"], "label columns must never be parsed"
        meta = pd.read_csv(self.MB, usecols=["file", "patient", "laterality"])
        assert sorted(meta.columns) == ["file", "laterality", "patient"], "no diagnosis column may be parsed"
        d = tr[tr.domain == 1].merge(meta, on="file", how="inner")
        d["eye"] = d.patient.astype(str) + "_" + d.laterality.astype(str)
        g = d.groupby("eye").file.apply(list)
        self.eyes = [v for v in g.values if len(v) >= 2]
        self.split_dir, self.tf = Path(split_dir), tf

    def __len__(self):
        return len(self.eyes)

    def __getitem__(self, i):
        a, b = np.random.choice(self.eyes[i], 2, replace=False)
        ld = lambda f: self.tf(Image.open(self.split_dir / f).convert("RGB"))
        return ld(a), ld(b)


def _cycle(loader):
    while True:
        for x in loader:
            yield x


def _grade_map(args):
    """filename stem -> ICDR grade, over both cohorts.

    BRSET grades live in DR_ICDR and mBRSET's in final_icdr; both are the same
    0-4 ICDR scale. BRSET's header carries a BOM, so the id column is taken
    positionally rather than by name.
    """
    m = {}
    for path, idcol, gcol in ((args.severity_csv, "image_id", "DR_ICDR"),
                              (args.ordinal_mbrset_csv, "file", "final_icdr")):
        if not path or not Path(path).exists():
            continue
        df = pd.read_csv(path)
        key = idcol if idcol in df.columns else df.columns[0]
        if gcol not in df.columns:
            continue
        ks = (df[key].astype(str).str.rsplit("/", n=1).str[-1]
                .str.replace(r"\.(jpg|jpeg|png)$", "", regex=True))
        for k, g in zip(ks, df[gcol]):
            if not pd.isna(g):
                m[k] = float(g)
    return m


def attach_ordinal_targets(labels_df, gmap, name=""):
    """Write the cumulative ordinal targets into labels_df, in place."""
    keys = (labels_df["file"].astype(str).str.rsplit("/", n=1).str[-1]
            .str.replace(r"\.(jpg|jpeg|png)$", "", regex=True))
    g = keys.map(gmap)
    known = (~g.isna()).values
    gv = g.fillna(0.0).values
    for k in (1, 2, 3, 4):
        labels_df[f"_ord{k}"] = ((gv >= k) & known).astype("float32")
    labels_df["_ordm"] = known.astype("float32")
    print(f"ordinal_aux: {name} {int(known.sum())}/{len(labels_df)} images graded",
          flush=True)
    return int(known.sum())


class OrdinalAuxLoss(nn.Module):
    """Binary objective, unchanged, plus masked BCE on the cumulative grade head.

    The mask arrives as the last target column and is applied as a soft
    per-image weight rather than a hard gate, because mixup and label smoothing
    both act on the target tensor before the loss sees it: a blended mask
    correctly down-weights an image whose grade is only partly known.
    """

    def __init__(self, base, nb, weight):
        super().__init__()
        self.base, self.nb, self.weight = base, nb, weight

    def forward(self, logits, targets):
        main = self.base(logits[:, :self.nb], targets[:, :self.nb])
        o_lg = logits[:, self.nb:self.nb + 4]
        o_tg = targets[:, self.nb:self.nb + 4]
        w = targets[:, self.nb + 4:self.nb + 5].clamp(0.0, 1.0)
        bce = F.binary_cross_entropy_with_logits(o_lg, o_tg, reduction="none")
        aux = (bce * w).sum() / w.sum().clamp(min=1.0) / 4.0
        return main + self.weight * aux


def apply_severity_match(weights, labels_df, args):
    """Reweight source positives to the target cohort's ICDR severity mix.

    The binary DR label collapses an ordinal scale, and the two cohorts sit at
    opposite ends of it. Reweighting changes WHICH positives the model spends
    its capacity on, without adding a single target image or label.
    """
    meta = pd.read_csv(args.severity_csv)
    idcol = "image_id" if "image_id" in meta.columns else meta.columns[0]
    meta["_k"] = meta[idcol].astype(str).str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)
    g = dict(zip(meta["_k"], meta["DR_ICDR"]))

    keys = (labels_df["file"].astype(str)
            .str.rsplit("/", n=1).str[-1]
            .str.replace(r"\.(jpg|jpeg|png)$", "", regex=True))
    grade = keys.map(g).values
    pos = labels_df["diabetic_retinopathy"].values == 1
    if "domain" in labels_df.columns:
        pos = pos & (labels_df["domain"].values == 0)   # source positives only
    have = pos & ~pd.isna(grade)
    if have.sum() < 50:
        print(f"severity_match: only {int(have.sum())} source positives matched a grade; "
              f"SKIPPING", flush=True)
        return weights

    tgt = [float(x) for x in args.severity_target.split(",")]
    gi = np.where(have, np.nan_to_num(grade, nan=0).astype(float), np.nan)
    cur = np.array([np.nansum(gi == k) for k in (1, 2, 3, 4)], dtype=float)
    cur = cur / max(cur.sum(), 1.0)
    w = np.asarray(weights, dtype=float).copy()
    print(f"severity_match: {int(have.sum())} source positives graded", flush=True)
    print(f"  {'grade':>6} {'source':>8} {'target':>8} {'weight':>8}", flush=True)
    for k in (1, 2, 3, 4):
        f_cur, f_tgt = cur[k - 1], tgt[k - 1]
        mult = (f_tgt / f_cur) if f_cur > 1e-9 else 1.0
        sel = have & (gi == k)
        w[sel] *= mult
        print(f"  {k:>6} {f_cur:8.3f} {f_tgt:8.3f} {mult:8.3f}", flush=True)
    # preserve total positive mass so this changes composition, not prevalence
    w[pos] *= np.asarray(weights, dtype=float)[pos].sum() / max(w[pos].sum(), 1e-9)
    return w


def apply_pos_rate(weights, labels_df, rate):
    """Reweight sampling so DR positives are drawn at `rate`.

    With n_pos positives and n_neg negatives, giving negatives weight 1 and
    positives weight w makes the expected positive share w*n_pos/(w*n_pos+n_neg).
    Solving for the requested rate gives w = rate*n_neg / (n_pos*(1-rate)).
    """
    y = labels_df["diabetic_retinopathy"].values.astype(float)
    n_pos, n_neg = float(y.sum()), float((1 - y).sum())
    if n_pos < 1 or n_neg < 1 or not (0.0 < rate < 1.0):
        return weights
    w_pos = rate * n_neg / (n_pos * (1.0 - rate))
    out = np.where(y == 1, w_pos, 1.0)
    eff = float((out * y).sum() / out.sum())
    print(f"target_pos_rate: requested {rate:.4f}, achieved {eff:.4f} "
          f"(positive weight {w_pos:.3f}, {int(n_pos)} pos / {int(n_neg)} neg)", flush=True)
    return out


def apply_target_budget(ds, args):
    """Keep only `target_budget` images from the target domain (domain == 1).

    The label-efficiency curve is the backbone of the synthesis study: it asks
    how many REAL handheld-camera labels are needed to reach a given accuracy,
    with and without synthesised ones. Subsampling here instead of on disk
    keeps a single copy of the data and makes the budget a run parameter.

    Two properties matter for the curve to be interpretable:
      - Stratified on the DR label, so a small budget does not drift away from
        the full target set's 23.4 percent positive rate and confound budget
        with class balance.
      - Nested across budgets. For a fixed budget_seed the permutation does not
        depend on n, so the 500-image set is a strict subset of the 1000-image
        set. Adjacent points on the curve then differ by which images were
        ADDED, not by an unrelated redraw.
    """
    n = getattr(args, "target_budget", -1)
    if n < 0:
        return
    df = ds.labels_df
    if "domain" not in df.columns:
        raise SystemExit("--target_budget requires a 'domain' column; "
                         "point --data_path at data/finetune_joint")
    src, tgt = df[df.domain == 0], df[df.domain == 1]
    if n >= len(tgt):
        print(f"target budget {n} >= available {len(tgt)}; keeping all", flush=True)
        return
    rng = np.random.RandomState(args.budget_seed)
    sev = getattr(args, "budget_severity", "random")
    gmap = _grade_map(args) if sev != "random" else {}
    keep = []
    for lab, grp in tgt.groupby("diabetic_retinopathy"):
        k = int(round(n * len(grp) / len(tgt)))
        pool = grp
        if sev != "random" and lab == 1:
            # Restrict only the POSITIVES. Budget and positive rate stay
            # identical to the random arm, so the single thing that varies is
            # the severity composition of the positives being bought.
            want = (1, 2) if sev == "mild" else (3, 4)
            g = (grp["file"].astype(str).str.rsplit("/", n=1).str[-1]
                   .str.replace(r"\.(jpg|jpeg|png)$", "", regex=True).map(gmap))
            pool = grp[g.isin(want)]
            print(f"budget_severity={sev}: {len(pool)} of {len(grp)} target "
                  f"positives are grade {want}; drawing {k}", flush=True)
            if len(pool) < k:
                raise SystemExit(f"budget_severity={sev}: need {k} positives but "
                                 f"only {len(pool)} are grade {want}")
        keep.append(rng.permutation(pool.index.values)[:k])
    keep = np.concatenate(keep)
    ds.labels_df = pd.concat([src, tgt.loc[keep]]).reset_index(drop=True)
    kept = ds.labels_df[ds.labels_df.domain == 1]
    print(f"target budget: kept {len(kept)} of {len(tgt)} target images "
          f"(DR+ {int(kept.diabetic_retinopathy.sum())}, "
          f"{100 * kept.diabetic_retinopathy.mean():.1f} percent) "
          f"alongside {len(src)} source images", flush=True)
    if sev != "random" and len(kept):
        # print the realised mix, so the log proves the arm did what it claims
        kg = (kept["file"].astype(str).str.rsplit("/", n=1).str[-1]
                .str.replace(r"\.(jpg|jpeg|png)$", "", regex=True).map(gmap))
        kp = kg[kept["diabetic_retinopathy"].values == 1]
        hist = {int(k): int(v) for k, v in kp.value_counts().sort_index().items()
                if not pd.isna(k)}
        print(f"  realised grade mix of bought positives: {hist}", flush=True)


def build_transforms(args):
    mode = getattr(args, "resize_mode", "squash")
    if mode == "preserve":
        # short side to resize_size, aspect ratio kept; the crop then selects the square
        resize = transforms.Resize(args.resize_size)
    else:
        resize = transforms.Resize((args.resize_size, args.resize_size))
    # When a translator is attached, normalisation is deferred to the training
    # loop: the translator operates on [0,1] images. Augmentation stays here, so
    # the source image is varied first and the camera model applied second, and
    # every other element of the pipeline is byte-identical to the baseline arms.
    train_steps = [
        resize,
        transforms.RandomCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
    ]
    if not getattr(args, "translator_ckpt", ""):
        train_steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    train_tf = transforms.Compose(train_steps)
    eval_tf = transforms.Compose([
        resize,
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
        with torch.autocast(device_type="cuda", dtype=_EVAL_DTYPE[0],
                            enabled=_EVAL_DTYPE[0] is not None):
            views = [images]
            if tta in ("hflip", "flip4"):
                views.append(torch.flip(images, dims=[3]))
            if tta == "flip4":
                views.append(torch.flip(images, dims=[2]))
                views.append(torch.flip(images, dims=[2, 3]))
            probs = torch.stack([torch.sigmoid(model(v).float()) for v in views]).mean(0)
        if _NB_OUT[0] is not None:
            # the aux head must not change the shape of anything downstream:
            # every cached run's metrics and predictions are two-column.
            probs = probs[:, :_NB_OUT[0]]
            targets = targets[:, :_NB_OUT[0]]
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


_NORM_MEAN = None   # set in main() when a translator is attached
_NORM_STD = None


def train_one_epoch(model, ema, loader, optimizer, scheduler, scaler, criterion, device,
                     epoch, args, print_freq=40, translator=None):
    model.train()
    running_loss, n_batches, n_bad = 0.0, 0, 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        flags = None
        if len(batch) == 3:
            images, targets, flags = batch
            flags = flags.to(device, non_blocking=True)
        else:
            images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if translator is not None:
            # synthesise the target camera, then normalise. Mixup runs AFTER so
            # it blends already-translated images, matching the baseline order.
            p_tr = getattr(args, "translate_p", 1.0)
            if flags is not None:
                sel = flags > 0.5
                if sel.any():
                    images = images.clone()
                    images[sel] = translator(images[sel]).clamp_(0.0, 1.0)
            elif p_tr >= 1.0:
                images = translator(images).clamp_(0.0, 1.0)
            elif p_tr > 0.0:
                # per-image Bernoulli, so a batch mixes real and synthetic
                sel = torch.rand(images.size(0), device=images.device) < p_tr
                if sel.any():
                    images = images.clone()
                    images[sel] = translator(images[sel]).clamp_(0.0, 1.0)
            images = (images - _NORM_MEAN) / _NORM_STD
        if args.mixup_alpha > 0:
            images, targets = mixup_batch(images, targets, args.mixup_alpha)
        if args.label_smoothing > 0:
            if _EXTRA_COLS[:] == ["_mdr", "_med"]:
                # smooth the labels only: smoothing the validity masks would give a
                # masked label 5 % weight and quietly teach it (14 Oct fix)
                nb = args.nb_classes
                targets = torch.cat([targets[:, :nb] * (1 - args.label_smoothing) + 0.5 * args.label_smoothing,
                                     targets[:, nb:]], dim=1)
            else:
                targets = targets * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE[args.amp_dtype],
                            enabled=args.amp_dtype != "fp32"):
            loss = criterion(model(images), targets) / args.accum_iter
        # A single non-finite batch poisons the accumulated gradients, and from
        # there the weights. Drop the whole accumulation group instead.
        if not torch.isfinite(loss):
            n_bad += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss).backward()
        if _MV[0] is not None:
            xa, xb = next(_MV[0])
            xa, xb = xa.to(device, non_blocking=True), xb.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE[args.amp_dtype],
                                enabled=args.amp_dtype != "fp32"):
                pr = torch.sigmoid(model(torch.cat([xa, xb])).float()[:, :args.nb_classes])
            pa, pb = pr.chunk(2)
            loss_mv = args.mv_weight * ((pa - pb) ** 2).mean() / args.accum_iter
            if torch.isfinite(loss_mv):
                scaler.scale(loss_mv).backward()
                _MV[1] = 0.98 * _MV[1] + 0.02 * loss_mv.item() * args.accum_iter
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
                  + (f"mv: {_MV[1]:.4f}  " if _MV[0] is not None else "") +
                  f"{time.time()-t0:.0f}s", flush=True)
    scheduler.step()
    if n_bad:
        print(f"WARNING epoch {epoch}: skipped {n_bad} non-finite batches", flush=True)
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
    apply_target_budget(ds_train, args)
    if args.translator_ckpt and args.translate_mode == "union":
        # Real BRSET and synthetic mBRSET side by side, which is what "joint
        # training on BRSET plus synthesized mBRSET" means literally. The epoch
        # doubles, so this sees twice the gradient steps of the baseline; that
        # is a real confound and the paper must report it alongside the prob
        # arms, which hold step count fixed.
        import pandas as _pd
        _a = ds_train.labels_df.copy(); _a["_translate"] = 0.0
        _b = ds_train.labels_df.copy(); _b["_translate"] = 1.0
        ds_train.labels_df = _pd.concat([_a, _b]).reset_index(drop=True)
        print(f"translate_mode=union: training set doubled to {len(ds_train.labels_df)} "
              f"({len(_a)} real + {len(_b)} synthetic)", flush=True)
    print(f"train/val/test sizes: {len(ds_train)}/{len(ds_val)}/{len(ds_test)}", flush=True)

    if args.ordinal_aux:
        _gm = _grade_map(args)
        print(f"ordinal_aux: {len(_gm)} graded images across both cohorts", flush=True)
        n_tr = attach_ordinal_targets(ds_train.labels_df, _gm, "train")
        # val/test get the columns too, so __getitem__ has them to read; the
        # ordinal outputs are never scored, only the binary ones are.
        attach_ordinal_targets(ds_val.labels_df, _gm, "val")
        attach_ordinal_targets(ds_test.labels_df, _gm, "test")
        if n_tr < 50:
            raise SystemExit(f"ordinal_aux: only {n_tr} training images matched a "
                             f"grade; refusing to run a no-op experiment")
        _EXTRA_COLS[:] = ORD_COLS
        _NB_OUT[0] = args.nb_classes

    if args.synth_dir:
        # --synth_dir may list several folders (comma-separated): different twin sets (BRSET twins, mBRSET
        # re-renderings) and/or several renderings of the same image (variants, D084). Each row keeps every
        # twin found under its stem and one is drawn at random on each load.
        sd = args.synth_dir
        have = {}
        for one in str(sd).split(","):
            for q in Path(one).glob("*.png"):
                if not q.name.startswith("."):
                    have[q.stem] = (have[q.stem] + "|" + str(q)) if q.stem in have else str(q)
        df = ds_train.labels_df
        is_src = (df["domain"].values == 0) if ("domain" in df.columns and not args.synth_all) else np.ones(len(df), bool)
        stems = df["file"].astype(str).map(lambda f: Path(f).stem)
        df["_synth"] = [have.get(st, "") if src else "" for st, src in zip(stems, is_src)]
        n_src, n_hit = int(is_src.sum()), int((df["_synth"] != "").sum())
        print(f"synth: {n_hit}/{n_src} BRSET training images have a synthetic twin in {sd}; "
              f"p={args.synth_p}", flush=True)
        if n_hit < 0.99 * n_src:
            raise SystemExit(f"synth: only {n_hit}/{n_src} twins found; refusing to run")
        if args.synth_p >= 1.0 and n_hit != n_src:
            # synthetic-only runs (D087): a row without a twin would silently stay a REAL image
            raise SystemExit(f"synth: synthetic-only run but {n_src - n_hit} rows have no twin; refusing to run")
        _SYNTH_P[0] = float(args.synth_p)

    if args.extra_manifest or args.pseudo_manifest:
        if args.ordinal_aux:
            raise SystemExit("--extra_manifest/--pseudo_manifest with --ordinal_aux is not wired "
                             "(the added images carry no ordinal mask); run "
                             "them as separate arms")
        if args.extra_manifest:
            load_extra_images(ds_train, args)
        pm = load_pseudo_images(ds_train, args) if args.pseudo_manifest else None
        # label-validity masks: everything native is valid; the external rows
        # have a valid DR label and an invalid oedema label.
        for d, nm in ((ds_train.labels_df, "train"), (ds_val.labels_df, "val"),
                      (ds_test.labels_df, "test")):
            ext = (d["domain"].values == 2) if "domain" in d.columns else \
                  np.zeros(len(d), bool)
            d["_mdr"] = 1.0
            d["_med"] = np.where(ext, 0.0, 1.0)
        if pm is not None:
            dtr = ds_train.labels_df
            isp = dtr["domain"].values == 3
            assert isp.sum() == len(pm), "pseudo rows lost or duplicated"
            dtr.loc[isp, "_mdr"] = pm["_mdr"].values.astype(float)
            dtr.loc[isp, "_med"] = pm["_med"].values.astype(float)
        _EXTRA_COLS[:] = ["_mdr", "_med"]
        _NB_OUT[0] = args.nb_classes
        print(f"label masks on: oedema is masked for "
              f"{int((ds_train.labels_df['_med'].values == 0).sum())} external "
              f"training rows", flush=True)

    # Attach the translator, if one was given. Training images only: validation
    # and test are real target-domain images and must never be synthesised.
    translator = None
    if args.translator_ckpt:
        global _NORM_MEAN, _NORM_STD
        import importlib.util as _ilu
        _sp = _ilu.spec_from_file_location("_synth", str(Path(__file__).parent / "88_synth_augment.py"))
        _sa = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_sa)
        translator = _sa.load_translator(args.translator_ckpt, device=device)
        _NORM_MEAN = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        _NORM_STD = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        print("translator attached: TRAIN images are synthesised on the fly; "
              "val/test remain real target images", flush=True)

    sample_weights, counts = build_sample_weights(ds_train.labels_df)
    if args.severity_match:
        sample_weights = apply_severity_match(sample_weights, ds_train.labels_df, args)
    if args.target_pos_rate > 0:
        sample_weights = apply_pos_rate(sample_weights, ds_train.labels_df,
                                        args.target_pos_rate)
    print(f"train label counts: {counts} (of {len(ds_train)})", flush=True)

    # Report the positive rate the loss will ACTUALLY see. With the sampler on this
    # is far higher than the dataset's own rate, which is what an asymmetric loss
    # must be tuned against -- not the raw prevalence.
    w = np.asarray(sample_weights, dtype=np.float64)
    pick = (w / w.sum()) if not args.no_oversample else np.full(len(w), 1.0 / len(w))
    for c in LABEL_COLS:
        eff = float((pick * ds_train.labels_df[c].values).sum())
        raw = counts[c] / len(ds_train)
        print(f"  {c}: raw positive rate {100*raw:.2f}%  ->  effective (as sampled) "
              f"{100*eff:.2f}%  [{(1-eff)/max(eff,1e-9):.2f}:1]", flush=True)

    if args.no_oversample:
        print("oversampling DISABLED - shuffling uniformly", flush=True)
        sampler = None
        shuffle = True
    else:
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                               shuffle=shuffle, num_workers=args.num_workers,
                               pin_memory=True, drop_last=True)
    if args.mv_weight > 0:
        pairs = EyePairs(data_path / "train", train_tf)
        _MV[0] = _cycle(DataLoader(pairs, batch_size=args.mv_batch, shuffle=True, num_workers=4,
                                   pin_memory=True, drop_last=True, persistent_workers=True))
        print(f"same-eye consistency: {len(pairs.eyes)} unlabelled target eyes with >= 2 photos; "
              f"weight {args.mv_weight}, {args.mv_batch} pairs per micro-step", flush=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    if args.amp_dtype == "auto":
        args.amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    print(f"autocast dtype: {args.amp_dtype}", flush=True)
    _EVAL_DTYPE[0] = (None if args.amp_dtype == "fp32" else AMP_DTYPE[args.amp_dtype])

    _n_out = args.nb_classes + (4 if args.ordinal_aux else 0)
    model = timm.create_model(args.model, pretrained=True, num_classes=_n_out,
                               drop_path_rate=args.drop_path).to(device)
    if args.ordinal_aux:
        print(f"ordinal_aux: head widened {args.nb_classes} -> {_n_out} "
              f"(4 cumulative grade outputs), weight {args.ordinal_weight}", flush=True)
    print(f"n_parameters: {sum(p.numel() for p in model.parameters())}", flush=True)

    if args.warm_start_ckpt:
        ck = torch.load(args.warm_start_ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"warm start: {args.warm_start_ckpt}", flush=True)
        print(f"  source epoch {ck.get('epoch')} variant {ck.get('variant')}; "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if missing or unexpected:
            raise SystemExit(f"refusing to continue: state_dict mismatch "
                             f"missing={missing[:5]} unexpected={unexpected[:5]}")

    if args.loss == "afl":
        criterion = AsymmetricFocalLoss(gamma=args.afl_gamma, delta=args.afl_delta)
        print(f"criterion = AsymmetricFocalLoss(gamma={args.afl_gamma}, "
              f"delta={args.afl_delta})  [no modulation on positives]", flush=True)
    elif args.loss == "asl":
        criterion = AsymmetricLoss(gamma_neg=args.asl_gamma_neg,
                                    gamma_pos=args.asl_gamma_pos, clip=args.asl_clip)
        print(f"criterion = AsymmetricLoss(gamma_neg={args.asl_gamma_neg}, "
              f"gamma_pos={args.asl_gamma_pos}, clip={args.asl_clip})", flush=True)
    else:
        criterion = MultiLabelFocalLoss(gamma=args.focal_gamma)
        print(f"criterion = MultiLabelFocalLoss(gamma={args.focal_gamma})", flush=True)

    if args.extra_manifest or args.pseudo_manifest:
        if not isinstance(criterion, MultiLabelFocalLoss):
            raise SystemExit("--extra_manifest needs --loss focal (only that "
                             "loss accepts the per-label validity mask)")
        criterion = MaskedLabelLoss(criterion, args.nb_classes, args.nb_classes)
        print("criterion wrapped in MaskedLabelLoss (oedema masked for "
              "external rows)", flush=True)

    if args.ordinal_aux:
        criterion = OrdinalAuxLoss(criterion, args.nb_classes, args.ordinal_weight)
        print(f"criterion wrapped in OrdinalAuxLoss(weight={args.ordinal_weight})",
              flush=True)

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
                                      scaler, criterion, device, epoch, args,
                                      translator=translator)

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
