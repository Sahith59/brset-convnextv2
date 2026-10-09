"""Joint training of the dual-domain BRSET -> mBRSET translator.

Trains stage 1 (`scripts/72_physical_layer.py`, 19 bounded acquisition
parameters) and stage 2 (`scripts/82_refiner.py`, a ~68k-parameter bounded
residual refiner) against the SAME objective at the SAME time. The DuDoNet
argument (Lin et al., CVPR 2019) is that the two domains have to be linked
differentiably or each one compensates for the other's errors in ways that
neither can be held responsible for. Fitting stage 1 first and then freezing it
would let stage 2 silently repair a bad physics fit, and the division of labour
between interpretable physics and bounded residual, which is the claim the
paper actually makes, would stop being true. `--freeze_phys` exists only as the
ablation that demonstrates this.

FOUR LOSS TERMS, and why each is shaped the way it is.

1. STATISTIC LOSS. Script 79's `distance` and `FIT_KEYS` are imported rather
   than reimplemented, so there is exactly one definition of the objective in
   the repository and no chance of the two drifting. It matches the mean AND the
   spread of 13 non-redundant statistics, because matching means alone is
   satisfied perfectly by a transform that collapses variability.

   The per-key weights are NOT uniform, and the reason is downstream, not
   statistical. The classifier trainer applies
   ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
   (scripts/30_train_strong_baseline.py line 343) to every image it sees. Any
   global brightness, contrast or saturation that this translator matches is
   therefore destroyed before the classifier reads it, and capacity spent on it
   is capacity wasted. ColorJitter does NOT touch hue, and RandomCrop does not
   resample, so the chromatic FRACTIONS and the per-channel Laplacian variances
   survive intact to the downstream task. Those are up-weighted and the washed-
   out ones are down-weighted; see STAT_WEIGHTS. This is also where the measured
   gap lives: blue fraction 1.43x, blue std 2.86x, green std 1.60x, red 1.03x.

2. STYLE LOSS against real mBRSET, Gram matrices on a frozen VGG16.
   VGG and not ConvNeXt V2, for a mechanical reason rather than a habitual one:
   a Gram matrix measures per-channel feature energy and its cross-channel
   correlation, and ConvNeXt V2's GRN layer explicitly normalises per-channel
   energy away. Measuring with the one encoder designed to delete the quantity
   would be self-defeating. VGG16 has no normalisation layers at all. Availability
   was verified before committing to it: timm's `vgg16.tv_in1k` resolves through
   the HuggingFace hub and is now in the local HF cache (528 MB), so compute
   nodes read from cache and never need the network mid-run.

   Scored on CROPS TAKEN FROM INSIDE THE FUNDUS, never whole images. The two
   datasets differ in field-of-view geometry (BRSET mean aspect 1.287 with the
   disc clipped top and bottom; mBRSET uniform 1600x1600 with the disc
   inscribed). That difference is the single most separable cue between the
   domains, it is handled geometrically outside the generator per decision D1 of
   report/STAGE2_DESIGN.md, and a whole-image style loss would notice it first
   and reward the refiner for painting black bars. Cropping inside the disc
   removes it from the comparison and leaves retinal colour and texture, which
   is what the loss is for.

3. ADVERSARIAL, optional, OFF by default (`--w_adv 0.0`). One knob rather than a
   flag plus a weight, so there is no configuration in which the GAN is "on"
   but contributes nothing. Rationale for defaulting it off is the deadline:
   mode collapse is the most common way this class of method fails, and the
   first three terms are a stable moment-matching objective that trains in
   hours. The discriminator is a small spectral-normalised PatchGAN scored on
   the same inside-the-fundus crops, for the same reason.

4. RESIDUAL PENALTY, and specifically NOT plain L1 on the full-resolution
   residual. L1 is a sparsity prior: it prices a residual by its total absolute
   mass, so a dense low-frequency chroma shift across the whole disc is the most
   expensive thing it can buy, while a handful of sparse lesion-sized blobs is
   almost free. That is precisely inverted relative to what the measurement
   asks for. The default is L2, which prices by amplitude rather than support
   and is indifferent to whether the change is dense or sparse;
   `--res_mode l1_down` applies L1 to a box-downsampled residual, which removes
   the sparsity preference below the pooling scale and is the honest version of
   an L1 prior here. Both are normalised by the branch's own eps so the two
   weights are comparable numbers rather than accidents of scale.

NUMERICAL PRECISION. `--amp bf16` applies to the frozen VGG and the
discriminator ONLY. The translator forward and the statistic loss stay in fp32
on purpose: the Laplacian-variance statistics are of order 1e-4 on a [0,1]
image and bf16 carries about three significant decimal digits, so computing
them in bf16 would quantise the single most important measured quantity into
noise.

RESOLUTION. Images are loaded Resize((presize,presize)) then centre-cropped to
`size`, with defaults 560 and 512, which is the trainer's own pitch and the
pitch script 76 measures the targets at. Laplacian variance is strongly scale
dependent (BRSET 0.000078 native against 0.000257 squashed to 512), so a
mismatch between the fitting pitch and the target pitch silently rescales the
sharpness target. The script refuses to start if the targets JSON was measured
at a different pitch unless `--allow_pitch_mismatch` is passed.

DATA HANDLING. BRSET and mBRSET are credentialed PhysioNet data under a DUA.
This script reads images from the gitignored `data/` tree and writes only
aggregate numbers, a checkpoint and, if `--save_panel` is passed, a qualitative
panel to a gitignored path. No patient image is ever written inside a tracked
location.

Unit tests: `--selftest`, synthetic tensors only, no image decoding, so it is
allowed to run on the login node per AGENTS.md. Training itself must go through
Slurm; see `scripts/84_train_translator.slurm`.
"""
import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent


def _load(name, mod):
    spec = importlib.util.spec_from_file_location(mod, str(HERE / name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("72_physical_layer.py", "physical_layer")
R = _load("82_refiner.py", "refiner")

# Import the fit objective rather than reimplementing it. If script 79 ever
# changes which statistics it fits, this script changes with it; a private copy
# would drift and the paper would be quoting two different objectives.
try:
    F79 = _load("79_fit_physics_v2.py", "fit_physics_v2")
    FIT_KEYS, DIAG_KEYS, distance = F79.FIT_KEYS, F79.DIAG_KEYS, F79.distance
    _OBJ_SRC = "scripts/79_fit_physics_v2.py"
except Exception as e:  # pragma: no cover - only if 79 is mid-edit
    print(f"[warn] could not import script 79 ({e}); using the frozen key list")
    FIT_KEYS = ["brightness", "contrast", "falloff", "saturation",
                "r_frac", "g_frac", "b_frac", "r_std", "g_std", "b_std",
                "r_lapvar", "g_lapvar", "b_lapvar"]
    DIAG_KEYS = ["sharpness", "r_mean", "g_mean", "b_mean"]
    _OBJ_SRC = "frozen fallback inside script 83"

    def distance(stats, target, weights=None, var_weight=1.0):
        tot = 0.0
        for k in FIT_KEYS:
            t = target[k]["mbrset_mean"]
            sd = max(target[k]["mbrset_std"], 1e-6)
            w = 1.0 if weights is None else weights.get(k, 1.0)
            v = stats[k]
            tot = tot + w * ((v.mean() - t) / sd) ** 2
            if var_weight:
                tot = tot + var_weight * w * ((v.std(unbiased=False) - sd) / sd) ** 2
        return tot


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Per-statistic weights, set by what survives the downstream augmentation rather
# than by taste. 0.25 means "the classifier will never see this, keep only
# enough pull to stop the fit wandering"; 2.0 means "this is both a measured
# dominant axis and invariant to ColorJitter, so it is where the gradient should
# go". Any key not listed defaults to 1.0 inside `distance`.
STAT_WEIGHTS = {
    # washed out by ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
    "brightness": 0.25,
    "contrast": 0.25,
    "saturation": 0.25,
    # spatial, and ColorJitter is global, so it survives
    "falloff": 1.0,
    # HUE. ColorJitter has no hue term. Measured dominant chromatic axis:
    # blue fraction 0.105 -> 0.151 (1.43x), green 0.297 -> 0.266 (0.90x).
    "r_frac": 2.0, "g_frac": 2.0, "b_frac": 2.0,
    # per-channel spread: part contrast (washed), part chromatic (survives)
    "r_std": 1.0, "g_std": 1.0, "b_std": 1.0,
    # per-channel high-frequency energy. Untouched by ColorJitter, and RandomCrop
    # is a crop rather than a resample so the pixel pitch is preserved. This is
    # the measured channel asymmetry (blue 2.86x, green 1.60x, red 1.03x) and
    # the reason stage 2 has a high-frequency branch at all.
    "r_lapvar": 2.0, "g_lapvar": 2.0, "b_lapvar": 2.0,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class FundusDir(torch.utils.data.Dataset):
    """Images from one split directory, loaded at the trainer's pitch.

    Labels are deliberately not read. This script only ever matches appearance
    distributions, so pulling per-image diagnostic labels in would add a DUA
    exposure for no benefit.
    """

    def __init__(self, split_dir, presize=560, size=512, limit=0, seed=0):
        import pandas as pd
        self.dir = Path(split_dir)
        files = pd.read_csv(self.dir / "labels.csv").file.values
        if limit and limit < len(files):
            files = np.random.default_rng(seed).choice(files, limit, replace=False)
        self.files = list(files)
        self.presize, self.size = int(presize), int(size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        try:
            im = Image.open(self.dir / self.files[i]).convert("RGB")
        except Exception:
            # A handful of BRSET files are truncated (results/corrupted_files.txt).
            # Dropping them in collate is better than substituting a different
            # image, which would silently reweight the appearance distribution.
            return None
        if self.presize:
            im = im.resize((self.presize, self.presize), Image.BILINEAR)
            o = (self.presize - self.size) // 2
            im = im.crop((o, o, o + self.size, o + self.size))
        else:
            im = im.resize((self.size, self.size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1)


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return torch.stack(batch) if batch else None


def infinite(loader):
    while True:
        for b in loader:
            if b is not None:
                yield b


# ---------------------------------------------------------------------------
# Inside-the-fundus crops: the support the style loss and the discriminator see
# ---------------------------------------------------------------------------

def inner_crops(x, mask, crop, n_crops=2, generator=None):
    """Random square crops guaranteed to lie inside the circular field of view.

    The field of view is a disc, so its area gives an effective radius and its
    first moment gives a centre. A square of half-side 0.9r/sqrt(2) centred
    there is inscribed in the disc with a 10 percent margin, and any crop whose
    top-left corner stays inside that square is entirely retina.

    Why bother. The style loss and the discriminator are the two terms that see
    raw pixels rather than scalar statistics, so they are the two that would
    otherwise learn the FRAMING difference between the datasets, which stage 2
    is forbidden to change and which is normalised geometrically instead
    (decision D1). Removing the surround from their field of view removes that
    failure mode at the source rather than penalising it afterwards.
    """
    b, _, h, w = x.shape
    out = []
    if mask.shape[0] == 1 and b > 1:
        mask = mask.expand(b, -1, -1, -1)   # a shared mask, e.g. in the tests
    area = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    rad = (area / math.pi).sqrt()
    yy = torch.arange(h, device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
    xx = torch.arange(w, device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
    cy = (yy * mask).sum(dim=(1, 2, 3)) / area
    cx = (xx * mask).sum(dim=(1, 2, 3)) / area
    for i in range(b):
        half = float(0.9 * rad[i] / math.sqrt(2.0))
        span = max(0.0, 2.0 * half - crop)
        for _ in range(n_crops):
            if span <= 0:
                # Crop wider than the inscribed square: fall back to centred,
                # which is still the deepest-inside-the-disc choice available.
                t = int(round(float(cy[i]) - crop / 2.0))
                l = int(round(float(cx[i]) - crop / 2.0))
            else:
                u = torch.rand(2, device=x.device, generator=generator)
                t = int(round(float(cy[i]) - half + float(u[0]) * span))
                l = int(round(float(cx[i]) - half + float(u[1]) * span))
            t = max(0, min(h - crop, t))
            l = max(0, min(w - crop, l))
            out.append(x[i:i + 1, :, t:t + crop, l:l + crop])
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# Style loss
# ---------------------------------------------------------------------------

def gram(f):
    """Normalised Gram matrix per image, (B,C,C).

    Divided by C*H*W so the value does not depend on feature-map size and the
    per-layer losses are commensurable without hand-tuned per-layer weights.
    """
    b, c, h, w = f.shape
    v = f.reshape(b, c, h * w)
    return v @ v.transpose(1, 2) / (c * h * w)


class VGGGram(nn.Module):
    """Frozen VGG16 feature extractor for Gram-matrix style matching.

    Early layers by default (out_indices 0,1,2 -> 64, 128, 256 channels at
    reductions 1, 2, 4). Two reasons for early rather than late: the gap being
    matched is colour and noise texture, not semantics, and a Gram matrix of 512
    channels over a 16x16 map is estimated from 256 samples per channel pair and
    is badly rank-deficient, which makes the deep layers' Grams mostly noise at
    the crop sizes used here.
    """

    def __init__(self, layers=(0, 1, 2), model="vgg16.tv_in1k"):
        super().__init__()
        import timm
        self.net = timm.create_model(model, pretrained=True, features_only=True,
                                     out_indices=tuple(layers))
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.channels = self.net.feature_info.channels()

    def forward(self, x):
        return self.net((x - self.mean) / self.std)

    def style_loss(self, fake, real):
        """Distribution-level Gram match: the MEAN Gram over the fake crops
        against the mean Gram over the real crops.

        Per-crop pairing would be meaningless because the two domains are
        unpaired; this is a first-moment match in Gram space, which is the
        stable version of the objective and is what the budget allows. The
        adversarial term is what a reviewer will ask for beyond it, and it is
        available behind --w_adv.
        """
        tot = fake.new_zeros(())
        ff, rr = self.forward(fake), self.forward(real)
        for a, b in zip(ff, rr):
            ga, gb = gram(a).mean(0), gram(b).mean(0)
            tot = tot + (ga - gb).pow(2).mean()
        return tot / max(len(ff), 1)


# ---------------------------------------------------------------------------
# Residual penalty
# ---------------------------------------------------------------------------

def residual_penalty(r, weight, eps, mode="l2", down=8):
    """Price of the residual, normalised by the branch's own bound.

    Normalising by eps makes the returned number live in roughly [0,1] for both
    branches, so --w_res_lf and --w_res_hf can be read as relative priorities
    instead of being silently dominated by the fact that eps_lf is 7.5x eps_hf
    and an unnormalised L2 would weight it 56x harder.

    mode l2       mean((r/eps)^2) inside the mask. Prices amplitude, and is
                  indifferent to whether the change is dense or sparse. This is
                  the default because the change we WANT is dense.
    mode l1_down  mean(|boxpool(r, down)|/eps). L1 after pooling, so the
                  sparsity preference is removed below the pooling scale: a
                  lesion-sized blob is averaged down and stops being the cheap
                  option. Plain full-resolution L1 is deliberately not offered.
    """
    if weight is None or eps <= 0:
        return r.new_zeros(())
    n = weight.sum().clamp(min=1.0)
    if mode == "l2":
        return ((r / eps) ** 2 * weight).sum() / (n * r.shape[1])
    if mode == "l1_down":
        rd = F.avg_pool2d(r, down)
        wd = F.avg_pool2d(weight, down)
        return ((rd / eps).abs() * wd).sum() / (wd.sum().clamp(min=1.0) * r.shape[1])
    raise ValueError(f"unknown res_mode {mode}")


# ---------------------------------------------------------------------------
# Adversarial (optional)
# ---------------------------------------------------------------------------

def lsgan_d_loss(d, real, fake):
    """Least-squares GAN. Chosen over the saturating or hinge forms because its
    gradient does not vanish when the discriminator is winning, which is the
    regime a 68k-parameter bounded generator will spend most of its time in."""
    return 0.5 * ((d(real) - 1.0) ** 2).mean() + 0.5 * (d(fake) ** 2).mean()


def lsgan_g_loss(d, fake):
    return ((d(fake) - 1.0) ** 2).mean()


# ---------------------------------------------------------------------------
# Objective assembly, shared by training and the self-test
# ---------------------------------------------------------------------------

def translator_losses(model, x_src, x_tgt, targets, a, vgg=None, disc=None,
                      generator=None, amp_dtype=None):
    """One forward pass and every loss term, returned itemised.

    The statistic mask is the fundus mask of the ORIGINAL source image, not of
    the output. If the mask were recomputed on the output, the optimiser could
    improve any masked statistic by dimming the rim until those pixels fell
    below the 0.04 threshold and left the support, which is the statistical
    equivalent of cropping the hard part out of the exam.
    """
    out, parts = model(x_src, generator=generator, return_parts=True)
    mask = parts["mask"]
    w_res = parts["mask_eff"]

    # fp32 on purpose; see the NUMERICAL PRECISION note in the module docstring
    stats = P.image_stats_torch(out, mask=mask)
    L_stat = distance(stats, targets, weights=STAT_WEIGHTS, var_weight=a.var_weight)

    L_style = out.new_zeros(())
    L_adv = out.new_zeros(())
    if (a.w_style > 0 and vgg is not None) or a.w_adv > 0:
        tm = P.fundus_mask(x_tgt)
        f_crops = inner_crops(out, mask, a.style_crop, a.n_crops, generator)
        r_crops = inner_crops(x_tgt, tm, a.style_crop, a.n_crops, generator)
        ctx = (torch.autocast(device_type=out.device.type, dtype=amp_dtype)
               if amp_dtype is not None else _nullctx())
        with ctx:
            if a.w_style > 0 and vgg is not None:
                L_style = vgg.style_loss(f_crops, r_crops).float()
            if a.w_adv > 0 and disc is not None:
                L_adv = lsgan_g_loss(disc, f_crops).float()
        parts["f_crops"], parts["r_crops"] = f_crops, r_crops

    L_lf = residual_penalty(parts["r_lf"], w_res, model.refiner.eps_lf,
                            a.res_mode, a.res_down)
    L_hf = (residual_penalty(parts["r_hf"], w_res, model.refiner.eps_hf,
                             a.res_mode, a.res_down)
            if model.refiner.hf_mode != "off" else out.new_zeros(()))

    total = (a.w_stat * L_stat + a.w_style * L_style + a.w_adv * L_adv
             + a.w_res_lf * L_lf + a.w_res_hf * L_hf)
    items = {"total": total, "stat": L_stat, "style": L_style, "adv": L_adv,
             "res_lf": L_lf, "res_hf": L_hf}
    return total, items, out, parts


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # data
    ap.add_argument("--src_train", default="data/finetune_multilabel/train")
    ap.add_argument("--src_val", default="data/finetune_multilabel/val")
    ap.add_argument("--tgt_train", default="data/finetune_mbrset_multilabel/train")
    ap.add_argument("--presize", type=int, default=560,
                    help="the trainer's pre-resize; must match the targets JSON")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_src", type=int, default=0)
    ap.add_argument("--limit_tgt", type=int, default=0)

    # targets
    ap.add_argument("--targets", default="results/chromatic_stats_sq512.json",
                    help="measured mBRSET statistics; must be measured at "
                         "--presize/--size, see scripts/76_measure_chromatic.py")
    ap.add_argument("--allow_pitch_mismatch", action="store_true")
    ap.add_argument("--max_artifact", type=float, default=-1.0,
                    help="Cap on spot and hole amplitude. Negative keeps the published 0.40 "
                         "ceiling. Fitted freely, hole_amp pins at that ceiling and paints dark "
                         "blobs that are visually indistinguishable from blot haemorrhages, "
                         "because they are the cheapest way to lower brightness (z +0.34) and "
                         "raise contrast (z -0.32) at once. A translated image keeps its SOURCE "
                         "label, so that manufactures mislabelled positives.")
    ap.add_argument("--init_phys", default="results/physics_v2_fit.json",
                    help="stage-1 init from script 79's fit; falls back to "
                         "PhysicalAcquisition.FITTED if absent")

    # stage 2 geometry and bounds
    ap.add_argument("--eps_lf", type=float, default=0.15)
    ap.add_argument("--eps_hf", type=float, default=0.02)
    ap.add_argument("--hf_mode", default="noise", choices=["noise", "residual", "off"])
    ap.add_argument("--lf_stride", type=int, default=16)
    ap.add_argument("--lf_width", type=int, default=48)
    ap.add_argument("--lf_depth", type=int, default=3)
    ap.add_argument("--hf_width", type=int, default=16)
    ap.add_argument("--mask_erode", type=int, default=4)
    ap.add_argument("--mask_taper", type=int, default=16)

    # loss weights
    ap.add_argument("--w_stat", type=float, default=1.0)
    ap.add_argument("--w_style", type=float, default=10.0,
                    help="Gram losses are of order 1e-3 at these normalisations, "
                         "so 10.0 puts the style term within an order of "
                         "magnitude of the statistic term at init")
    ap.add_argument("--w_adv", type=float, default=0.0,
                    help="0 disables the discriminator entirely; this is the "
                         "GAN flag and it is off by default for stability")
    ap.add_argument("--w_res_lf", type=float, default=0.02)
    ap.add_argument("--w_res_hf", type=float, default=0.10)
    ap.add_argument("--res_mode", default="l2", choices=["l2", "l1_down"])
    ap.add_argument("--res_down", type=int, default=8)
    ap.add_argument("--var_weight", type=float, default=1.0,
                    help="weight on the across-image spread term inside "
                         "script 79's distance()")

    # style / adversarial support
    ap.add_argument("--style_crop", type=int, default=128)
    ap.add_argument("--n_crops", type=int, default=2)
    ap.add_argument("--style_layers", default="0,1,2")
    ap.add_argument("--vgg", default="vgg16.tv_in1k")
    ap.add_argument("--d_width", type=int, default=32)
    ap.add_argument("--lr_d", type=float, default=1e-4)

    # optimisation
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr_phys", type=float, default=0.02,
                    help="stage 1's 19 parameters are sigmoid-reparameterised, "
                         "so they need the ~0.05 scale script 79 fits at; a "
                         "network learning rate leaves them frozen in practice")
    ap.add_argument("--lr_refine", type=float, default=2e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--freeze_phys", action="store_true",
                    help="ablation only: breaks the joint-training claim")
    ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"],
                    help="applies to VGG and the discriminator only; the "
                         "translator and the statistics stay fp32")
    ap.add_argument("--seed", type=int, default=0)

    # bookkeeping
    ap.add_argument("--task", default="translator_v1")
    ap.add_argument("--output_dir", default="results")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--save_panel", action="store_true",
                    help="writes a qualitative PNG containing PATIENT IMAGES to "
                         "a gitignored path; never commit the output")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def check_pitch(targets, a):
    meta = targets.get("_meta", {})
    tp = meta.get("presize")
    # scripts/76 records the pitch two different ways. The --size path writes
    # `size` and leaves `crop` at 0; the --presize/--crop path (the one that
    # reproduces the trainer) writes presize and crop and leaves `size` at 0.
    # The EFFECTIVE pitch is the crop whenever a crop was taken, so read that
    # first. Comparing `size` alone rejects correctly-measured targets.
    ts = meta.get("crop") or meta.get("size")
    if ts is None and meta.get("size") is None and meta.get("crop") is None:
        print("[warn] targets JSON has no size metadata at all; cannot verify pitch")
        return
    if not ts:
        # Both fields present and zero means the targets were measured at NATIVE
        # resolution, which has no single pixel pitch and can never match a
        # fixed training pitch. Treat it as a mismatch, not as missing data.
        ts = "native"
    if (ts, tp) != (a.size, a.presize):
        msg = (f"targets were measured at presize={tp} size={ts} but training "
               f"runs at presize={a.presize} size={a.size}. Laplacian variance "
               f"is strongly scale dependent (BRSET 0.000078 native vs "
               f"0.000257 at 512), so the sharpness and *_lapvar targets would "
               f"be wrong by several times. Re-measure with "
               f"scripts/76_measure_chromatic.py at the training pitch.")
        if not a.allow_pitch_mismatch:
            raise SystemExit("[fatal] " + msg)
        print("[warn] " + msg)


def build_model(a, device):
    init = None
    p = Path(a.init_phys)
    try:
        init = json.load(open(p)).get("params")
        print(f"stage 1 initialised from {p}")
    except Exception as e:
        # A warm start is a convenience, not a requirement: PhysicalAcquisition
        # already defaults to the fitted values. Failing the run because the fit
        # has not landed yet would be the wrong trade.
        print(f"[warn] no stage-1 warm start from {p} ({type(e).__name__}); "
              f"starting at PhysicalAcquisition.FITTED. Run "
              f"scripts/80_fit_physics_v2.slurm for a fitted init.")
    phys = P.PhysicalAcquisition(init=init, learnable=not a.freeze_phys,
                                 max_artifact=(a.max_artifact if a.max_artifact >= 0 else None))
    ref = R.ResidualRefiner(lf_stride=a.lf_stride, lf_width=a.lf_width,
                            lf_depth=a.lf_depth, hf_width=a.hf_width,
                            eps_lf=a.eps_lf, eps_hf=a.eps_hf,
                            hf_mode=a.hf_mode, mask_erode=a.mask_erode,
                            mask_taper=a.mask_taper)

    b = ref.max_adjacent_step()
    print(f"refiner {ref.n_params():,} params; worst-case adjacent-pixel edit: "
          f"deterministic {b['deterministic_total']:.4f}, stochastic "
          f"{b['hf_stochastic']:.4f}, lesion floor {R.LESION_CONTRAST_FLOOR}")
    if b["deterministic_total"] >= R.LESION_CONTRAST_FLOOR:
        print(f"[WARN] the DETERMINISTIC worst case {b['deterministic_total']:.4f} "
              f"reaches the {R.LESION_CONTRAST_FLOOR} lesion contrast floor, so "
              f"this configuration can in principle fill in a microaneurysm and "
              f"mislabel it. For hf_mode=residual use "
              f"--eps_hf {ref.max_safe_eps_hf_residual():.4f} or smaller.")
    return R.Translator(phys=phys, refiner=ref).to(device)


def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[a.amp]
    out_dir = Path(a.output_dir) / a.task
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device {device}  task {a.task}  objective from {_OBJ_SRC}")
    print(f"fitting {len(FIT_KEYS)} statistics: {FIT_KEYS}")

    targets = json.load(open(a.targets))
    check_pitch(targets, a)

    model = build_model(a, device)
    vgg = None
    if a.w_style > 0:
        layers = tuple(int(s) for s in a.style_layers.split(","))
        vgg = VGGGram(layers=layers, model=a.vgg).to(device)
        print(f"style extractor {a.vgg} layers {layers} channels {vgg.channels}")
    disc = None
    if a.w_adv > 0:
        disc = R.PatchGAN(width=a.d_width).to(device)
        print(f"[note] adversarial term ON at w_adv={a.w_adv}; this is the "
              f"unstable path, watch the loss trace")

    groups = [{"params": list(model.refiner.parameters()), "lr": a.lr_refine}]
    if not a.freeze_phys:
        groups.append({"params": [model.phys.raw], "lr": a.lr_phys})
    opt = torch.optim.Adam(groups)
    opt_d = torch.optim.Adam(disc.parameters(), lr=a.lr_d) if disc else None

    ds_s = FundusDir(a.src_train, a.presize, a.size, a.limit_src, a.seed)
    ds_t = FundusDir(a.tgt_train, a.presize, a.size, a.limit_tgt, a.seed)
    ds_v = FundusDir(a.src_val, a.presize, a.size, a.n_eval, a.seed + 1)
    print(f"source train {len(ds_s)}, target train {len(ds_t)}, "
          f"HELD-OUT source val {len(ds_v)}")
    kw = dict(batch_size=a.batch_size, num_workers=a.num_workers,
              collate_fn=collate_skip_none, drop_last=True, pin_memory=True)
    it_s = infinite(torch.utils.data.DataLoader(ds_s, shuffle=True, **kw))
    it_t = infinite(torch.utils.data.DataLoader(ds_t, shuffle=True, **kw))
    dl_v = torch.utils.data.DataLoader(ds_v, batch_size=a.batch_size,
                                       num_workers=a.num_workers,
                                       collate_fn=collate_skip_none)

    hist, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        x_s = next(it_s).to(device, non_blocking=True)
        x_t = next(it_t).to(device, non_blocking=True)

        # Discriminator first, on a detached generator output, so D never pushes
        # gradient back into the translator through its own update.
        if disc is not None:
            with torch.no_grad():
                o = model(x_s)
            m_s, m_t = P.fundus_mask(x_s), P.fundus_mask(x_t)
            fc = inner_crops(o, m_s, a.style_crop, a.n_crops)
            rc = inner_crops(x_t, m_t, a.style_crop, a.n_crops)
            opt_d.zero_grad(set_to_none=True)
            ld = lsgan_d_loss(disc, rc, fc)
            ld.backward()
            opt_d.step()

        opt.zero_grad(set_to_none=True)
        total, items, out, parts = translator_losses(
            model, x_s, x_t, targets, a, vgg=vgg, disc=disc, amp_dtype=amp_dtype)
        total.backward()
        if a.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], a.grad_clip)
        opt.step()

        if step % a.log_every == 0 or step == 1:
            r = {k: float(v) for k, v in items.items()}
            r["step"] = step
            r["res_lf_rms"] = float(parts["r_lf"].pow(2).mean().sqrt())
            r["res_hf_rms"] = float(parts["r_hf"].pow(2).mean().sqrt())
            hist.append(r)
            print(f"step {step:5d}  total {r['total']:9.3f}  stat {r['stat']:9.3f}  "
                  f"style {r['style']:.3e}  adv {r['adv']:.3f}  "
                  f"res_lf {r['res_lf']:.4f}/{r['res_lf_rms']:.4f}  "
                  f"res_hf {r['res_hf']:.4f}/{r['res_hf_rms']:.4f}  "
                  f"{(time.time() - t0) / step:.2f}s/step", flush=True)

        if step % a.eval_every == 0 or step == a.steps:
            z = evaluate(model, dl_v, targets, device, step, a)
            save(model, a, out_dir, hist, z, step)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min; artifacts in {out_dir}")


@torch.no_grad()
def evaluate(model, loader, targets, device, step, a):
    """Held-out statistic report on SOURCE VAL images, which the translator has
    never been fitted on. Script 79 established this protocol and the number is
    not comparable to anything measured on the fitting subset."""
    model.eval()
    acc, nimg = {}, 0
    for x in loader:
        if x is None:
            continue
        x = x.to(device)
        out, parts = model(x, return_parts=True)
        s = P.image_stats_torch(out, mask=parts["mask"])
        for k, v in s.items():
            acc.setdefault(k, []).append(v.cpu())
        nimg += x.shape[0]
    model.train()
    if not nimg:
        print("[warn] evaluation set empty")
        return {}
    s = {k: torch.cat(v) for k, v in acc.items()}
    print(f"\n--- held-out source val, step {step}, n={nimg}")
    print(f"{'statistic':12s} {'w':>5s} {'got':>11s} {'target':>11s} {'z':>8s}")
    z2 = 0.0
    for k in FIT_KEYS:
        g = float(s[k].mean())
        t = targets[k]["mbrset_mean"]
        sd = max(targets[k]["mbrset_std"], 1e-6)
        z = (g - t) / sd
        z2 += STAT_WEIGHTS.get(k, 1.0) * z * z
        print(f"{k:12s} {STAT_WEIGHTS.get(k, 1.0):5.2f} {g:11.5f} {t:11.5f} {z:+8.2f}")
    for k in DIAG_KEYS:
        if k in s and k in targets:
            print(f"{k:12s} {'diag':>5s} {float(s[k].mean()):11.5f} "
                  f"{targets[k]['mbrset_mean']:11.5f}")
    print(f"{'WEIGHTED z^2':12s} {'':>5s} {z2:11.3f}\n", flush=True)
    return {"step": step, "n": nimg, "weighted_z2": z2,
            "stats": {k: float(s[k].mean()) for k in s}}


def save(model, a, out_dir, hist, z, step):
    p = model.phys.params()
    ck = {"refiner": model.refiner.state_dict(),
          "phys_raw": model.phys.raw.detach().cpu(),
          "phys_params": {k: float(v) for k, v in p.items()},
          "args": vars(a), "step": step}
    # .pth, not .pt: results/**/*.pth is gitignored, so a checkpoint trained on
    # credentialed data cannot be committed by accident.
    torch.save(ck, out_dir / "translator_last.pth")
    json.dump({"history": hist, "eval": z,
               "phys_params": ck["phys_params"],
               "stat_weights": STAT_WEIGHTS, "fit_keys": FIT_KEYS,
               "objective_source": _OBJ_SRC},
              open(out_dir / "train_log.json", "w"), indent=1)
    pinned = [k for k in model.phys.KEYS
              if not (0.02 < (ck["phys_params"][k] - model.phys.RANGES[k][0])
                      / (model.phys.RANGES[k][1] - model.phys.RANGES[k][0]) < 0.98)]
    if pinned:
        print(f"stage-1 parameters pinned at a bound: {pinned}")
    if a.save_panel:
        save_panel(model, a, out_dir, step)


@torch.no_grad()
def save_panel(model, a, out_dir, step):
    """Qualitative panel. CONTAINS PATIENT IMAGES: the path is gitignored and the
    file must never be committed or attached to anything outside the DUA."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds_s = FundusDir(a.src_val, a.presize, a.size, 3, a.seed + 7)
    ds_t = FundusDir(a.tgt_train, a.presize, a.size, 3, a.seed + 7)
    rows = []
    for i in range(min(3, len(ds_s))):
        x = ds_s[i]
        if x is None:
            continue
        x = x.unsqueeze(0).to(next(model.parameters()).device)
        out, parts = model(x, return_parts=True)
        t = ds_t[i]
        rows.append([x[0], parts["x_phys"][0], out[0],
                     t if t is not None else torch.zeros_like(x[0])])
    if not rows:
        return
    titles = ["BRSET source", "stage 1 (physics)", "stage 1+2", "real mBRSET"]
    fig, ax = plt.subplots(len(rows), 4, figsize=(13, 3.3 * len(rows)))
    ax = np.atleast_2d(ax)
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            ax[i, j].imshow(im.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(titles[j], fontsize=10)
    fig.tight_layout()
    f = out_dir / f"panel_step{step}.png"
    fig.savefig(f, dpi=110)
    plt.close(fig)
    print(f"wrote {f} (patient images, gitignored, do not commit)")


# ---------------------------------------------------------------------------
# Self-tests. Synthetic tensors only; safe on the login node.
# ---------------------------------------------------------------------------

def _fake_targets(x):
    """Target dict in the layout script 76 writes, derived from a synthetic
    batch so the test needs no file and no decoded image."""
    s = P.image_stats_torch(x)
    t = {}
    for k, v in s.items():
        t[k] = {"mbrset_mean": float(v.mean()),
                "mbrset_std": max(float(v.std(unbiased=False)), 1e-4),
                "brset_mean": float(v.mean()), "brset_std": 1e-4}
    t["_meta"] = {"presize": 0, "size": int(x.shape[-1])}
    return t


def _selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    torch.manual_seed(0)
    size = 128
    x_s, disc = R._synthetic_fundus(b=4, h=size, w=size, seed=0)
    # A "target domain" batch: more blue, more per-channel noise, which is the
    # measured direction (blue fraction 1.43x, blue std 2.86x, green 1.60x).
    x_t, _ = R._synthetic_fundus(b=4, h=size, w=size, seed=1)
    x_t = (x_t * torch.tensor([0.95, 0.90, 1.45]).view(1, 3, 1, 1)
           + torch.randn_like(x_t) * torch.tensor([0.004, 0.012, 0.022]).view(1, 3, 1, 1))
    x_t = (x_t * disc).clamp(0, 1)
    targets = _fake_targets(x_t)
    print(f"synthetic source {tuple(x_s.shape)} and target {tuple(x_t.shape)}; "
          f"objective from {_OBJ_SRC}")
    print(f"fitting {len(FIT_KEYS)} statistics")

    a = build_args(["--size", str(size), "--presize", "0",
                    "--style_crop", "64", "--n_crops", "2",
                    "--targets", "__none__", "--init_phys", "__none__"])

    # 1. the inside-the-fundus crops really are inside
    print("\n--- inside-the-fundus crops")
    cr = inner_crops(disc.expand(4, 3, size, size).contiguous(), disc, 64, 3)
    print(f"  {tuple(cr.shape)} crops from a {size}px disc of radius "
          f"{math.sqrt(float(disc.sum()) / math.pi):.1f}px")
    check("every crop pixel lies inside the field of view",
          float(cr.min()) == 1.0, f"min mask value in a crop {float(cr.min()):.3f}")

    # 2. VGG style extractor loads from the local HF cache
    print("\n--- frozen style extractor")
    vgg = None
    try:
        vgg = VGGGram(layers=(0, 1, 2), model=a.vgg)
        n = sum(p.numel() for p in vgg.parameters())
        sl = float(vgg.style_loss(x_s, x_t))
        sl0 = float(vgg.style_loss(x_t, x_t))
        print(f"  {a.vgg} loaded, {n / 1e6:.1f}M frozen params, "
              f"channels {vgg.channels}")
        check("no VGG parameter is trainable",
              all(not p.requires_grad for p in vgg.parameters()))
        check("style loss is positive across domains and ~0 within one",
              sl > 0 and sl0 < sl, f"cross {sl:.3e} vs self {sl0:.3e}")
    except Exception as e:
        print(f"  [SKIP] could not load {a.vgg}: {e}")
        check("style extractor available (prefetch it on the login node)", False)

    # 3. every loss term is finite, and gradients reach BOTH stages through the
    # full objective, not just through a toy scalar.
    for w_adv in (0.0, 1.0):
        print(f"\n--- full objective, w_adv={w_adv}")
        torch.manual_seed(0)
        a.w_adv = w_adv
        model = build_model(a, "cpu")
        disc_ = R.PatchGAN(width=16) if w_adv > 0 else None
        model.zero_grad()
        total, items, out, parts = translator_losses(
            model, x_s, x_t, targets, a, vgg=vgg, disc=disc_)
        print("  " + "  ".join(f"{k} {float(v):.4e}" for k, v in items.items()))
        check("every loss term is finite",
              all(torch.isfinite(v).all() for v in items.values()))
        check("every loss term is non-negative",
              all(float(v) >= 0 for k, v in items.items()))
        total.backward()
        gp = model.phys.raw.grad
        check("gradient reaches stage 1 through the full objective",
              gp is not None and float(gp.abs().max()) > 0,
              f"|g| max {float(gp.abs().max()):.3e}, "
              f"{int((gp.abs() > 0).sum())}/{gp.numel()} nonzero")
        nog = [n for n, p in model.refiner.named_parameters()
               if p.grad is None or float(p.grad.abs().max()) == 0.0]
        check("gradient reaches every stage 2 parameter", len(nog) == 0,
              f"zero-grad: {nog if nog else 'none'}")
        if disc_ is not None:
            disc_.zero_grad()
            lsgan_d_loss(disc_, x_t, out.detach()).backward()
            ndog = [n for n, p in disc_.named_parameters()
                    if p.grad is None or float(p.grad.abs().max()) == 0.0]
            check("gradient reaches every discriminator parameter", len(ndog) == 0,
                  f"zero-grad: {ndog if ndog else 'none'}")
    a.w_adv = 0.0

    # 4. what the residual penalty actually does, established rather than
    # asserted. An earlier version of this test compared a dense shift to a
    # sparse blob of equal amplitude and found the SAME ratio under every mode,
    # because for a one-signed residual all three reduce to total mass over the
    # mask. The real difference is the MARGINAL cost in amplitude, which is what
    # the optimiser responds to.
    print("\n--- residual penalty: marginal cost in amplitude")
    w = disc
    frac = float(w.mean())

    def l1_full(r):
        return (r.abs() * w).sum() / (w.sum() * r.shape[1])

    def marginal(fn, amp):
        """d(penalty)/d(amplitude) for a dense residual at this amplitude."""
        aa = torch.tensor(float(amp), requires_grad=True)
        pen = fn(aa * torch.ones(1, 3, size, size))
        pen.backward()
        return float(aa.grad), float(pen)

    eps_lf = a.eps_lf
    print(f"{'amplitude':>10s} {'d/da L2':>10s} {'d/da L1':>10s} {'L1/L2':>8s}")
    ratios = []
    for amp in (0.005, 0.01, 0.02, 0.05):
        g2, _ = marginal(lambda r: residual_penalty(r, w, eps_lf, "l2", a.res_down), amp)
        g1, _ = marginal(l1_full, amp)
        ratios.append(g1 / max(g2, 1e-30))
        print(f"{amp:10.3f} {g2:10.4f} {g1:10.4f} {ratios[-1]:8.1f}x")
    print("  READ: L1's marginal cost is constant in amplitude, L2's falls to "
          "zero.\n  The dense low-amplitude chroma shift the measurement asks "
          "for (blue 1.43x,\n  observed residual rms ~0.02) is exactly where L1 "
          "charges most relative to L2,\n  which is why plain L1 is not offered "
          "as the default.")
    check("L1 charges more at the margin than L2 at every realistic amplitude, "
          "and increasingly so as the residual gets smaller",
          all(r > 1.0 for r in ratios) and ratios[0] > ratios[-1],
          f"ratios {[round(r, 1) for r in ratios]}")

    # The other half of the brief's point, and it cuts against relying on the
    # penalty at all: a lesion-sized blob is nearly free under EVERY mode,
    # because every mode normalises by mask area. So no residual penalty
    # protects lesions. The bound does. This is worth having measured because it
    # is the argument for why eps_hf, not w_res_hf, is the safety mechanism.
    dense = torch.full((1, 3, size, size), 0.05)
    sparse = torch.zeros_like(dense)
    sparse[:, :, size // 2 - 1:size // 2 + 2, size // 2 - 1:size // 2 + 2] = 0.05
    print(f"\n  cost of a 3x3 lesion-sized blob as a fraction of a dense shift "
          f"of equal amplitude:")
    for name, fn in (("l2", lambda r: residual_penalty(r, w, eps_lf, "l2", a.res_down)),
                     ("l1_down", lambda r: residual_penalty(r, w, eps_lf, "l1_down", a.res_down)),
                     ("l1_full", l1_full)):
        fr = float(fn(sparse)) / max(float(fn(dense)), 1e-30)
        print(f"    {name:8s} {100 * fr:.4f}%")
    blob_frac = float(l1_full(sparse)) / max(float(l1_full(dense)), 1e-30)
    check("a lesion-sized blob costs under 0.1% of a dense shift, so the "
          "penalty cannot be the lesion safeguard",
          blob_frac < 1e-3, f"{100 * blob_frac:.4f}% of the dense cost; "
                            f"the eps bounds are what protect lesions")

    # And what l1_down is and is not. Average pooling preserves mass, so for a
    # one-signed residual l1_down equals l1_full exactly; it differs only where
    # the residual alternates sign inside a pool, i.e. it stops charging for
    # fine texture. Useful to know before anyone reaches for it as a stronger
    # texture deterrent, which it is not.
    alt = torch.zeros_like(dense)
    alt[:, :, ::2, :] = 0.05
    alt[:, :, 1::2, :] = -0.05
    one = float(residual_penalty(dense, w, eps_lf, "l1_down", a.res_down)) \
        / max(float(l1_full(dense)), 1e-30)
    altr = float(residual_penalty(alt, w, eps_lf, "l1_down", a.res_down)) \
        / max(float(l1_full(alt)), 1e-30)
    print(f"\n  l1_down / l1_full: one-signed {one:.3f}, sign-alternating {altr:.3f}")
    check("l1_down equals l1_full on a one-signed residual (mass preserving)",
          abs(one - 1.0 / eps_lf * eps_lf) < 1e-3 or abs(one - 1.0) < 1e-3,
          f"{one:.4f}")
    check("l1_down nearly stops charging for sign-alternating fine texture, so "
          "it is NOT a stronger texture deterrent", altr < 0.05, f"{altr:.4f}")

    # 5. THE INTEGRATION TEST. The bound, the mask and the range are supposed to
    # hold at every step of training, not only at initialisation, so take real
    # optimiser steps and re-assert all of them afterwards.
    print("\n--- guarantees survive 25 optimiser steps")
    torch.manual_seed(0)
    model = build_model(a, "cpu")
    opt = torch.optim.Adam([
        {"params": list(model.refiner.parameters()), "lr": 1e-2},   # 50x normal,
        {"params": [model.phys.raw], "lr": 0.1}])                   # to stress it
    first = last = None
    for i in range(25):
        opt.zero_grad()
        total, items, out, parts = translator_losses(
            model, x_s, x_t, targets, a, vgg=vgg, disc=None)
        total.backward()
        opt.step()
        if i == 0:
            first = float(items["stat"])
        last = float(items["stat"])
    out, parts = model(x_s, return_parts=True)
    x1, res, m, meff = parts["x_phys"], parts["residual"], parts["mask"], parts["mask_eff"]
    eps = model.refiner.eps_total()
    print(f"  statistic loss {first:.3f} -> {last:.3f}; "
          f"residual rms lf {float(parts['r_lf'].pow(2).mean().sqrt()):.5f} "
          f"hf {float(parts['r_hf'].pow(2).mean().sqrt()):.5f}")
    check("statistic loss decreased", last < first, f"{first:.3f} -> {last:.3f}")
    check("output still finite and in [0,1]",
          bool(torch.isfinite(out).all() and (out >= 0).all() and (out <= 1).all()))
    check(f"residual bound still respected (eps_total {eps:.3f})",
          float((out - x1).abs().max()) <= eps + 1e-5,
          f"max |out - x_phys| {float((out - x1).abs().max()):.6f}")
    check("residual still zero outside the fundus mask",
          float((res.abs() * (1.0 - m)).max()) == 0.0)
    check("residual still zero where the residual weight is zero",
          float((res.abs() * (meff == 0).float()).max()) == 0.0)
    check("stage 1 parameters still inside their physical ranges",
          all(model.phys.RANGES[k][0] - 1e-6 <= float(v) <= model.phys.RANGES[k][1] + 1e-6
              for k, v in model.phys.params().items()))

    # 6. the pitch guard, since a silent scale mismatch is the bug script 79
    # was just corrected for and this script must not reintroduce it.
    print("\n--- target pitch guard")
    try:
        check_pitch({"_meta": {"presize": 0, "size": 1024}},
                    build_args(["--size", "512", "--presize", "560"]))
        check("mismatched target pitch is refused", False)
    except SystemExit:
        check("mismatched target pitch is refused", True)
    try:
        check_pitch({"_meta": {"presize": 560, "size": 512}},
                    build_args(["--size", "512", "--presize", "560"]))
        check("matching target pitch is accepted", True)
    except SystemExit:
        check("matching target pitch is accepted", False)

    print(f"\n{'ALL SELF-TESTS PASSED' if ok else 'SELF-TESTS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    args = build_args()
    if args.selftest:
        raise SystemExit(_selftest())
    main(args)
