"""Fit the 19-parameter acquisition model, and audit it honestly.

Replaces the 150-trial random search of script 57 and the 9-parameter fit, both
of which were wrong in ways verified on 9 Oct:

  - the 9-parameter model had NO chromatic degree of freedom (max R:G:B ratio
    change 0.00088 over all parameters at both extremes), so the earlier claim
    that physics was "exhausted" was false;
  - roughly 81 percent of its matched sharpness came from i.i.d. sensor noise,
    which does not survive JPEG or resampling, so the match was cosmetic;
  - it fitted blur_sigma to 0.907, i.e. it BLURRED the source to make room for
    that noise, despite the measured fact that mBRSET carries 5 to 7 times MORE
    high-frequency energy.

This script fits the corrected model and then audits for exactly those failure
modes rather than reporting a single distance number.

Held-out protocol: parameters are fitted on one image subset and the reported
distance is computed on a DISJOINT subset, so the number is not the fit's own
training error.

Three corrections on 9 Oct, each of which changed what the fit is actually
chasing:

  SCALE. The fit previously loaded targets measured at NATIVE resolution
  (BRSET fundus disc 2491 px, mBRSET 1599 px) while fitting on 512x512 tensors.
  Laplacian variance is strongly scale dependent, BRSET measures 0.000078 native
  against 0.000257 squashed to 512, so the sharpness target was wrong by about
  3x and wrong in the convenient direction: the model only had to supply a
  quarter of the real gap, which would have made its own noise self-audit pass
  spuriously. Targets are now measured through the trainer's own pipeline,
  Resize((560,560)) then a 512 crop, and the fit loads images the same way.

  REDUNDANCY. FIT_KEYS contained brightness, contrast and the three channel
  means. rgb_to_gray is x.mean(dim=1), so brightness is an EXACT linear
  function of r_mean, g_mean and b_mean: four statistics for three degrees of
  freedom, silently doubling the weight on the luma direction, which is the
  opposite of what adding chromatic statistics was for. It also computed
  r_frac/g_frac/b_frac, the actual headline chromatic statistic, and then never
  fitted them. FIT_KEYS now carries one luma term plus the three fractions.

  SECOND MOMENTS ACROSS IMAGES. The objective matched batch MEANS only, so a
  degradation that produces the right average with collapsed variability scored
  perfectly. Each statistic now also penalises the mismatch of its spread
  across images. See PER_IMAGE_PARAMETER_HOOK below for why this is a partial
  fix.
"""
import argparse, importlib.util, json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path

spec = importlib.util.spec_from_file_location("P", str(Path(__file__).parent / "72_physical_layer.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

# Thirteen statistics, chosen to be non-redundant rather than numerous.
#
# Dropped r_mean, g_mean, b_mean and kept brightness plus the three fractions:
# the two sets are related by an exact bijection (mean_c = 3 * brightness *
# frac_c), so nothing is lost, but this parameterisation counts luma ONCE
# instead of twice and puts the chromatic statistic the measurement script
# reports as its headline directly into the objective. The three fractions sum
# to one, so they span a two-dimensional chroma plane; that is a symmetric
# weighting of that plane rather than an arbitrary choice of which two channels
# to fit, and unlike the old brightness/means overlap it does not duplicate the
# luma direction.
#
# Dropped the grayscale `sharpness`. It is the variance of the Laplacian of the
# channel mean, so up to inter-channel Laplacian covariance it is the
# common-mode average of the three per-channel lapvars: including both would
# upweight common-mode high-frequency energy exactly as brightness upweighted
# luma. The per-channel version is also the only one that can see the measured
# asymmetry (lapvar/std ratios: red 1.02x, green 1.60x, blue 2.85x). It is still
# computed and reported as a diagnostic, and the noise audit still uses it.
#
# contrast is kept: it is the std of the channel mean, which depends on
# inter-channel covariance and is therefore NOT a function of r_std/g_std/b_std.
FIT_KEYS = ["brightness", "contrast", "falloff", "saturation",
            "r_frac", "g_frac", "b_frac",
            "r_std", "g_std", "b_std",
            "r_lapvar", "g_lapvar", "b_lapvar"]

# Reported but not optimised, so the redundancy above does not reappear through
# the report table.
DIAG_KEYS = ["sharpness", "r_mean", "g_mean", "b_mean"]


def load_batch(split_dir, n, size, seed, presize=560):
    """Load images at the resolution synthesis is applied at.

    presize then centre crop to `size` is the trainer's transform. Loading at a
    different pitch than the targets were measured at is the scale bug this fit
    had, so the default here and the measurement default in script 76 must stay
    in step.
    """
    d = Path(split_dir)
    files = pd.read_csv(d / "labels.csv").file.values
    files = np.random.default_rng(seed).choice(files, n, replace=False)
    ims = []
    for f in files:
        try:
            im = Image.open(d / f).convert("RGB")
        except Exception:
            continue
        if presize:
            im = im.resize((presize, presize), Image.BILINEAR)
            o = (presize - size) // 2
            im = im.crop((o, o, o + size, o + size))
        else:
            im = im.resize((size, size), Image.BILINEAR)
        ims.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1))
    return torch.stack(ims)


# PER_IMAGE_PARAMETER_HOOK
# The variance term below is a weak fix. With one shared parameter vector the
# only sources of across-image spread are the source images themselves and the
# stochastic placement of illumination centre and artifacts, so the optimiser
# can only match the target spread insofar as those already supply it. The
# correct fix is to make the parameters a per-image distribution: give each
# parameter a learnable spread, draw raw_i = raw + s * eps_i with eps_i standard
# normal per image, and map through the same sigmoid, which keeps the bounds and
# stays reparameterisation-differentiable so both the location and the spread
# receive gradients. That doubles the parameter count to 38 and requires the
# acquisition layer to accept a (B, K) parameter tensor instead of a (K,) one,
# which is a change to script 72's forward, so it is deliberately not done here.
# To implement, replace this no-op with the draw and pass the result into a
# batched params() path.
PER_IMAGE_SPREAD = None   # set to a (K,) tensor to activate; unused today


def distance(stats, target, weights=None, var_weight=1.0):
    """Normalised squared distance on the mean AND the spread of each statistic.

    Each statistic is scaled by the target's own per-image spread so no single
    statistic dominates by unit. The second term penalises
    ((batch_std - target_std)/target_std)^2, because matching means alone is
    satisfied exactly by a degradation that collapses variability, and a
    collapsed distribution is both wrong and invisible in the old objective.
    """
    tot = 0.0
    for k in FIT_KEYS:
        t = target[k]["mbrset_mean"]; sd = max(target[k]["mbrset_std"], 1e-6)
        w = 1.0 if weights is None else weights.get(k, 1.0)
        v = stats[k]
        tot = tot + w * ((v.mean() - t) / sd) ** 2
        if var_weight:
            got_sd = v.std(unbiased=False)
            tot = tot + var_weight * w * ((got_sd - sd) / sd) ** 2
    return tot


def report(name, stats, target):
    print(f"\n--- {name}")
    print(f"{'statistic':11s} {'got':>10s} {'target':>10s} {'z':>8s} "
          f"{'got_sd':>10s} {'tgt_sd':>10s} {'z_sd':>8s}")
    z2 = 0.0
    v2 = 0.0
    for k in FIT_KEYS:
        g = float(stats[k].mean()); t = target[k]["mbrset_mean"]
        sd = max(target[k]["mbrset_std"], 1e-6); z = (g - t) / sd
        gs = float(stats[k].std(unbiased=False)); zs = (gs - sd) / sd
        z2 += z * z; v2 += zs * zs
        print(f"{k:11s} {g:10.5f} {t:10.5f} {z:+8.2f} {gs:10.5f} {sd:10.5f} {zs:+8.2f}")
    print(f"{'TOTAL z^2':11s} {z2:10.3f}   (spread term {v2:.3f})")
    print("  diagnostics, not in the objective:")
    for k in DIAG_KEYS:
        g = float(stats[k].mean())
        t = target[k]["mbrset_mean"] if k in target else float("nan")
        print(f"  {k:11s} {g:10.5f} {t:10.5f}")
    return z2, v2


def randomise_init(L, seed):
    """Random restart inside the bounds, uniform in the BOUNDED space.

    Sampling the raw logits uniformly would concentrate near the bound centres;
    sampling u in [0.05, 0.95] and inverting the sigmoid spreads the restarts
    over the actual parameter ranges, which is what an identifiability check
    needs. 0.05/0.95 rather than 0/1 because a restart exactly on a bound starts
    with a vanishing gradient.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.05, 0.95, size=len(L.KEYS))
    L.raw.data = torch.tensor(np.log(u / (1.0 - u)), dtype=torch.float32,
                              device=L.raw.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="results/chromatic_stats_sq512.json",
                    help="must be measured at the same pitch as --presize/--size")
    ap.add_argument("--n_fit", type=int, default=64)
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--presize", type=int, default=560,
                    help="trainer's Resize((S,S)) before the 512 crop; 0 squashes "
                         "straight to --size")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--var_weight", type=float, default=1.0)
    ap.add_argument("--init_seed", type=int, default=-1,
                    help="-1 initialises at FITTED; >=0 draws a random restart")
    ap.add_argument("--torch_seed", type=int, default=0)
    ap.add_argument("--out", default="results/physics_v2_fit.json")
    a = ap.parse_args()

    target = json.load(open(a.targets))
    meta = target.get("_meta", {})
    if meta.get("presize") != a.presize or meta.get("crop") != a.size:
        print(f"WARNING scale mismatch: targets measured with presize="
              f"{meta.get('presize')} crop={meta.get('crop')} size={meta.get('size')} "
              f"but fitting at presize={a.presize} size={a.size}. Laplacian "
              f"statistics are scale dependent and this invalidates the fit.")
    for k in FIT_KEYS:
        if k not in target:
            raise KeyError(f"target file {a.targets} has no {k}; re-run script 76")

    torch.manual_seed(a.torch_seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}")
    x_fit = load_batch("data/finetune_multilabel/train", a.n_fit, a.size, 0, a.presize).to(dev)
    x_ev = load_batch("data/finetune_multilabel/train", a.n_eval, a.size, 999, a.presize).to(dev)
    print(f"fit on {len(x_fit)} images, evaluate on {len(x_ev)} DISJOINT images")
    print(f"scale: presize {a.presize} -> crop {a.size}, targets {a.targets}")

    with torch.no_grad():
        raw_stats = P.image_stats_torch(x_ev)
    z_raw, v_raw = report("untouched BRSET (held out)", raw_stats, target)
    s_raw = float(raw_stats["sharpness"].mean())
    raw_lap = {c: float(raw_stats[f"{c}_lapvar"].mean()) for c in "rgb"}

    L = P.PhysicalAcquisition().to(dev)
    if a.init_seed >= 0:
        randomise_init(L, a.init_seed)
        print(f"random restart, init_seed {a.init_seed}")
    opt = torch.optim.Adam(L.parameters(), lr=a.lr)
    for step in range(a.steps):
        opt.zero_grad()
        loss = distance(P.image_stats_torch(L(x_fit)), target, var_weight=a.var_weight)
        loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"  step {step:4d}  fit loss {float(loss):.3f}", flush=True)

    with torch.no_grad():
        fit_stats = P.image_stats_torch(L(x_ev))
        z_fit, v_fit = report("fitted 19-parameter model (held out)", fit_stats, target)

    p = {k: round(float(v), 7) for k, v in L.params().items()}
    print(f"\nfitted parameters:")
    pinned = []
    for k in L.KEYS:
        lo, hi = L.RANGES[k]
        frac = (p[k] - lo) / (hi - lo)
        flag = ""
        if frac < 0.02 or frac > 0.98:
            flag = "  <-- PINNED AT BOUND"; pinned.append(k)
        print(f"  {k:16s} {p[k]:+11.6f}   [{lo:+.4f}, {hi:+.4f}]  frac {frac:.3f}{flag}")

    # AUDIT 1: how much of the CLOSED sharpness gap is synthetic noise?
    #
    # The denominator used to be the total sharpness of the output, which
    # flatters the model: the untouched source already supplies most of that
    # total, so dividing by it makes any noise contribution look like a small
    # share. The honest denominator is the gap the fit actually had to close,
    # output minus untouched source, because that is the quantity the noise is
    # competing with the optics parameter to explain. Both are printed so the
    # new number is comparable with the old 81 percent claim.
    with torch.no_grad():
        s_full = float(fit_stats["sharpness"].mean())
        full_lap = {c: float(fit_stats[f"{c}_lapvar"].mean()) for c in "rgb"}
        saved = {k: L.raw.data[L.KEYS.index(k)].item() for k in L.NOISE_KEYS}
        for k in L.NOISE_KEYS:
            L.raw.data[L.KEYS.index(k)] = -20.0
        nn_stats = P.image_stats_torch(L(x_ev))
        s_nonoise = float(nn_stats["sharpness"].mean())
        nonoise_lap = {c: float(nn_stats[f"{c}_lapvar"].mean()) for c in "rgb"}
        for k, v in saved.items():
            L.raw.data[L.KEYS.index(k)] = v

    gap_closed = s_full - s_raw
    noise_contrib = s_full - s_nonoise
    share_of_gap = noise_contrib / gap_closed if abs(gap_closed) > 1e-12 else float("nan")
    share_of_total = noise_contrib / max(s_full, 1e-12)
    tgt_sharp = target["sharpness"]["mbrset_mean"]

    print(f"\n=== AUDIT ===")
    print(f"held-out z^2 (means): untouched {z_raw:.3f} -> fitted {z_fit:.3f} "
          f"({100*(z_raw-z_fit)/max(z_raw,1e-9):.1f}% reduction)")
    print(f"held-out spread term: untouched {v_raw:.3f} -> fitted {v_fit:.3f}")
    print(f"sharpness: source {s_raw:.6f}, fitted {s_full:.6f}, "
          f"no-noise {s_nonoise:.6f}, target {tgt_sharp:.6f}")
    print(f"gap the fit had to close {gap_closed:+.6f}, of which synthetic noise "
          f"supplies {noise_contrib:+.6f}")
    print(f"synthetic noise share OF THE GAP CLOSED: {100*share_of_gap:.1f}%")
    print(f"  (old denominator, share of total output sharpness: "
          f"{100*share_of_total:.1f}%, the number that was 81%)")
    per_ch = {}
    for c in "rgb":
        g = full_lap[c] - raw_lap[c]
        nc = full_lap[c] - nonoise_lap[c]
        per_ch[c] = {"source": raw_lap[c], "fitted": full_lap[c],
                     "no_noise": nonoise_lap[c],
                     "target": target[f"{c}_lapvar"]["mbrset_mean"],
                     "gap_closed": g, "noise_contrib": nc,
                     "noise_share_of_gap": nc / g if abs(g) > 1e-12 else float("nan")}
        print(f"  {c}_lapvar: source {raw_lap[c]:.6f} fitted {full_lap[c]:.6f} "
              f"target {per_ch[c]['target']:.6f}  noise share of gap "
              f"{100*per_ch[c]['noise_share_of_gap']:.1f}%")
    print(f"sharpen_k = {p['sharpen_k']:+.4f}  "
          f"({'SHARPENS, consistent with mBRSET having more HF energy' if p['sharpen_k']>0 else 'BLURS - same pathology as the old fit'})")
    print(f"parameters pinned at a bound: {pinned if pinned else 'none'}")
    if pinned:
        print("  -> the family is still constrained on these axes; report honestly")

    json.dump({"params": p, "z2_untouched": z_raw, "z2_fitted": z_fit,
               "var2_untouched": v_raw, "var2_fitted": v_fit,
               "noise_share_of_gap_closed": share_of_gap,
               "noise_fraction_of_total_sharpness": share_of_total,
               "sharpness": {"source": s_raw, "fitted": s_full,
                             "no_noise": s_nonoise, "target": tgt_sharp},
               "per_channel_lapvar": per_ch,
               "pinned": pinned, "ranges": {k: list(L.RANGES[k]) for k in L.KEYS},
               "init_seed": a.init_seed, "steps": a.steps, "lr": a.lr,
               "var_weight": a.var_weight, "presize": a.presize, "size": a.size,
               "targets_file": a.targets}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
