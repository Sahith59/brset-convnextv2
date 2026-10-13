"""Measure the CHROMATIC gap between BRSET and mBRSET. Previously unmeasured.

Why this exists. The five statistics fitted so far (sharpness, brightness,
contrast, falloff, saturation) are ALL computed on grayscale or on
(max-min)/max. None of them carries chromatic information, so nothing in the
pipeline has ever had a gradient toward correct hue. Verified 9 Oct: sweeping
every stage-1 parameter to both extremes moves the R:G:B ratio by at most
0.00088. The acquisition model has no chromatic degree of freedom at all.

That invalidates the earlier claim that the physical model was "exhausted". It
was not exhausted, it was missing a colour parameter. This script measures the
target those new parameters have to hit.

Also addresses a scale confound. The earlier sharpness comparison varied output
resolution but always applied a fixed 3x3 Laplacian at matched PIXEL pitch, so
it never equalised cycles per retinal degree. BRSET's fundus circle is larger in
pixels than mBRSET's, which attenuates BRSET's measured Laplacian variance on a
power-law spectrum. --match_disc resizes each image so the fundus circle
diameter is equal before measuring, which is the honest comparison.

SCALE DECISION, 9 Oct. A target measured at one resolution and chased at another
is not a target. The native-resolution run measures BRSET with a 2491 px fundus
disc and mBRSET with a 1599 px disc, but the acquisition model is fitted on
512x512 tensors and the trainer consumes Resize((560,560)) then a 512 crop, so
the Laplacian statistics differ by about 3x between the two scales purely
through the sampling pitch. The measurement resolution is therefore pinned to
the one synthesis is applied at: --presize 560 --crop 512, which is the
trainer's own pipeline. The trainer's RandomCrop and the CenterCrop used here
differ only by a translation, which leaves the sampling pitch, and so every
statistic here, unchanged in expectation. Native and disc-matched runs are kept
because they answer a different question (what the cameras do), but they must
not be used as fit targets.

Also adds PER-CHANNEL Laplacian variance. The headline finding of this script is
that the high-frequency excess is channel-asymmetric (blue std ratio 2.85x, red
1.02x), yet every high-frequency statistic measured so far was grayscale, so the
fit had no target that could see the asymmetry it exists to reproduce.
"""
import argparse, json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from scipy import ndimage


def fundus_mask(gray, thresh=0.04):
    return gray > thresh


def disc_diameter(mask):
    """Diameter in pixels of the circular field of view."""
    cols = mask.any(axis=0); rows = mask.any(axis=1)
    if not cols.any() or not rows.any():
        return 0
    w = int(cols.nonzero()[0].ptp() + 1); h = int(rows.nonzero()[0].ptp() + 1)
    return max(w, h)


def stats(img, match_disc=None):
    a = np.asarray(img, dtype=np.float32) / 255.0
    g = a.mean(axis=2)
    m = fundus_mask(g)
    if m.sum() < 1000:
        return None
    if match_disc:
        d = disc_diameter(m)
        if d < 50:
            return None
        s = match_disc / d
        new = (max(8, int(round(a.shape[1] * s))), max(8, int(round(a.shape[0] * s))))
        img = img.resize(new, Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
        g = a.mean(axis=2); m = fundus_mask(g)
        if m.sum() < 1000:
            return None

    out = {}
    # CHROMATIC: per-channel moments inside the field of view. This is what the
    # new colour parameters must match, and it is what hue actually is.
    for i, c in enumerate("rgb"):
        out[f"{c}_mean"] = float(a[..., i][m].mean())
        out[f"{c}_std"] = float(a[..., i][m].std())
    # chromatic ratios, scale-invariant to overall brightness
    tot = sum(out[f"{c}_mean"] for c in "rgb") + 1e-9
    for c in "rgb":
        out[f"{c}_frac"] = out[f"{c}_mean"] / tot

    # the original five, retained so the new fit is comparable to the old one
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    inner = m & (r < 0.25 * min(h, w)); outer = m & (r > 0.39 * min(h, w))
    mx = a.max(axis=2); mn = a.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    out["sharpness"] = float(ndimage.laplace(g)[m].var())
    # PER-CHANNEL high-frequency energy, the target the per-channel noise
    # parameters are fitted against. Same operator as the grayscale sharpness
    # above, applied to each channel separately instead of to their mean, so the
    # two are directly comparable.
    for i, c in enumerate("rgb"):
        out[f"{c}_lapvar"] = float(ndimage.laplace(a[..., i])[m].var())
    out["brightness"] = float(g[m].mean())
    out["contrast"] = float(g[m].std())
    out["falloff"] = (float(g[inner].mean() - g[outer].mean())
                      if outer.sum() > 500 else np.nan)
    out["saturation"] = float(sat[m].mean())
    out["disc_px"] = float(disc_diameter(m))
    return out


def to_pipeline(im, size, presize, crop):
    """Bring a native image to the resolution synthesis is applied at.

    presize/crop reproduce the classifier's transform, Resize((S,S)) then a
    square crop, which is a squash: it ignores aspect ratio exactly as the
    trainer does. Doing anything else here would measure a target at a sampling
    pitch the fitted model never sees.
    """
    if presize:
        im = im.resize((presize, presize), Image.BILINEAR)
        if crop and crop < presize:
            o = (presize - crop) // 2
            im = im.crop((o, o, o + crop, o + crop))
        return im
    if size:
        im = im.resize((size, size), Image.BILINEAR)
    return im


def run(split_dir, n, size, match_disc, seed=0, presize=0, crop=0):
    d = Path(split_dir)
    files = pd.read_csv(d / "labels.csv").file.values
    rng = np.random.default_rng(seed)
    files = rng.choice(files, min(n, len(files)), replace=False)
    rows = []
    for f in files:
        try:
            im = Image.open(d / f).convert("RGB")
        except Exception:
            continue
        im = to_pipeline(im, size, presize, crop)
        s = stats(im, match_disc)
        if s:
            rows.append(s)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--size", type=int, default=0,
                   help="0 keeps native resolution")
    p.add_argument("--match_disc", type=int, default=0,
                   help="resize so the fundus circle diameter equals this many pixels")
    p.add_argument("--presize", type=int, default=0,
                   help="squash-resize to this square size first; 560 with --crop 512 "
                        "reproduces the trainer pipeline and is the only setting whose "
                        "output may be used as a fit target")
    p.add_argument("--crop", type=int, default=0,
                   help="centre crop after --presize")
    p.add_argument("--out", default="results/chromatic_stats.json")
    a = p.parse_args()

    B = run("data/finetune_multilabel/train", a.n, a.size, a.match_disc, 0, a.presize, a.crop)
    M = run("data/finetune_mbrset_multilabel/train", a.n, a.size, a.match_disc, 1, a.presize, a.crop)
    scale = (f"presize={a.presize},crop={a.crop}" if a.presize
             else f"size={a.size or 'native'}")
    print(f"measured BRSET n={len(B)}, mBRSET n={len(M)}  "
          f"({scale}, match_disc={a.match_disc or 'off'})\n")

    keys = [k for k in B.columns if k != "disc_px"]
    print(f"{'statistic':12s} {'BRSET':>10s} {'mBRSET':>10s} {'ratio':>8s} {'delta':>10s}")
    res = {}
    for k in keys:
        b, m = float(B[k].mean()), float(M[k].mean())
        res[k] = {"brset_mean": b, "brset_std": float(B[k].std()),
                  "mbrset_mean": m, "mbrset_std": float(M[k].std())}
        print(f"{k:12s} {b:10.5f} {m:10.5f} {m/(b+1e-9):8.3f} {m-b:+10.5f}")
    print(f"\nfundus disc diameter px: BRSET {B.disc_px.mean():.0f}, "
          f"mBRSET {M.disc_px.mean():.0f}")
    res["_meta"] = {"n_brset": len(B), "n_mbrset": len(M), "size": a.size,
                    "match_disc": a.match_disc, "presize": a.presize, "crop": a.crop,
                    "brset_disc_px": float(B.disc_px.mean()),
                    "mbrset_disc_px": float(M.disc_px.mean())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
