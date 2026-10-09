"""Reproducible audit of the sharpness measurement and the fit's identifiability.

Exists because two findings were previously produced by throwaway scripts that
no longer exist, leaving only log text. For a paper whose contribution is
measurement discipline that is not acceptable. Everything here regenerates.

AUDIT 1: rim contamination of the sharpness statistic.
  image_stats computes a Laplacian on the FULL frame and then masks, so pixels
  at the field-of-view rim (retina meeting black surround, a hard step) are
  included. The two datasets have different FOV geometry: BRSET's disc is
  clipped by the frame top and bottom, so part of its boundary is the image
  border (reflect-padded, no step), while mBRSET's disc is fully inscribed and
  therefore has a complete 360-degree rim. If the rim dominates, the measured
  "sharpness" ratio is reporting FOV geometry, not image quality.
  Test: erode the mask and watch the ratio.

AUDIT 2: is the 15-parameter fit identified?
  Fit the same objective from several random inits. If parameters land in
  different places at comparable loss, the family is not identified and no
  claim may be made about any individual parameter, including the claim that
  the model "pins at its bounds" and is therefore exhausted.
"""
import argparse, importlib.util, json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from scipy import ndimage

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("P", str(HERE / "72_physical_layer.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


def load(split, n, size, seed):
    d = Path(split)
    files = pd.read_csv(d / "labels.csv").file.values
    files = np.random.default_rng(seed).choice(files, n, replace=False)
    out = []
    for f in files:
        try:
            im = Image.open(d / f).convert("RGB")
        except Exception:
            continue
        if size:
            im = im.resize((size, size), Image.BILINEAR)
        out.append(np.asarray(im, dtype=np.float32) / 255.0)
    return out


def sharp_eroded(a, erode):
    """Laplacian variance inside the FOV, with the mask eroded by `erode` px."""
    g = a.mean(axis=2)
    m = g > 0.04
    if erode:
        m = ndimage.binary_erosion(m, iterations=erode)
    if m.sum() < 1000:
        return np.nan, np.nan
    lap = ndimage.laplace(g)
    return float(lap[m].var()), float(m.mean())


def audit_rim(n, sizes, erosions):
    print("=" * 74)
    print("AUDIT 1: is the sharpness ratio a camera property or FOV geometry?")
    print("=" * 74)
    rows = []
    for size in sizes:
        B = load("data/finetune_multilabel/train", n, size, 0)
        M = load("data/finetune_mbrset_multilabel/train", n, size, 1)
        tag = f"{'native' if not size else f'sq{size}'}"
        print(f"\n--- {tag}  (n={len(B)}/{len(M)})")
        print(f"{'erode px':>9} {'BRSET':>11} {'mBRSET':>11} {'ratio':>8} "
              f"{'B maskfrac':>11} {'M maskfrac':>11}")
        for e in erosions:
            bs = [sharp_eroded(a, e) for a in B]
            ms = [sharp_eroded(a, e) for a in M]
            b = np.nanmean([x[0] for x in bs]); bm = np.nanmean([x[1] for x in bs])
            mm_ = np.nanmean([x[0] for x in ms]); mf = np.nanmean([x[1] for x in ms])
            r = mm_ / max(b, 1e-12)
            print(f"{e:>9} {b:11.6f} {mm_:11.6f} {r:8.2f} {bm:11.3f} {mf:11.3f}")
            rows.append({"size": tag, "erode": e, "brset": b, "mbrset": mm_,
                         "ratio": r, "brset_maskfrac": bm, "mbrset_maskfrac": mf})
    print("\nREAD: if the ratio collapses once the rim is excluded and then stays")
    print("flat and scale-invariant, the un-eroded number was FOV geometry.")
    return rows


def audit_identifiability(n, size, steps, trials, targets):
    print("\n" + "=" * 74)
    print("AUDIT 2: is the 15-parameter fit identified?")
    print("=" * 74)
    tgt = json.load(open(targets))
    KEYS = ["sharpness", "brightness", "contrast", "falloff", "saturation",
            "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"]
    imgs = load("data/finetune_multilabel/train", n, size, 0)
    x = torch.stack([torch.from_numpy(a).permute(2, 0, 1) for a in imgs])

    def dist(L):
        s = P.image_stats_torch(L(x))
        return sum(((s[k].mean() - tgt[k]["mbrset_mean"])
                    / max(tgt[k]["mbrset_std"], 1e-6)) ** 2 for k in KEYS)

    res = []
    for t in range(trials):
        torch.manual_seed(1000 + t)
        L = P.PhysicalAcquisition()
        with torch.no_grad():
            L.raw.data += torch.randn_like(L.raw) * 1.5   # perturb the init
        opt = torch.optim.Adam(L.parameters(), lr=0.05)
        for _ in range(steps):
            opt.zero_grad(); l = dist(L); l.backward(); opt.step()
        p = {k: float(v) for k, v in L.params().items()}
        p["_z2"] = float(dist(L))
        res.append(p)
        print(f"  trial {t}: z2={p['_z2']:.3f}  sharpen_k={p['sharpen_k']:+.3f}  "
              f"sat={p['saturation_gain']:.3f}  hole={p['hole_amp']:.3f}")

    print(f"\n{'parameter':16s} {'min':>9} {'max':>9} {'spread/range':>13}  flag")
    unid = []
    for k in P.PhysicalAcquisition.KEYS:
        v = [r[k] for r in res]
        lo, hi = P.PhysicalAcquisition.RANGES[k]
        frac = (max(v) - min(v)) / (hi - lo)
        flag = ""
        if frac > 0.25:
            flag = "NOT IDENTIFIED"; unid.append(k)
        print(f"{k:16s} {min(v):+9.4f} {max(v):+9.4f} {frac:12.1%}  {flag}")
    signflip = [k for k in ["sharpen_k"] if min(r[k] for r in res) < 0 < max(r[k] for r in res)]
    if signflip:
        print(f"\nSIGN FLIPS ACROSS INITS: {signflip}")
        print("  -> no claim may be made about the direction of these parameters")
    print(f"\nunidentified (>25% of range): {unid if unid else 'none'}")
    return res, unid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--fit_n", type=int, default=48)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--targets", default="results/chromatic_stats_native.json")
    ap.add_argument("--out", default="results/physics_audit.json")
    a = ap.parse_args()

    rim = audit_rim(a.n, [0, 512], [0, 1, 2, 5, 10, 20])
    ident, unid = audit_identifiability(a.fit_n, 512, a.steps, a.trials, a.targets)
    json.dump({"rim": rim, "identifiability": ident, "unidentified": unid},
              open(a.out, "w"), indent=1, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
