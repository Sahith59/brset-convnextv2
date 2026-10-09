"""Fit the 15-parameter acquisition model, and audit it honestly.

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
"""
import argparse, importlib.util, json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path

spec = importlib.util.spec_from_file_location("P", str(Path(__file__).parent / "72_physical_layer.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

FIT_KEYS = ["sharpness", "brightness", "contrast", "falloff", "saturation",
            "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"]


def load_batch(split_dir, n, size, seed):
    d = Path(split_dir)
    files = pd.read_csv(d / "labels.csv").file.values
    files = np.random.default_rng(seed).choice(files, n, replace=False)
    ims = []
    for f in files:
        try:
            im = Image.open(d / f).convert("RGB").resize((size, size), Image.BILINEAR)
        except Exception:
            continue
        ims.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1))
    return torch.stack(ims)


def distance(stats, target, weights=None):
    """Normalised squared distance, each statistic scaled by the target's own
    per-image spread so no single statistic dominates by unit."""
    tot = 0.0
    for k in FIT_KEYS:
        t = target[k]["mbrset_mean"]; sd = max(target[k]["mbrset_std"], 1e-6)
        w = 1.0 if weights is None else weights.get(k, 1.0)
        tot = tot + w * ((stats[k].mean() - t) / sd) ** 2
    return tot


def report(name, stats, target):
    print(f"\n--- {name}")
    print(f"{'statistic':11s} {'got':>10s} {'target':>10s} {'z':>8s}")
    z2 = 0.0
    for k in FIT_KEYS:
        g = float(stats[k].mean()); t = target[k]["mbrset_mean"]
        sd = max(target[k]["mbrset_std"], 1e-6); z = (g - t) / sd
        z2 += z * z
        print(f"{k:11s} {g:10.5f} {t:10.5f} {z:+8.2f}")
    print(f"{'TOTAL z^2':11s} {z2:10.3f}")
    return z2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="results/chromatic_stats_native.json")
    ap.add_argument("--n_fit", type=int, default=64)
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default="results/physics_v2_fit.json")
    a = ap.parse_args()

    target = json.load(open(a.targets))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}")
    x_fit = load_batch("data/finetune_multilabel/train", a.n_fit, a.size, 0).to(dev)
    x_ev = load_batch("data/finetune_multilabel/train", a.n_eval, a.size, 999).to(dev)
    print(f"fit on {len(x_fit)} images, evaluate on {len(x_ev)} DISJOINT images")

    z_raw = report("untouched BRSET (held out)", P.image_stats_torch(x_ev), target)

    L = P.PhysicalAcquisition().to(dev)
    opt = torch.optim.Adam(L.parameters(), lr=a.lr)
    for step in range(a.steps):
        opt.zero_grad()
        loss = distance(P.image_stats_torch(L(x_fit)), target)
        loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"  step {step:4d}  fit loss {float(loss):.3f}", flush=True)

    with torch.no_grad():
        z_fit = report("fitted 15-parameter model (held out)", P.image_stats_torch(L(x_ev)), target)

    p = {k: round(float(v), 5) for k, v in L.params().items()}
    print(f"\nfitted parameters:")
    pinned = []
    for k in L.KEYS:
        lo, hi = L.RANGES[k]
        frac = (p[k] - lo) / (hi - lo)
        flag = ""
        if frac < 0.02 or frac > 0.98:
            flag = "  <-- PINNED AT BOUND"; pinned.append(k)
        print(f"  {k:16s} {p[k]:+9.5f}   [{lo:+.3f}, {hi:+.3f}]{flag}")

    # AUDIT 1: how much of the matched sharpness is synthetic noise?
    with torch.no_grad():
        s_full = float(P.image_stats_torch(L(x_ev))["sharpness"].mean())
        ra, rb = L.raw.data[L.KEYS.index("noise_a")].item(), L.raw.data[L.KEYS.index("noise_b")].item()
        L.raw.data[L.KEYS.index("noise_a")] = -20.0
        L.raw.data[L.KEYS.index("noise_b")] = -20.0
        s_nonoise = float(P.image_stats_torch(L(x_ev))["sharpness"].mean())
        L.raw.data[L.KEYS.index("noise_a")] = ra
        L.raw.data[L.KEYS.index("noise_b")] = rb
    noise_frac = (s_full - s_nonoise) / max(s_full, 1e-12)
    tgt_sharp = target["sharpness"]["mbrset_mean"]

    print(f"\n=== AUDIT ===")
    print(f"held-out z^2: untouched {z_raw:.3f} -> fitted {z_fit:.3f} "
          f"({100*(z_raw-z_fit)/max(z_raw,1e-9):.1f}% reduction)")
    print(f"sharpness from synthetic noise: {100*noise_frac:.1f}%  (was 81% in the 9-param fit)")
    print(f"  with noise {s_full:.6f}, without {s_nonoise:.6f}, target {tgt_sharp:.6f}")
    print(f"sharpen_k = {p['sharpen_k']:+.4f}  "
          f"({'SHARPENS, consistent with mBRSET having more HF energy' if p['sharpen_k']>0 else 'BLURS - same pathology as the old fit'})")
    print(f"parameters pinned at a bound: {pinned if pinned else 'none'}")
    if pinned:
        print("  -> the family is still constrained on these axes; report honestly")

    json.dump({"params": p, "z2_untouched": z_raw, "z2_fitted": z_fit,
               "noise_fraction_of_sharpness": noise_frac, "pinned": pinned,
               "targets_file": a.targets}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
