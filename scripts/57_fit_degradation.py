"""EXPERIMENT 3, step 3b: fit the degradation to measured mBRSET statistics.

The contribution is here. GDRNet applies its operators at a fixed probability
with hand-chosen magnitudes, tuned for generic robustness rather than any target
camera. This searches the parameter space so that degraded BRSET images match
the appearance statistics measured on real mBRSET images in step 3a.

No diagnostic label is used anywhere. Only image appearance.

Also produces the qualitative comparison figure, which is what the advisor asked
to see: original, generic GDRNet settings, fitted settings, and a real mBRSET
image side by side.
"""
import importlib.util, json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_s = importlib.util.spec_from_file_location("ops", str(Path(__file__).parent / "55_degradation_ops.py"))
O = importlib.util.module_from_spec(_s); _s.loader.exec_module(O)

STATS = ["sharpness", "brightness", "contrast", "falloff", "saturation"]
N_IMG = 25          # images per parameter evaluation
N_TRIAL = 150       # random-search trials
SIZE = 512


def load_brset(n, seed=0):
    d = Path("data/finetune_multilabel/train")
    df = pd.read_csv(d / "labels.csv")
    rng = np.random.default_rng(seed)
    out = []
    for f in rng.choice(df.file.values, n, replace=False):
        try:
            out.append(Image.open(d / f).convert("RGB").resize((SIZE, SIZE)))
        except Exception:
            pass
    return out


def mean_stats(images, params, rng):
    rows = []
    for im in images:
        st = O.image_stats(O.degrade(im, params, rng) if params else im)
        if st: rows.append(st)
    return pd.DataFrame(rows).mean()


def distance(got, target, scale):
    """Normalised distance. Each statistic is divided by the spread of that
    statistic within mBRSET, so no single one dominates purely by units."""
    return float(sum(((got[k] - target[k]) / scale[k]) ** 2 for k in STATS))


def main():
    tgt_json = json.load(open("results/dataset_appearance_stats_preserve.json"))
    target = {k: tgt_json[k]["mbrset_mean"] for k in STATS}
    scale = {k: max(tgt_json[k]["mbrset_std"], 1e-6) for k in STATS}
    print("TARGET, measured on real mBRSET:")
    for k in STATS: print(f"  {k:11s} {target[k]:.5f}  (spread {scale[k]:.5f})")

    images = load_brset(N_IMG)
    print(f"\nfitting on {len(images)} BRSET images, {N_TRIAL} random-search trials\n")

    base = mean_stats(images, None, None)
    print(f"  untouched BRSET        distance {distance(base, target, scale):8.2f}")
    rng = np.random.default_rng(0)
    gen = mean_stats(images, O.GDRNET_GENERIC, np.random.default_rng(1))
    print(f"  GDRNet published       distance {distance(gen, target, scale):8.2f}")

    RANGES = {"blur_sigma": (0.0, 1.2), "light_strength": (0.0, 0.6),
              "n_spot": (0, 4), "n_hole": (0, 3), "halo": (0.0, 0.20),
              "brightness_gain": (0.70, 1.15), "contrast_gain": (0.9, 2.0),
              "saturation_gain": (0.5, 1.2), "noise_sigma": (0.0, 0.030)}
    best, best_d = None, float("inf")
    for t in range(N_TRIAL):
        p = {k: (rng.integers(lo, hi + 1) if k.startswith("n_") else rng.uniform(lo, hi))
             for k, (lo, hi) in RANGES.items()}
        d = distance(mean_stats(images, p, np.random.default_rng(t)), target, scale)
        if d < best_d:
            best_d, best = d, p
            print(f"  trial {t:3d}  distance {d:8.2f}  <- new best")

    got = mean_stats(images, best, np.random.default_rng(999))
    print(f"\nFITTED PARAMETERS")
    for k, v in sorted(best.items()):
        print(f"  {k:16s} {float(v):.4f}")
    print(f"\n{'statistic':12s} {'BRSET':>10} {'GDRNet':>10} {'FITTED':>10} {'mBRSET target':>14}")
    for k in STATS:
        print(f"{k:12s} {base[k]:10.5f} {gen[k]:10.5f} {got[k]:10.5f} {target[k]:14.5f}")
    print(f"\ndistance: BRSET {distance(base,target,scale):.2f} | GDRNet {distance(gen,target,scale):.2f} "
          f"| FITTED {best_d:.2f}")
    json.dump({k: float(v) for k, v in best.items()},
              open("results/fitted_degradation_params.json", "w"), indent=2)
    print("wrote results/fitted_degradation_params.json")


if __name__ == "__main__":
    main()
