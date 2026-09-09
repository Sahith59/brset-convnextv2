"""EXPERIMENT 3, step 3b deliverable: the qualitative comparison figure.

The advisor asked to see the degraded images before trusting the procedure:
"We need to validate this degradation procedure, so maybe it's just random. You
need to do some qualitative evaluation."

Each row is one BRSET image shown four ways:
  1. original BRSET
  2. GDRNet at its published settings
  3. fitted to measured mBRSET statistics
  4. a real mBRSET image, for reference

The point of the figure is column 2 against column 3. The published settings
move the image AWAY from the target on four of five measured statistics.
"""
import importlib.util, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

_s = importlib.util.spec_from_file_location("ops", str(Path(__file__).parent / "55_degradation_ops.py"))
O = importlib.util.module_from_spec(_s); _s.loader.exec_module(O)

N_ROWS = 5
SIZE = 512


def main():
    fitted = json.load(open("results/fitted_degradation_params.json"))
    stats = json.load(open("results/dataset_appearance_stats_preserve.json"))
    rng = np.random.default_rng(7)

    bd = Path("data/finetune_multilabel/train")
    md = Path("data/finetune_mbrset_multilabel/train")
    bf = rng.choice(pd.read_csv(bd / "labels.csv").file.values, N_ROWS, replace=False)
    mf = rng.choice(pd.read_csv(md / "labels.csv").file.values, N_ROWS, replace=False)

    fig, ax = plt.subplots(N_ROWS, 4, figsize=(14.5, 3.6 * N_ROWS))
    titles = ["BRSET original\n(tabletop camera)",
              "GDRNet published settings\n(generic, not fitted)",
              "Fitted to measured\nmBRSET statistics",
              "Real mBRSET\n(handheld camera)"]
    for r in range(N_ROWS):
        src = Image.open(bd / bf[r]).convert("RGB").resize((SIZE, SIZE))
        gen = O.degrade(src, O.GDRNET_GENERIC, np.random.default_rng(100 + r))
        fit = O.degrade(src, fitted, np.random.default_rng(200 + r))
        real = Image.open(md / mf[r]).convert("RGB").resize((SIZE, SIZE))
        for c, im in enumerate([src, gen, fit, real]):
            ax[r, c].imshow(im); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(titles[c], fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    out = "results/degradation_examples.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")

    # the quantitative panel that goes with it
    ST = ["sharpness", "brightness", "contrast", "falloff", "saturation"]
    imgs = [Image.open(bd / f).convert("RGB").resize((SIZE, SIZE))
            for f in pd.read_csv(bd / "labels.csv").file.values[:40]]
    def agg(p, seed):
        r = np.random.default_rng(seed)
        rows = [O.image_stats(O.degrade(i, p, r) if p else i) for i in imgs]
        return pd.DataFrame([x for x in rows if x]).mean()
    base, gen_s, fit_s = agg(None, 0), agg(O.GDRNET_GENERIC, 1), agg(fitted, 2)
    tgt = {k: stats[k]["mbrset_mean"] for k in ST}

    fig2, axes = plt.subplots(1, 5, figsize=(18, 3.6))
    for i, k in enumerate(ST):
        a = axes[i]
        vals = [base[k], gen_s[k], fit_s[k], tgt[k]]
        cols = ["#8C8C8C", "#B3562A", "#1F4E79", "#1E6B3C"]
        a.bar(range(4), vals, color=cols)
        a.axhline(tgt[k], color="#1E6B3C", ls="--", lw=1.2)
        a.set_xticks(range(4))
        a.set_xticklabels(["BRSET", "GDRNet", "fitted", "target"], fontsize=9, rotation=20)
        a.set_title(k, fontsize=11, fontweight="bold")
        a.spines[["top", "right"]].set_visible(False)
    fig2.suptitle("Green dashed line is the measured mBRSET target. "
                  "GDRNet's published settings move away from it on four of five statistics.",
                  fontsize=11, y=1.06)
    plt.tight_layout()
    out2 = "results/degradation_statistics.png"
    plt.savefig(out2, dpi=110, bbox_inches="tight")
    print(f"wrote {out2}")

    print(f"\n{'statistic':12s} {'BRSET':>10} {'GDRNet':>10} {'fitted':>10} {'target':>10}   {'GDRNet dir':>11} {'fitted dir':>11}")
    for k in ST:
        dg = "AWAY" if abs(gen_s[k] - tgt[k]) > abs(base[k] - tgt[k]) else "closer"
        df = "AWAY" if abs(fit_s[k] - tgt[k]) > abs(base[k] - tgt[k]) else "closer"
        print(f"{k:12s} {base[k]:10.5f} {gen_s[k]:10.5f} {fit_s[k]:10.5f} {tgt[k]:10.5f}   {dg:>11} {df:>11}")


if __name__ == "__main__":
    main()
