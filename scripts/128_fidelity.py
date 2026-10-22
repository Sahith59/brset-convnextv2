"""How close is each synthetic image to the real photo it was made from (research/codex D088).

Reviewers will ask whether 'synthetic' images are near-copies of real training photos,
especially the mBRSET re-renderings, which inherit real mBRSET labels. For a fixed random
subset of source images (same subset for every variant) this reports, inside the retina mask:

  mae     mean absolute RGB difference, 0-255 scale
  psnr    peak signal-to-noise ratio over the masked pixels, dB
  ssim    structural similarity of the green channel (Gaussian window, sigma 1.5), masked mean
  s_corr  Pearson correlation of the structure maps (scripts/109 structure_map: vessels,
          disc, lesions as local contrast) -- the disease-detail carrier

Each variant is compared with its SOURCE photo (BRSET or mBRSET train cache). A second
rendering of the same source with another seed, when one exists, is also compared with
the first: that is the spread of the generator itself.
Only file names are read from the meta CSVs; no label column is parsed.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).parent
G = Path("/data/users3/sthummala2/brset_gen")
sp = importlib.util.spec_from_file_location("GC", str(HERE / "109_gen_common.py"))
GC = importlib.util.module_from_spec(sp)
sp.loader.exec_module(GC)

VARIANTS = {
    # name: (source, folder)
    "mbr_s0.40": ("mbrset", "full_mbr_gen_both_e40k_source_s0.4_w1.5"),
    "mbr_s0.40_seed1": ("mbrset", "full_mbr_gen_both_e40k_source_s0.4_w1.5_seed1"),
    "mbr_s0.25": ("mbrset", "full_mbr_gen_both_e40k_source_s0.25_w1.5"),
    "mbr_noise_s1.0": ("mbrset", "full_mbr_gen_both_e40k_none_s1.0_w1.5"),
    "brs_phys_only": ("brset", "full_physonly_v2noartifact"),
    "brs_s0.40": ("brset", "full_gen_both_e40k_phys_s0.4_w1.5"),
    "brs_s0.40_seed1": ("brset", "full_gen_both_e40k_phys_s0.4_w1.5_seed1"),
    "brs_s0.30": ("brset", "full_gen_both_e40k_phys_s0.3_w1.5"),
    "brs_100k_s0.40": ("brset", "full_gen_both_ld100_e100k_phys_s0.4_w1.5"),
}
PAIRS = [("mbr_s0.40", "mbr_s0.40_seed1"), ("brs_s0.40", "brs_s0.40_seed1")]


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def ssim_green(a, b, m):
    a, b = a[..., 1] / 255.0, b[..., 1] / 255.0
    f = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = f(a), f(b)
    va, vb, cov = f(a * a) - ma * ma, f(b * b) - mb * mb, f(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(s[m].mean())


def smap(x):
    t = torch.from_numpy(x / 255.0).permute(2, 0, 1)[None].float()
    m = GC.fundus_mask(t)
    return GC.structure_map(t, m)[0, 0].numpy(), GC.erode(m, 4)[0, 0].numpy() > 0.5


def compare(a, b):
    m = a.mean(-1) > 0.04 * 255
    d = np.abs(a - b)[m]
    mse = float((d ** 2).mean())
    sa, me = smap(a)
    sb, _ = smap(b)
    return dict(mae=float(d.mean()), psnr=float(10 * np.log10(255 ** 2 / max(mse, 1e-8))),
                ssim=ssim_green(a, b, m), s_corr=float(np.corrcoef(sa[me], sb[me])[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=str(HERE.parent / "results/fidelity_d088.json"))
    a = ap.parse_args()
    caches, subsets = {}, {}
    rng = np.random.RandomState(0)
    for src in ("brset", "mbrset"):
        meta = pd.read_csv(G / f"cache/{src}_train_meta.csv", usecols=["idx", "file"])
        caches[src] = (np.load(G / f"cache/{src}_train_560.npy", mmap_mode="r"), meta)
        subsets[src] = np.sort(rng.choice(len(meta), size=a.n, replace=False))
    R = {"n": a.n, "variants": {}, "seed_pairs": {}}
    for name, (src, folder) in VARIANTS.items():
        arr, meta = caches[src]
        rows = []
        for i in subsets[src]:
            p = G / "synth" / folder / (Path(meta.file[i]).stem + ".png")
            if p.exists():
                rows.append(compare(arr[int(meta.idx[i])].astype(np.float32), load(p)))
        if not rows:
            print(f"{name}: no images yet", flush=True)
            continue
        R["variants"][name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | {"n": len(rows)}
        print(name, {k: round(v, 4) for k, v in R["variants"][name].items()}, flush=True)
    for x, y in PAIRS:
        src, fx = VARIANTS[x]
        _, fy = VARIANTS[y]
        meta = caches[src][1]
        rows = []
        for i in subsets[src]:
            st = Path(meta.file[i]).stem + ".png"
            px, py = G / "synth" / fx / st, G / "synth" / fy / st
            if px.exists() and py.exists():
                rows.append(compare(load(px), load(py)))
        if rows:
            R["seed_pairs"][f"{x} vs {y}"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | {"n": len(rows)}
            print(f"{x} vs {y}", {k: round(v, 4) for k, v in R["seed_pairs"][f'{x} vs {y}'].items()}, flush=True)
    Path(a.out).write_text(json.dumps(R, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    main()
