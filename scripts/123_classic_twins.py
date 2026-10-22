"""Classical BRSET -> mBRSET 'twins' for the baseline arms (research/codex D076).

Same input as the diffusion twins (the 560 px BRSET-train cache), same output
convention (<stem>.png in brset_gen/synth/<tag>/), same retina mask applied at the
end, so the classifier pipeline treats every twin corpus identically.

  fda       Fourier Domain Adaptation (Yang & Soatto, CVPR 2020): the centred
            low-frequency amplitude window (half-width floor(min(h, w) * beta)) of
            the BRSET image is replaced by that of a randomly drawn mBRSET image;
            phase is kept.
  reinhard  Reinhard colour transfer: per-channel Lab mean and std inside the
            retina mask matched to a randomly drawn mBRSET image.
  --fov_fill (D113, Dr. Ye's review item 4): FDA restricted to the field of view.
            Before the FFT, each image's out-of-field pixels are set to its in-field
            per-channel mean, so the black border no longer feeds the low-frequency
            amplitude window; same beta, same mBRSET partner per image, output masked
            by the source retina mask as before. Output tag full_fdafov_b<beta>.

The mBRSET references are UNLABELLED training images (cache holds no labels).
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import cv2

G = Path("/data/users3/sthummala2/brset_gen")


def mask_of(x):
    return x.astype(np.float32).mean(-1) > 0.04 * 255


def fill_outside(x):
    m = mask_of(x); y = x.astype(np.float32).copy()
    for c in range(3):
        y[..., c][~m] = y[..., c][m].mean()
    return y


def fda(src, tgt, beta, fov_fill=False):
    s, t = (fill_outside(src), fill_outside(tgt)) if fov_fill else (src.astype(np.float32), tgt.astype(np.float32))
    h, w = s.shape[:2]
    b = int(np.floor(min(h, w) * beta))
    cy, cx = h // 2, w // 2
    out = np.empty_like(s)
    for c in range(3):
        fs = np.fft.fftshift(np.fft.fft2(s[..., c]))
        ft = np.fft.fftshift(np.fft.fft2(t[..., c]))
        amp, pha = np.abs(fs), np.angle(fs)
        amp[cy - b:cy + b + 1, cx - b:cx + b + 1] = np.abs(ft)[cy - b:cy + b + 1, cx - b:cx + b + 1]
        out[..., c] = np.real(np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha))))
    return np.clip(out, 0, 255)


def reinhard(src, tgt):
    ms, mt = mask_of(src), mask_of(tgt)
    # float32 RGB in [0, 1] -> Lab with L in [0, 100] (OpenCV's float convention)
    ls = cv2.cvtColor((src / 255.0).astype(np.float32), cv2.COLOR_RGB2LAB)
    lt = cv2.cvtColor((tgt / 255.0).astype(np.float32), cv2.COLOR_RGB2LAB)
    out = ls.copy()
    for c in range(3):
        a, b = ls[..., c][ms], lt[..., c][mt]
        out[..., c][ms] = (a - a.mean()) / (a.std() + 1e-6) * b.std() + b.mean()
    return np.clip(cv2.cvtColor(out.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1) * 255


def work(job):
    i, j, stem, method, beta, out, fov_fill = job
    p = out / f"{stem}.png"
    if p.exists():
        return
    src = np.array(SRC[i]); tgt = np.array(TGT[j])
    y = fda(src, tgt, beta, fov_fill) if method == "fda" else reinhard(src, tgt)
    y = (y * mask_of(src)[..., None]).round().astype(np.uint8)
    tmp = out / f".{stem}.tmp.png"
    Image.fromarray(y).save(tmp)
    tmp.replace(p)


def init(src_path, tgt_path):
    global SRC, TGT
    SRC = np.load(src_path, mmap_mode="r")
    TGT = np.load(tgt_path, mmap_mode="r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["fda", "reinhard"])
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fov_fill", action="store_true", help="D113: FDA restricted to the field of view (see docstring)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N images only")
    ap.add_argument("--tag", default="", help="override the output tag (smoke tests)")
    a = ap.parse_args()
    tag = (f"full_fdafov_b{a.beta}" if a.fov_fill else f"full_fda_b{a.beta}") if a.method == "fda" else "full_reinhard"
    tag = a.tag or tag
    out = G / "synth" / tag
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(G / "cache/brset_train_meta.csv")
    n_t = np.load(G / "cache/mbrset_train_560.npy", mmap_mode="r").shape[0]
    rng = np.random.RandomState(a.seed)
    ref = rng.randint(0, n_t, size=len(meta))
    jobs = [(int(i), int(r), Path(str(f)).stem, a.method, a.beta, out, a.fov_fill)
            for i, f, r in zip(meta.idx.values, meta.file.values, ref)]
    if a.limit:
        jobs = jobs[:a.limit]
    with Pool(a.workers, initializer=init,
              initargs=(G / "cache/brset_train_560.npy", G / "cache/mbrset_train_560.npy")) as pool:
        for k, _ in enumerate(pool.imap_unordered(work, jobs, chunksize=16)):
            if k % 2000 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    n = len(list(out.glob("img*.png")))
    print(f"{a.method}: {n} twins in {out}", flush=True)


if __name__ == "__main__":
    main()
