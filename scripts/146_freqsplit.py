"""Frequency-split twins (research/codex D103): the final synthetic-only try under Claim A.

Measured motivation (analysis/physics_stats.py, 19 Oct): the physics stage brings BRSET's fine texture to the phone level
(red Laplacian-variance z -2.03 -> -0.02), but the diffusion renderer smooths it away again (-1.51 for Translator-v2).
Mild DR (microaneurysms, a few pixels at 560 px) lives in that band, and mild DR is where C1 loses (grade-1 caught 0.42 vs 0.64).

twin_fs = lowpass_sigma(Translator-v2 twin) + highpass_sigma(physics-only image of the same BRSET photo)
  low band  = the renderer's phone look (colour, illumination, camera style)
  high band = the physics image's fine detail (the BRSET eye's own lesions and vessel edges + phone-matched grain)
Both images come from the same BRSET training photo and are pixel-aligned; no mBRSET photo is used.
Blurs are normalised inside the field of view (no halo from the black border).

--mode pilot : 400 BRSET train photos (same sample as physics_stats.py); for each sigma: sum of z^2 over the 13 camera
               statistics vs mBRSET, and the alignment check corr(highpass twin, highpass physics). Chosen sigma = the
               smallest z^2 (label-free; fixed in D103 before any output). Writes results/freqsplit_pilot.json.
--mode build : all 11,372 twins at --sigma into brset_gen/synth/full_c1fs/ (sigma recorded in its SIGMA.txt).
"""
import argparse, importlib.util, json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).parent.parent
G = Path("/data/users3/sthummala2/brset_gen")
TWIN = G / "synth/full_gen_both_dc_phys_s0.4_w1.5"        # Translator-v2 = C1
PHYS = G / "synth/full_physonly_v2noartifact"             # physics only
KEYS = ["brightness", "contrast", "falloff", "saturation", "r_frac", "g_frac", "b_frac", "r_std", "g_std", "b_std",
        "r_lapvar", "g_lapvar", "b_lapvar"]


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def nblur(x, m, s):                       # Gaussian blur normalised inside the mask
    num = cv2.GaussianBlur(x * m[..., None], (0, 0), s)
    den = cv2.GaussianBlur(m, (0, 0), s)[..., None]
    return num / np.maximum(den, 1e-3)


def split(stem, s):
    t, p = load(TWIN / f"{stem}.png"), load(PHYS / f"{stem}.png")
    m = ((t.mean(2) > 10) & (p.mean(2) > 10)).astype(np.float32)
    hp_p = p - nblur(p, m, s)
    out = nblur(t, m, s) + hp_p
    out = np.where(m[..., None] > 0, out, t)
    return np.clip(out, 0, 255).round().astype(np.uint8), t, hp_p, m


def build_one(args):
    stem, s, od = args
    out, *_ = split(stem, s)
    Image.fromarray(out).save(od / f"{stem}.png")
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pilot", "build"])
    ap.add_argument("--sigmas", default="1.0,1.5,2.0,3.0")
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    meta = pd.read_csv(G / "cache/brset_train_meta.csv")
    stems = [f.split(".")[0] for f in meta.file]
    if a.mode == "build":
        assert a.sigma > 0
        od = G / "synth/full_c1fs"
        od.mkdir(parents=True, exist_ok=True)
        (od / "SIGMA.txt").write_text(f"{a.sigma:g}\n")
        with Pool(a.workers) as pool:
            n = sum(1 for _ in pool.imap_unordered(build_one, [(s, a.sigma, od) for s in stems], chunksize=32))
        print(f"built {n} frequency-split twins -> {od}", flush=True)
        return
    sp = importlib.util.spec_from_file_location("M76", str(ROOT / "scripts/76_measure_chromatic.py"))
    M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
    T = json.loads((ROOT / "results/chromatic_stats_sq512.json").read_text())
    idx = np.random.RandomState(0).choice(len(meta), 400, replace=False)      # same sample as analysis/physics_stats.py
    crop = lambda im: im.crop((24, 24, 536, 536))
    sig = [float(x) for x in a.sigmas.split(",")]
    rows = {k: [] for k in ["Translator-v2 (C1)", "physics only"] + [f"split sigma {s:g}" for s in sig]}
    align = {s: [] for s in sig}
    for i in idx:
        st = stems[i]
        for name, im in (("Translator-v2 (C1)", TWIN / f"{st}.png"), ("physics only", PHYS / f"{st}.png")):
            r = M.stats(crop(Image.open(im).convert("RGB")))
            if r: rows[name].append(r)
        for s in sig:
            out, t, hp_p, m = split(st, s)
            r = M.stats(crop(Image.fromarray(out)))
            if r: rows[f"split sigma {s:g}"].append(r)
            hp_t = t - nblur(t, m, s)
            msk = m > 0
            x, y = hp_t.mean(2)[msk], hp_p.mean(2)[msk]
            ys = np.roll(hp_p.mean(2), 16, axis=1)[msk]                 # control: physics detail shifted 16 px
            align[s].append((float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(x, ys)[0, 1])))
    res = {}
    for name, rr in rows.items():
        df = pd.DataFrame(rr)
        z = {k: float((df[k].mean() - T[k]["mbrset_mean"]) / T[k]["mbrset_std"]) for k in KEYS}
        res[name] = {"n": len(df), "z": z, "sum_z2": float(sum(v * v for v in z.values()))}
        print(f"{name:20s} sum z^2 {res[name]['sum_z2']:6.2f} | lapvar z r {z['r_lapvar']:+.2f} g {z['g_lapvar']:+.2f} "
              f"b {z['b_lapvar']:+.2f} | contrast {z['contrast']:+.2f}", flush=True)
    al = {f"{s:g}": {"aligned_mean": float(np.mean([c for c, _ in v])), "aligned_min": float(np.min([c for c, _ in v])),
                     "shifted_mean": float(np.mean([c for _, c in v]))} for s, v in align.items()}
    print("alignment corr(highpass twin, highpass physics) vs 16-px-shifted control:", json.dumps(al), flush=True)
    best = min(sig, key=lambda s: res[f"split sigma {s:g}"]["sum_z2"])
    A, z2b, z2t = al[f"{best:g}"], res[f"split sigma {best:g}"]["sum_z2"], res["Translator-v2 (C1)"]["sum_z2"]
    # D103 pre-classifier gate: images line up, and the split moves the camera statistics toward the phone
    ok = A["aligned_mean"] >= 0.1 and A["aligned_mean"] >= 2 * abs(A["shifted_mean"]) and z2b < z2t
    out = {"stats": res, "alignment": al, "chosen_sigma": best, "rule": "smallest sum z^2 (D103)", "gate_pass": ok}
    (ROOT / "results/freqsplit_pilot.json").write_text(json.dumps(out, indent=1))
    print(f"CHOSEN sigma {best:g}  gate {'PASS' if ok else 'FAIL'} (aligned {A['aligned_mean']:.3f} vs shifted "
          f"{A['shifted_mean']:.3f}; z^2 {z2b:.2f} vs Translator-v2 {z2t:.2f})", flush=True)
    if not ok:
        raise SystemExit("D103 pre-classifier gate failed: STOP")


if __name__ == "__main__":
    main()
