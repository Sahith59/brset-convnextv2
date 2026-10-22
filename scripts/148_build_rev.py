"""D105 experiment A, arm REV1: the user's literal idea -- diffusion renderer FIRST, physics camera model AFTER.

Input : synth/full_gen_both_e40k_source_s0.4_w1.5 (gen_both 40k renderer started from the plain BRSET photo, no physics;
        the SRC1 arm). Output: synth/full_gen_both_e40k_source_s0.4_w1.5_postphys, same stems, 560-px PNG.
Physics: the fitted stage-1 model exactly as scripts/111_translate.py loads it (88_synth_augment.load_translator(
        results/xlate_v2_noartifact/translator_last.pth).phys), output clamped to [0, 1] as in 111; noise seed = image index.
--mode build --shard k --n_shards n : writes images (CPU).
--mode check : D105 kill test on the BUILT images -- lesion-detail correlation with the ORIGINAL BRSET photo on the
        analysis/detail_retention.py protocol (200 grade 1-2 photos, DoG 0.7-2.5 px, green) must lie in 0.93-0.97
        (agents measured 0.952), and the 13-statistic camera z^2 on the physics_stats.py sample is reported (agents: 5.94).
"""
import argparse, importlib.util, json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).parent.parent
G = Path("/data/users3/sthummala2/brset_gen")
SRC = G / "synth/full_gen_both_e40k_source_s0.4_w1.5"
OUT = G / "synth/full_gen_both_e40k_source_s0.4_w1.5_postphys"
KEYS = ["brightness", "contrast", "falloff", "saturation", "r_frac", "g_frac", "b_frac", "r_std", "g_std", "b_std",
        "r_lapvar", "g_lapvar", "b_lapvar"]


def _load(name, mod):
    sp = importlib.util.spec_from_file_location(mod, str(ROOT / "scripts" / name))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["build", "check"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    a = ap.parse_args()
    meta = pd.read_csv(G / "cache/brset_train_meta.csv")
    stems = [f.split(".")[0] for f in meta.file]
    if a.mode == "build":
        torch.set_num_threads(4)
        phys = _load("88_synth_augment.py", "S88").load_translator(
            str(ROOT / "results/xlate_v2_noartifact/translator_last.pth"), device="cpu").phys.eval()
        OUT.mkdir(parents=True, exist_ok=True)
        ids = list(range(len(stems)))[a.shard::a.n_shards]
        n = 0
        for i in ids:
            o = OUT / f"{stems[i]}.png"
            if o.exists():
                continue
            x = torch.from_numpy(np.asarray(Image.open(SRC / f"{stems[i]}.png").convert("RGB"), dtype=np.float32) / 255.0)
            with torch.no_grad():
                y = phys(x.permute(2, 0, 1)[None], generator=torch.Generator().manual_seed(i)).clamp(0, 1)
            Image.fromarray((y[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(o)
            n += 1
        print(f"shard {a.shard}/{a.n_shards}: wrote {n} of {len(ids)}", flush=True)
        return
    # ---- check
    A = np.load(G / "cache/brset_train_560.npy", mmap_mode="r")
    pick = meta[meta.grade.isin([1, 2])].sample(200, random_state=0)
    band = lambda g: cv2.GaussianBlur(g, (0, 0), 0.7) - cv2.GaussianBlur(g, (0, 0), 2.5)
    ret = {"REV (renderer -> physics)": [], "SRC (renderer only)": []}
    for i, f in zip(pick.index, pick.file):
        st = f.split(".")[0]
        o = np.array(A[i]).astype(np.float32); m = o.mean(2) > 10
        m = cv2.erode(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        bo = band(o[..., 1])
        for k, d in (("REV (renderer -> physics)", OUT), ("SRC (renderer only)", SRC)):
            x = np.asarray(Image.open(d / f"{st}.png").convert("RGB"), dtype=np.float32)[..., 1]
            ret[k].append(np.corrcoef(bo[m], band(x)[m])[0, 1])
    M = _load("76_measure_chromatic.py", "M76")
    T = json.loads((ROOT / "results/chromatic_stats_sq512.json").read_text())
    idx = np.random.RandomState(0).choice(len(meta), 400, replace=False)
    rows = [M.stats(Image.open(OUT / f"{stems[i]}.png").convert("RGB").crop((24, 24, 536, 536))) for i in idx]
    df = pd.DataFrame([r for r in rows if r])
    z = {k: float((df[k].mean() - T[k]["mbrset_mean"]) / T[k]["mbrset_std"]) for k in KEYS}
    res = {k: float(np.mean(v)) for k, v in ret.items()}
    res["REV_camera_sum_z2"] = float(sum(v * v for v in z.values()))
    res["n_images"] = sum(1 for _ in OUT.glob("*.png"))
    ok = 0.93 <= res["REV (renderer -> physics)"] <= 0.97 and res["n_images"] == len(stems)
    res["pass"] = ok
    print(json.dumps(res, indent=1), flush=True)
    (ROOT / "results/rev_build_check.json").write_text(json.dumps(res, indent=1))
    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()
