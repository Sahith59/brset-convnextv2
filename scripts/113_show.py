"""Side-by-side figure for a human eye: BRSET original | synthetic mBRSET (ours) |
real mBRSET photographs. One row per BRSET eye, chosen to span the ICDR grades.

The real mBRSET photos are of DIFFERENT eyes. The two datasets share no patients,
so there is no "same eye on the other camera" to show; the right-hand columns are
what the target camera's pictures look like, to compare style against.
"""
import argparse, importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
ROOT = HERE.parent
GEN = Path("/data/users3/sthummala2/brset_gen")


def _load(name, mod):
    sp = importlib.util.spec_from_file_location(mod, str(HERE / name))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


C = _load("109_gen_common.py", "C109")


def font(size):
    for f in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="ckpt_last.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--w_struct", type=float, default=1.5)
    ap.add_argument("--cell", type=int, default=420)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(GEN / "runs" / a.run / a.ckpt, map_location="cpu", weights_only=False)
    unet = C.build_unet(base=ck["args"]["base"]).to(dev).eval(); unet.load_state_dict(ck["ema"])
    S = _load("88_synth_augment.py", "S88")
    phys = S.load_translator(str(ROOT / "results/xlate_v2_noartifact/translator_last.pth"), device=dev).phys

    meta = pd.read_csv(GEN / "cache/brset_heldout_meta.csv")       # eyes nobody trained on
    A_b = np.load(GEN / "cache/brset_heldout_560.npy", mmap_mode="r")
    A_m = np.load(GEN / "cache/mbrset_train_560.npy", mmap_mode="r")
    rng = np.random.RandomState(a.seed)
    rows = [(g, int(rng.choice(meta[meta.grade == g].idx.values))) for g in (0, 1, 2, 3, 4)]
    real = rng.choice(len(A_m), size=2 * len(rows), replace=False)
    t = lambda arr, i: torch.from_numpy(np.array(arr[i])).permute(2, 0, 1).float().div(255)[None].to(dev)
    im = lambda x: Image.fromarray((x[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
                                   ).resize((a.cell, a.cell), Image.LANCZOS)

    heads = ["BRSET original\n(tabletop camera)", "OURS: diffusion only\n(synthetic mBRSET)",
             "OURS: physics + diffusion\n(synthetic mBRSET)", "REAL mBRSET photo\n(a different eye)",
             "REAL mBRSET photo\n(a different eye)"]
    pad, top, left = 8, 70, 150
    W = left + len(heads) * (a.cell + pad); H = top + len(rows) * (a.cell + pad)
    fig = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(fig)
    f1, f2 = font(18), font(17)
    for c, h in enumerate(heads):
        d.multiline_text((left + c * (a.cell + pad) + 6, 10), h, fill=(0, 0, 0), font=f1, spacing=4)
    names = {0: "no DR", 1: "mild", 2: "moderate", 3: "severe", 4: "proliferative"}
    for r, (g, i) in enumerate(rows):
        x = t(A_b, i); m = C.fundus_mask(x); s = C.structure_map(x, m)
        gen = torch.Generator(device=dev).manual_seed(a.seed * 100 + r)
        y_d = C.render(unet, s, m, steps=a.steps, w_struct=a.w_struct, generator=gen)
        y_p = C.render(unet, s, m, x_init=phys(x, generator=gen).clamp(0, 1), strength=a.strength,
                       steps=a.steps, w_struct=a.w_struct, generator=gen)
        cells = [im(x), im(y_d), im(y_p), im(t(A_m, real[2 * r])), im(t(A_m, real[2 * r + 1]))]
        yy = top + r * (a.cell + pad)
        d.multiline_text((8, yy + a.cell // 2 - 30), f"BRSET eye\nICDR grade {g}\n({names[g]})",
                         fill=(0, 0, 0), font=f2, spacing=5)
        for c, cell in enumerate(cells):
            fig.paste(cell, (left + c * (a.cell + pad), yy))
        print(f"row {r}: grade {g}, heldout idx {i}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); fig.save(a.out)
    print(f"generator {a.run} at step {ck['step']}; wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
