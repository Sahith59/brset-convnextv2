"""Cascade sampling (research/codex D091): NEW synthetic phone photographs with labels.

  stage A  scripts/130 model, from pure noise at 256 px: camera = mBRSET, ICDR grade g and
           oedema e as asked, classifier-free guidance w_class on (grade, oedema).
  stage B  upsample to 560 px, read the structure map from that image and render it with the
           phone renderer (scripts/110 / 129, device = mBRSET) from the upsampled image noised to
           `strength`: the renderer supplies the full-resolution phone texture stage A cannot.

No source photograph is involved: the only thing taken from a real image is the field-of-view
outline (the black circle) of a randomly chosen mBRSET train frame, so the generated eye has a
phone-shaped frame. Labels: the list of (grade, oedema) pairs is the mBRSET TRAIN joint label
distribution repeated `mult` times (counts only, no image), shuffled with a fixed seed; the
image's DR label is grade >= 1, its oedema label is e.

Output: synth/<out_tag>/syn<id>.png (560 px), synth/<out_tag>/_base256/syn<id>.png (stage A),
synth/<out_tag>/labels_shard<k>.csv (file = absolute path, diabetic_retinopathy, macular_edema,
grade). Sharded and resumable.
"""
import argparse, importlib.util, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).parent
GEN = Path("/data/users3/sthummala2/brset_gen")


def _load(name, mod):
    sp = importlib.util.spec_from_file_location(mod, str(HERE / name))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


C = _load("109_gen_common.py", "C109")
A = _load("130_train_base256.py", "A130")


def label_plan(mult, seed):
    mm = pd.read_csv(GEN / "cache/mbrset_train_meta.csv")
    raw = pd.read_csv(A.RAW_MBRSET).set_index("file").loc[mm.file]
    g = raw.final_icdr.astype(int).values
    e = (raw.final_edema.values == "yes").astype(int)
    pairs = np.stack([np.tile(g, mult), np.tile(e, mult)], 1)
    return pairs[np.random.RandomState(seed).permutation(len(pairs))]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_run", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--render_run", default="gen_both_dc")
    ap.add_argument("--render_ckpt", default="ema_0002000.pt")
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--mult", type=int, default=3, help="label plan = mBRSET train labels x mult")
    ap.add_argument("--w_class", type=float, default=2.0)
    ap.add_argument("--base_steps", type=int, default=50)
    ap.add_argument("--strength", type=float, default=0.5, help="stage B; 0 = no stage B (upsampled only)")
    ap.add_argument("--render_steps", type=int, default=30)
    ap.add_argument("--w_struct", type=float, default=1.5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N of the plan only (pilots)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")

    ck = torch.load(GEN / "runs" / a.base_run / a.base_ckpt, map_location="cpu", weights_only=False)
    base = A.build_model().to(dev).eval()
    base.load_state_dict(ck["ema"])
    print(f"stage A {a.base_run}/{a.base_ckpt} at step {ck['step']}", flush=True)
    rend = None
    if a.strength > 0:
        ck2 = torch.load(GEN / "runs" / a.render_run / a.render_ckpt, map_location="cpu", weights_only=False)
        rend = C.build_unet(base=ck2["args"]["base"], cond=ck2["args"].get("cond", "structure")).to(dev).eval()
        rend.load_state_dict(ck2["ema"])
        print(f"stage B {a.render_run}/{a.render_ckpt} at step {ck2['step']}, strength {a.strength}", flush=True)

    plan = label_plan(a.mult, seed=1234)
    ids = np.arange(len(plan))
    if a.limit:
        ids = ids[:a.limit]
    ids = ids[a.shard::a.n_shards]
    A_m = np.load(GEN / "cache/mbrset_train_560.npy", mmap_mode="r")
    frame_of = np.random.RandomState(99).randint(0, len(A_m), size=len(plan))   # FOV outline donor
    out = GEN / "synth" / a.out_tag; (out / "_base256").mkdir(parents=True, exist_ok=True)
    name = lambda i: f"syn{i:06d}"
    todo = [i for i in ids if not (out / f"{name(i)}.png").exists()]
    print(f"shard {a.shard}/{a.n_shards}: {len(todo)} of {len(ids)} to do -> {out}", flush=True)

    t0 = time.time()
    for b0 in range(0, len(todo), a.batch):
        bi = todo[b0:b0 + a.batch]
        g = torch.Generator(device=dev).manual_seed(a.seed * 1_000_003 + int(bi[0]))
        fr = torch.from_numpy(np.stack([np.array(A_m[frame_of[i]]) for i in bi])).permute(0, 3, 1, 2).float().div(255).to(dev)
        m560 = C.fundus_mask(fr)
        m256 = C.fundus_mask(A.to256(fr))
        lab = torch.tensor([[1, int(plan[i, 0]), int(plan[i, 1])] for i in bi], device=dev)
        y256 = A.sample_base(base, m256, lab, steps=a.base_steps, w=a.w_class, generator=g)
        y = F.interpolate(y256, size=fr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1) * m560
        if rend is not None:
            s = C.structure_map(y, m560)
            y = C.render(rend, s, m560, x_init=y, strength=a.strength, steps=a.render_steps,
                         w_struct=a.w_struct, generator=g)
        y = (y * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        y2 = (y256 * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for k, i in enumerate(bi):
            tmp = out / f".{name(i)}.tmp.png"
            Image.fromarray(y[k]).save(tmp); tmp.replace(out / f"{name(i)}.png")
            Image.fromarray(y2[k]).save(out / "_base256" / f"{name(i)}.png")
        done = b0 + len(bi)
        if (b0 // a.batch) % 10 == 0 or done == len(todo):
            r = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(todo)}  {r:.2f} img/s  eta {(len(todo) - done) / max(r, 1e-6) / 60:.0f} min", flush=True)
    rows = pd.DataFrame(dict(file=[str(out / f"{name(i)}.png") for i in ids],
                             diabetic_retinopathy=(plan[ids, 0] >= 1).astype(int),
                             macular_edema=plan[ids, 1].astype(int), grade=plan[ids, 0].astype(int)))
    rows.to_csv(out / f"labels_shard{a.shard:03d}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
