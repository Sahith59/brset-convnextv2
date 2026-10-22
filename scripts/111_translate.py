"""Translate BRSET -> synthetic mBRSET with the two-stage generative translator.

    BRSET image x
      |-- stage 1, acquisition domain: 19-parameter physics model  -> x_phys
      |       sets the global colour, illumination and noise budget
      |-- structure map s(x): where the vessels and lesions are (the label lives here)
      '-- stage 2, image domain: diffusion renderer p(image | s, device=mBRSET),
              started from x_phys noised to `strength`, not from pure noise

`--init none --strength 1` is the diffusion-only ablation, `--init source` the
no-physics ablation and `--init physics_only` stage 1 alone (the physics image
the full method starts from, masked like the renderer's output, no diffusion);
together they are the three arms that show whether BOTH domains are needed,
which is the dual-domain claim.

Outputs are lossless PNG at 560 px, the classifier's own pre-crop geometry, so
the downstream loader reads them without resampling. Sharded and resumable.
"""
import argparse, importlib.util, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parent
GEN = Path("/data/users3/sthummala2/brset_gen")


def _load(name, mod):
    sp = importlib.util.spec_from_file_location(mod, str(HERE / name))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


C = _load("109_gen_common.py", "C109")


def gate_subset(meta, per_grade=80, n_neg=320, seed=0):
    """Small stratified set for the gates: every grade represented, mild included."""
    rng = np.random.RandomState(seed)
    keep = []
    for g in (1, 2, 3, 4):
        ids = meta[(meta.dr == 1) & (meta.grade == g)].idx.values
        keep.append(rng.permutation(ids)[:per_grade])
    neg = meta[(meta.dr == 0) & (meta.grade == 0)].idx.values
    keep.append(rng.permutation(neg)[:n_neg])
    return np.sort(np.concatenate(keep))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="generator run tag under runs/ (unused for physics_only)")
    ap.add_argument("--ckpt", default="ckpt_last.pt")
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--cache_tag", default="")
    ap.add_argument("--subset", default="all", choices=["all", "gate"])
    ap.add_argument("--source", default="brset", choices=["brset", "mbrset"],
                    help="mbrset = re-render mBRSET photos (D082); only the train cache exists, file names only")
    ap.add_argument("--pool", default="train", choices=["train", "heldout"],
                    help="train = the corpus the classifier will learn from. "
                         "heldout = BRSET val+test, for the gates only.")
    ap.add_argument("--init", default="physics", choices=["physics", "source", "none", "physics_only"])
    ap.add_argument("--phys_ckpt", default=str(ROOT / "results/xlate_v2_noartifact/translator_last.pth"))
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--w_struct", type=float, default=1.5)
    ap.add_argument("--w_device", type=float, default=0.0)
    ap.add_argument("--target", default="mbrset", choices=["mbrset", "brset"],
                    help="camera token to render as; brset = the D111 same-camera re-rendering control")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")

    unet, cond = None, "structure"
    if a.init != "physics_only":
        ck = torch.load(GEN / "runs" / a.run / a.ckpt, map_location="cpu", weights_only=False)
        cond = ck["args"].get("cond", "structure")
        unet = C.build_unet(base=ck["args"]["base"], cond=cond).to(dev).eval()
        unet.load_state_dict(ck["ema"])
        print(f"generator {a.run}/{a.ckpt} at step {ck['step']} (EMA weights)", flush=True)

    phys = None
    if a.init in ("physics", "physics_only"):
        S = _load("88_synth_augment.py", "S88")
        phys = S.load_translator(a.phys_ckpt, device=dev).phys   # stage 1 ONLY

    arr = np.load(GEN / "cache" / f"{a.cache_tag}{a.source}_{a.pool}_560.npy", mmap_mode="r")
    meta = pd.read_csv(GEN / "cache" / f"{a.cache_tag}{a.source}_{a.pool}_meta.csv")
    if a.source == "mbrset" and (a.subset != "all" or a.init in ("physics", "physics_only")):
        raise SystemExit("mbrset source: --subset all with --init source or none (the physics stage is BRSET -> mBRSET)")
    ids = gate_subset(meta) if a.subset == "gate" else meta.idx.values
    ids = ids[a.shard::a.n_shards]
    if a.limit:
        ids = ids[:a.limit]
    out = GEN / "synth" / a.out_tag; out.mkdir(parents=True, exist_ok=True)
    stem = lambda i: Path(str(meta.file.values[i])).stem
    todo = [i for i in ids if not (out / f"{stem(i)}.png").exists()]
    print(f"shard {a.shard}/{a.n_shards}: {len(todo)} of {len(ids)} to do -> {out}", flush=True)
    print(f"init={a.init} strength={a.strength} steps={a.steps} "
          f"w_struct={a.w_struct} w_device={a.w_device} target={a.target}", flush=True)
    target = C.DOM_BRSET if a.target == "brset" else C.DOM_MBRSET

    t0 = time.time()
    for b0 in range(0, len(todo), a.batch):
        bi = todo[b0:b0 + a.batch]
        x = torch.from_numpy(np.stack([np.array(arr[i]) for i in bi])
                             ).permute(0, 3, 1, 2).float().div(255).to(dev)
        m = C.fundus_mask(x)
        s = C.structure_map(x, m)            # the label-bearing state, from the SOURCE
        g = torch.Generator(device=dev).manual_seed(a.seed * 1_000_003 + int(bi[0]))
        x_init = None
        if a.init in ("physics", "physics_only"):
            x_init = phys(x, generator=g).clamp(0, 1)
        elif a.init == "source":
            x_init = x
        if unet is not None and cond == "class":
            # RetSyn-style baseline: a NEW image of the source's DR class, from noise;
            # only the field-of-view mask comes from the source. w_device = DR-class guidance.
            drs = torch.tensor(meta.dr.values[bi], device=dev)
            y = C.render_class(unet, m, drs, steps=a.steps, w_class=a.w_device, generator=g)
        elif a.init == "physics_only":
            y = x_init * m                   # stage 1 alone: same mask as render()
        else:
            y = C.render(unet, s, m, device_id=target, x_init=x_init,
                         strength=(1.0 if a.init == "none" else a.strength),
                         steps=a.steps, w_struct=a.w_struct, w_device=a.w_device, generator=g)
        y = (y * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for k, i in enumerate(bi):
            tmp = out / f".{stem(i)}.tmp.png"
            Image.fromarray(y[k]).save(tmp); tmp.replace(out / f"{stem(i)}.png")
        done = b0 + len(bi)
        if (b0 // a.batch) % 20 == 0 or done == len(todo):
            r = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(todo)}  {r:.2f} img/s  eta {(len(todo)-done)/max(r,1e-6)/60:.0f} min",
                  flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
