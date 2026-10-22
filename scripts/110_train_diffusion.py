"""Train the stage-2 generative renderer p(image | structure map, device).

Two variants, chosen by --domains:
  target  only real mBRSET images. The model knows one rendering, so any
          structure map it is handed comes out looking like mBRSET.
  both    BRSET and mBRSET with a device token. One retina, two renderings --
          the device-conditioned "world model" form. Sees BRSET-shaped structure
          maps in training, at the risk of inferring the device from the map and
          ignoring the token. Which variant wins is decided by the gates, not here.

No mBRSET label is read anywhere in this file, and no val/test image: the cache
holds the TRAIN split only and the mBRSET meta file has file names only.

Trained on 256 px crops of the 560 px frame. The UNet has no attention, so it is
applied to the whole frame at sampling time. Crops also mean the model sees
arbitrary field-of-view boundaries, which makes it indifferent to the framing
difference between the two datasets (STAGE2_DESIGN decision D1).
"""
import argparse, copy, importlib.util, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).parent
GEN = Path("/data/users3/sthummala2/brset_gen")


def _load(name, mod):
    sp = importlib.util.spec_from_file_location(mod, str(HERE / name))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


C = _load("109_gen_common.py", "C109")


class FullFrames(Dataset):
    """Returns one whole 560 px uint8 frame and its device id. Cropping happens
    on the GPU so the structure map can be read from the full frame first."""

    def __init__(self, paths, doms, length, labels=None):
        self.paths, self.doms, self.length, self.arrs = paths, doms, length, None
        self.labels = labels            # per path: DR label array, or None (= null token 2)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.arrs is None:                       # one memmap per worker
            self.arrs = [np.load(p, mmap_mode="r") for p in self.paths]
        k = int(torch.randint(len(self.arrs), (1,)))      # devices balanced
        j = int(torch.randint(len(self.arrs[k]), (1,)))
        dr = 2 if self.labels is None or self.labels[k] is None else int(self.labels[k][j])
        return torch.from_numpy(np.array(self.arrs[k][j])), self.doms[k], dr


def pick_crops(mask, crop, min_frac=0.3, stride=8):
    """Random crop origins, restricted to windows that are >= min_frac retina."""
    frac = F.avg_pool2d(mask, crop, stride=stride)
    B, _, h, w = frac.shape
    ok = (frac.view(B, -1) >= min_frac).float()
    ok = torch.where(ok.sum(1, keepdim=True) > 0, ok, torch.ones_like(ok))
    idx = torch.multinomial(ok, 1).squeeze(1)
    return (idx // w) * stride, (idx % w) * stride


def to_img(t):
    return (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def save_panel(ema, a, step, mon, dev):
    """rows: [real | map | re-rendered] for mBRSET, [source | map | translated] for BRSET."""
    ema.eval()
    g = torch.Generator(device=dev).manual_seed(0)
    rows = []
    for name, x in mon:
        m = C.fundus_mask(x); s = C.structure_map(x, m)
        o = (x.shape[-1] - a.crop) // 2
        sl = (slice(None), slice(None), slice(o, o + a.crop), slice(o, o + a.crop))
        y = C.render(ema, s[sl], m[sl], steps=a.panel_steps, w_struct=a.panel_w, generator=g)
        for i in range(x.shape[0]):
            sm = ((s[sl][i] + 1) / 2).repeat(3, 1, 1)
            rows.append(np.concatenate([to_img(x[sl][i]), to_img(sm), to_img(y[i])], axis=1))
    out = Path(a.out) / "panels"; out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(out / f"step_{step:07d}.png")
    # one full-frame BRSET translation: this is the resolution that gets used
    x = mon[-1][1][:1]; m = C.fundus_mask(x); s = C.structure_map(x, m)
    y = C.render(ema, s, m, steps=a.panel_steps, w_struct=a.panel_w, generator=g)
    Image.fromarray(np.concatenate([to_img(x[0]), to_img(y[0])], axis=1)).save(
        out / f"full_{step:07d}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--domains", default="target", choices=["target", "both"])
    ap.add_argument("--cond", default="structure", choices=["structure", "class"],
                    help="class = RetSyn-style BASELINE: device + DR-class tokens, no structure "
                         "map; DR labels from BRSET only, mBRSET frames get the null DR token")
    ap.add_argument("--cache_tag", default="")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--cond_blur", type=float, default=1.8,
                    help="max sigma (px) of the blur on the copy the structure map "
                         "is read from. Larger = more of the fine texture is "
                         "re-synthesised as the target device, less is copied.")
    ap.add_argument("--cond_noise", type=float, default=0.015)
    ap.add_argument("--p_drop_struct", type=float, default=0.1)
    ap.add_argument("--p_drop_device", type=float, default=0.1)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--keep_every", type=int, default=25000)
    ap.add_argument("--sample_every", type=int, default=5000)
    ap.add_argument("--panel_steps", type=int, default=40)
    ap.add_argument("--panel_w", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--lr_decay_start", type=int, default=-1,
                    help="cosine-decay the LR from this step to --steps (to lr*lr_min_ratio). "
                         "-1 = constant LR after warmup, which is how steps 0-40,000 were trained.")
    ap.add_argument("--lr_min_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out = str(GEN / "runs" / a.tag)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    # Multi-GPU: launched with `python -m torch.distributed.run --nproc_per_node=N`.
    # --batch is PER GPU. Rank 0 alone logs, keeps the EMA, checkpoints and samples.
    import faulthandler
    faulthandler.enable()
    if os.environ.get("GEN_DEBUG"):          # dump every thread's stack if we stall
        faulthandler.dump_traceback_later(75, repeat=True)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    say = (lambda msg: print(f"[rank {rank}] {msg}", flush=True)) if os.environ.get("GEN_DEBUG") \
        else (lambda msg: None)
    local, ddp = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        # under srun a task may be bound to a single GPU; never index past what is visible
        local = min(local, torch.cuda.device_count() - 1)
        torch.cuda.set_device(local)
        torch.distributed.init_process_group(os.environ.get("GEN_BACKEND", "nccl"))
        say("process group up")
    main_proc = rank == 0
    # the SAME seed on every rank here: DDP does not synchronise initial weights,
    # so each process must build an identical model. Ranks are reseeded apart
    # just before the loop.
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    if main_proc:
        print("args:", json.dumps(vars(a)), flush=True)
        print(f"GPUs: {world}  effective batch {a.batch * world}", flush=True)

    cache = GEN / "cache"
    p_m = cache / f"{a.cache_tag}mbrset_train_560.npy"
    p_b = cache / f"{a.cache_tag}brset_train_560.npy"
    paths, doms = [str(p_m)], [C.DOM_MBRSET]
    if a.domains == "both":
        paths.append(str(p_b)); doms.append(C.DOM_BRSET)
    n_m, n_b = len(np.load(p_m, mmap_mode="r")), len(np.load(p_b, mmap_mode="r"))
    if main_proc:
        print(f"training images: mBRSET {n_m}" + (f", BRSET {n_b}" if a.domains == "both" else "")
              + "  (train split only, no target labels)", flush=True)
    labels = None
    if a.cond == "class":
        if a.domains != "both":
            raise SystemExit("--cond class needs --domains both (DR classes come from BRSET)")
        mb = pd.read_csv(cache / f"{a.cache_tag}brset_train_meta.csv")
        lab_b = np.full(n_b, 2, dtype=np.int64)
        lab_b[mb.idx.values] = mb.dr.values
        labels = [None, lab_b]          # mBRSET: no label is read, null token
        if main_proc:
            print(f"cond=class: BRSET DR+ {int((lab_b == 1).sum())} / {n_b}; mBRSET frames use the null DR token", flush=True)
    ds = FullFrames(paths, doms, length=a.steps * a.batch, labels=labels)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=True)

    # fixed monitoring images: 3 mBRSET, 3 BRSET with disease (lesions to keep)
    meta_b = pd.read_csv(cache / f"{a.cache_tag}brset_train_meta.csv")
    pos = meta_b[meta_b.dr == 1].idx.values[:3]
    A_m, A_b = np.load(p_m, mmap_mode="r"), np.load(p_b, mmap_mode="r")
    f = lambda arr, ids: torch.from_numpy(np.stack([np.array(arr[i]) for i in ids])
                                          ).permute(0, 3, 1, 2).float().div(255).to(dev)
    mon = [("mbrset", f(A_m, [0, n_m // 2, n_m - 1])), ("brset", f(A_b, pos))]

    unet = C.build_unet(base=a.base, cond=a.cond).to(dev).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(unet).eval().requires_grad_(False)
    if main_proc:
        print(f"UNet parameters: {sum(p.numel() for p in unet.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(unet.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = C.train_scheduler()

    step, last = 0, Path(a.out) / "ckpt_last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        unet.load_state_dict(ck["unet"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); step = ck["step"]
        if main_proc:
            print(f"resumed from step {step}", flush=True)
    say("model on GPU, wrapping in DDP")
    net = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[local]) if ddp else unet
    say("DDP wrapped, starting data loader")
    torch.manual_seed(a.seed * 1000 + rank * 7919 + step)   # ranks draw different data

    t0, run, n_bad = time.time(), None, 0
    unet.train()
    for frames, dom, drl in dl:
        if step >= a.steps:
            break
        if step < 3:
            say(f"batch loaded (step {step})")
        x = frames.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div(255)
        dom = dom.to(dev)
        with torch.no_grad():
            m_full = C.fundus_mask(x)
            s_full = C.structure_map(C.degrade_for_cond(x, a.cond_blur, a.cond_noise), m_full)
            ys, xs = pick_crops(m_full, a.crop)
            cut = lambda t: torch.stack([t[i, :, ys[i]:ys[i] + a.crop, xs[i]:xs[i] + a.crop]
                                         for i in range(t.shape[0])])
            x0, s, m = cut(x), cut(s_full), cut(m_full)
            B = x0.shape[0]
            fh = torch.rand(B, device=dev) < 0.5; fv = torch.rand(B, device=dev) < 0.5
            flip = lambda t: torch.where(fv.view(-1, 1, 1, 1), torch.where(
                fh.view(-1, 1, 1, 1), t.flip(3), t).flip(2), torch.where(
                fh.view(-1, 1, 1, 1), t.flip(3), t))
            x0, s, m = flip(x0), flip(s), flip(m)
            x0 = x0 * 2 - 1
            noise = torch.randn_like(x0)
            t = torch.randint(0, 1000, (B,), device=dev)
            xt = sched.add_noise(x0, noise, t)
            target = sched.get_velocity(x0, noise, t)
            s = s * (torch.rand(B, 1, 1, 1, device=dev) >= a.p_drop_struct).float()
            cls = torch.where(torch.rand(B, device=dev) < a.p_drop_device,
                              torch.full_like(dom, C.DOM_NULL), dom)
            if a.cond == "class":
                s = torch.zeros_like(s)                      # no structure map at all
                drl = drl.to(dev)
                drl = torch.where(torch.rand(B, device=dev) < a.p_drop_struct,
                                  torch.full_like(drl, C.DOM_NULL), drl)
                # flips were applied to the image only; the class token is unaffected
                cls = torch.stack([cls, drl], 1)
            inp = torch.cat([xt, s, m], 1).contiguous(memory_format=torch.channels_last)

        if step < 3:
            say("inputs built, forward")
        lr = a.lr * min(1.0, (step + 1) / a.warmup)
        if 0 <= a.lr_decay_start <= step:
            import math
            frac = (step - a.lr_decay_start) / max(1, a.steps - a.lr_decay_start)
            lr = a.lr * (a.lr_min_ratio + (1 - a.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))
        for gp in opt.param_groups:
            gp["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(inp, t, class_labels=cls).sample
        loss = F.mse_loss(pred.float(), target)
        if not torch.isfinite(loss):
            if ddp:     # skipping a step on one rank would deadlock the others
                raise SystemExit(f"non-finite loss at step {step}; restart resumes "
                                 f"from the last checkpoint")
            n_bad += 1; opt.zero_grad(set_to_none=True)
            if n_bad > 50:
                raise SystemExit("more than 50 non-finite losses; stopping")
            continue
        opt.zero_grad(set_to_none=True)
        if step < 3:
            say(f"forward done, loss {loss.item():.4f}, backward")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        step += 1
        if not main_proc:
            continue
        d = min(a.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), unet.parameters()):
                pe.lerp_(pm.detach(), 1 - d)
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % a.log_every == 0:
            el = time.time() - t0
            print(f"step {step:7d}/{a.steps}  loss {run:.4f}  lr {lr:.2e}  "
                  f"{step_rate(step, el, a):s}", flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            ck = {"unet": unet.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                  "step": step, "args": vars(a)}
            tmp = Path(a.out) / "ckpt_tmp.pt"
            torch.save(ck, tmp); tmp.replace(last)
            print(f"checkpoint at step {step}", flush=True)
        # Independent of ckpt_every on purpose: nesting this inside the block above
        # silently skipped every snapshot whose step was not ALSO a multiple of
        # ckpt_every (2,500 and 7,500 were never written with ckpt_every=1000).
        if step % a.keep_every == 0 or step == a.steps:
            torch.save({"ema": ema.state_dict(), "step": step, "args": vars(a)},
                       Path(a.out) / f"ema_{step:07d}.pt")
            print(f"snapshot ema_{step:07d}.pt", flush=True)
        if step % a.sample_every == 0 or step == a.steps:
            if a.cond == "structure":
                save_panel(ema, a, step, mon, dev)
            unet.train()
            print(f"panel at step {step}", flush=True)
    if main_proc:
        print(f"done at step {step}, non-finite losses skipped: {n_bad}", flush=True)
    if ddp:
        torch.distributed.destroy_process_group()


_T = {"s0": None, "t0": None}


def step_rate(step, el, a):
    """it/s since this process started (correct across resumes) plus an ETA."""
    if _T["s0"] is None:
        _T["s0"], _T["t0"] = step, el
        return "rate --"
    r = (step - _T["s0"]) / max(el - _T["t0"], 1e-6)
    return f"{r:.2f} it/s  eta {(a.steps - step) / max(r, 1e-6) / 3600:.1f}h"


if __name__ == "__main__":
    main()
