"""C6 lesion painter (research/codex D097): teach the full-resolution phone renderer to DRAW mild-DR lesions.

Why: every BRSET-derived synthetic set stops at real-BRSET level; the missing knowledge is mild DR (grade 1) as
mBRSET's graders see it (validation: synthetic models catch 42-44 % of grade-1 eyes, the target 64 %). The cascade
could not draw it (256 px: a microaneurysm is smaller than a pixel). This model works at the native 560-px scale.

  network  Translator-v2's renderer (gen_both_dc, 104.7 M, no attention) + grade {0-4, null} and oedema {0, 1, null}
           embeddings (zero-initialised) added to the camera token: identical to Translator-v2 at step 0.
  input    noisy image + LESION-SUPPRESSED structure map + mask. Small round dark spots (microaneurysms, dot
           haemorrhages) are erased from the map before it is computed: green channel closed with 12 line
           structuring elements (15 px) and the minimum taken -- vessels survive (one line lies along them),
           round spots smaller than the line do not. So the lesions must come from the GRADE token.
  data     BRSET train + mBRSET train photos WITH grade and oedema labels (train splits only); 384-px crops at
           native scale (>= 30 % retina); per sample: grade-1 eye with p 0.2, other DR+ eye with p 0.3, else any.
  loss     ordinary v-prediction MSE; grade/oedema dropped together (p 0.15), camera (p 0.1), map (p 0.1).
  NO judge or classifier is used anywhere (lesson of D095/D096: no validation-selected model inside a generator).
"""
import argparse, copy, importlib.util, json, os, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).parent
GEN = Path("/data/users3/sthummala2/brset_gen")


def _load(name, mod):
    sp = importlib.util.spec_from_file_location(mod, str(HERE / name))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


C = _load("109_gen_common.py", "C109")
A = _load("130_train_base256.py", "A130")      # GradeCondUNet, load_labels, null-token constants
P110 = _load("110_train_diffusion.py", "T110")  # pick_crops

_LINES = []
for k in range(12):
    th = np.pi * k / 12; L = 15
    ker = np.zeros((L, L), np.uint8); c = L // 2
    for t in np.linspace(-c, c, 4 * L):
        ker[int(round(c + t * np.sin(th))), int(round(c + t * np.cos(th)))] = 1
    _LINES.append(ker)


def suppress_dark_spots(green_u8):
    """Erase small round dark spots, keep vessels: min over 12 oriented line closings."""
    return np.min(np.stack([cv2.morphologyEx(green_u8, cv2.MORPH_CLOSE, k) for k in _LINES]), 0)


def build_painter(init_ckpt):
    unet = C.build_unet(cond="class").unet                     # same layout as gen_both, identity class embedding
    model = A.GradeCondUNet(unet, 512)
    if init_ckpt:
        sd = dict(torch.load(init_ckpt, map_location="cpu", weights_only=False)["ema"])
        dev_w = sd.pop("class_embedding.weight")
        missing, unexpected = model.unet.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise SystemExit(f"renderer weights do not match: missing {missing[:3]} unexpected {unexpected[:3]}")
        with torch.no_grad():
            model.dev_emb.weight.copy_(dev_w)
    return model


class Frames(Dataset):
    def __init__(self, cams, length, p_g1, p_pos):
        self.cams, self.length, self.p_g1, self.p_pos, self.arrs = cams, length, p_g1, p_pos, None
        self.keys = sorted(cams)
        self.g1 = {k: np.where(cams[k][1] == 1)[0] for k in self.keys}
        self.pos = {k: np.where((cams[k][1] >= 1) & (cams[k][1] <= 4))[0] for k in self.keys}

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.arrs is None:
            self.arrs = {k: np.load(self.cams[k][0], mmap_mode="r") for k in self.keys}
        k = self.keys[int(torch.randint(len(self.keys), ()))]
        u = float(torch.rand(()))
        pool = self.g1[k] if u < self.p_g1 else (self.pos[k] if u < self.p_g1 + self.p_pos else None)
        j = int(pool[int(torch.randint(len(pool), ()))]) if pool is not None else int(torch.randint(len(self.arrs[k]), ()))
        x = np.array(self.arrs[k][j])
        g_supp = suppress_dark_spots(x[..., 1])
        return (torch.from_numpy(x), torch.from_numpy(g_supp),
                torch.tensor([k, self.cams[k][1][j], self.cams[k][2][j]], dtype=torch.long))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init", default=str(GEN / "runs/gen_both_dc/ema_0002000.pt"))
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--batch", type=int, default=5, help="per GPU")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--p_g1", type=float, default=0.2)
    ap.add_argument("--p_pos", type=float, default=0.3)
    ap.add_argument("--p_drop_label", type=float, default=0.15)
    ap.add_argument("--p_drop_device", type=float, default=0.1)
    ap.add_argument("--p_drop_struct", type=float, default=0.1)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--keep_every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lesion_weight", type=float, default=0.0,
                    help="v2 (D098): extra loss weight on the pixels whose structure-map value the spot erasure changed")
    ap.add_argument("--emb_lr_mult", type=float, default=1.0, help="v2: learning-rate multiplier for grade/oedema embeddings")
    ap.add_argument("--lesion_weight_dr_only", type=int, default=0,
                    help="v3 (D101): apply the erased-spot weight only to photos labelled DR (grade 1-4), so spots are tied to the grade")
    a = ap.parse_args()
    out = GEN / "runs" / a.tag; out.mkdir(parents=True, exist_ok=True)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local, ddp = int(os.environ.get("LOCAL_RANK", 0)), world > 1
    if ddp:
        local = min(local, torch.cuda.device_count() - 1); torch.cuda.set_device(local)
        torch.distributed.init_process_group(os.environ.get("GEN_BACKEND", "nccl"))
    main_proc = rank == 0
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")

    model = build_painter(a.init).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    emb = [model.grade_emb.weight, model.me_emb.weight]
    rest = [p_ for n_, p_ in model.named_parameters() if not n_.startswith(("grade_emb", "me_emb"))]
    opt = torch.optim.AdamW([{"params": rest, "mult": 1.0}, {"params": emb, "mult": a.emb_lr_mult, "weight_decay": 0.0}],
                            lr=a.lr, weight_decay=1e-2)
    sched = C.train_scheduler()
    cams = A.load_labels()
    if main_proc:
        print("args:", json.dumps(vars(a)), flush=True)
        print(f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M; GPUs {world}; batch {a.batch * world}", flush=True)
        for k, (p, g, e) in cams.items():
            print(f"camera {k}: {len(g)} images, grade counts {np.bincount(g, minlength=6).tolist()}, oedema {np.bincount(e, minlength=3).tolist()}", flush=True)
    dl = DataLoader(Frames(cams, a.steps * a.batch, a.p_g1, a.p_pos), batch_size=a.batch, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    step, last = 0, out / "ckpt_last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        if main_proc:
            print(f"resumed from step {step}", flush=True)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if ddp else model
    torch.manual_seed(a.seed * 1000 + rank * 7919 + step)
    t0, s0, run = time.time(), step, None
    model.train()
    for frames, gsupp, lab in dl:
        if step >= a.steps:
            break
        with torch.no_grad():
            x = frames.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div(255)
            xs = x.clone(); xs[:, 1:2] = gsupp.to(dev).unsqueeze(1).float().div(255)     # green with spots erased
            lab = lab.to(dev)
            is_dr = ((lab[:, 1] >= 1) & (lab[:, 1] <= 4)).float().view(-1, 1, 1, 1)   # before label dropout
            m = C.fundus_mask(x)
            s_full = C.structure_map(C.degrade_for_cond(xs, 1.8, 0.015), m)
            if a.lesion_weight > 0:     # where the erasure changed the map = where lesions were removed (v2)
                ch = (C.structure_map(x, m) - C.structure_map(xs, m)).abs()
                spot = (F.max_pool2d(ch, 5, 1, 2) > 0.15).float() * m
                if a.lesion_weight_dr_only:
                    spot = spot * is_dr
            else:
                spot = torch.zeros_like(m)
            ys, xx = P110.pick_crops(m, a.crop)
            cut = lambda t: torch.stack([t[i, :, ys[i]:ys[i] + a.crop, xx[i]:xx[i] + a.crop] for i in range(t.shape[0])])
            x0, s, mc, sp = cut(x) * 2 - 1, cut(s_full), cut(m), cut(spot)
            B = x0.shape[0]
            fh = torch.rand(B, device=dev) < 0.5
            flip = lambda t: torch.where(fh.view(-1, 1, 1, 1), t.flip(3), t)
            x0, s, mc, sp = flip(x0), flip(s), flip(mc), flip(sp)
            noise = torch.randn_like(x0); t = torch.randint(0, 1000, (B,), device=dev)
            xt = sched.add_noise(x0, noise, t); target = sched.get_velocity(x0, noise, t)
            s = s * (torch.rand(B, 1, 1, 1, device=dev) >= a.p_drop_struct).float()
            dl_ = torch.rand(B, device=dev) < a.p_drop_label
            lab[:, 1] = torch.where(dl_, torch.full_like(lab[:, 1], A.GRADE_NULL), lab[:, 1])
            lab[:, 2] = torch.where(dl_, torch.full_like(lab[:, 2], A.ME_NULL), lab[:, 2])
            lab[:, 0] = torch.where(torch.rand(B, device=dev) < a.p_drop_device, torch.full_like(lab[:, 0], A.DEV_NULL), lab[:, 0])
            inp = torch.cat([xt, s, mc], 1)
        lr = a.lr * min(1.0, (step + 1) / a.warmup)
        for gp in opt.param_groups:
            gp["lr"] = lr * gp["mult"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(inp, t, lab)
        wmap = 1.0 + a.lesion_weight * sp
        loss = (((pred.float() - target) ** 2) * wmap).sum() / (wmap.sum() * pred.shape[1])
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
        if not main_proc:
            continue
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm.detach(), 1 - d)
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 100 == 0:
            r = (step - s0) / max(time.time() - t0, 1e-6)
            print(f"step {step}/{a.steps} loss {run:.4f} lr {lr:.1e} {r:.2f} it/s eta {(a.steps - step) / max(r, 1e-6) / 3600:.1f}h", flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            tmp = out / "ckpt_tmp.pt"
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(a)}, tmp)
            tmp.replace(last)
        if step % a.keep_every == 0 or step == a.steps:
            torch.save({"ema": ema.state_dict(), "step": step, "args": vars(a)}, out / f"ema_{step:07d}.pt")
            print(f"snapshot ema_{step:07d}.pt", flush=True)
    if main_proc:
        print("done", flush=True)
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
