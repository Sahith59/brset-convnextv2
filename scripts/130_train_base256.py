"""Cascade stage A (research/codex D091): a grade-conditioned generator of WHOLE fundus photos.

Why: images made from a real labelled mBRSET photo are near-copies of it (D089), so they
cannot stand in for real data. A synthetic image must be NEW: drawn from noise, told only
what to show (camera, ICDR grade 0-4, macular oedema yes/no). The structure-map renderer
(scripts/110) was trained on 256 px crops and has no attention, so from noise it cannot lay
out a whole eye (one disc, one macula, vessels converging on the disc) -- the class-
conditional baseline's full-frame samples show exactly that failure. This model sees the
WHOLE frame, downsampled to 256 px, and has self-attention at the 32 and 16 px levels.
Stage B (scripts/131) renders its output at 560 px with the phone renderer.

  input   noisy RGB (3) + zero structure channel (1) + field-of-view mask (1) -- the same
          5-channel interface as gen_both, whose weights initialise every convolution
  tokens  camera {BRSET 0, mBRSET 1, null 2} + ICDR grade {0..4, null 5} + oedema
          {0, 1, null 2}; three embeddings summed into the timestep embedding. Grade and
          oedema dropped together (p 0.15) and camera (p 0.1) for classifier-free guidance.
  data    BRSET train (11,372; grade + oedema) and mBRSET train (3,402; final_icdr +
          final_edema), cameras balanced 50/50, DR+ eyes over-sampled to p_pos within each
          camera. TRAIN SPLITS ONLY (the cache holds nothing else). The 11 BRSET images whose
          grade contradicts their DR label get the null grade token.
  init    convolutions from gen_both EMA (40k); new attention output projections and the
          grade / oedema embeddings zero-initialised, so step 0 behaves like gen_both
          without structure.
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
RAW_MBRSET = Path("/home/users/sthummala2/mbrset-retfound/data/raw/labels_mbrset.csv")


def _load(name, mod):
    sp = importlib.util.spec_from_file_location(mod, str(HERE / name))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


C = _load("109_gen_common.py", "C109")
from diffusers import DDIMScheduler, UNet2DModel  # noqa: E402  (path set up by 109)

DEV_NULL, GRADE_NULL, ME_NULL = 2, 5, 2
RES = 256


class GradeCondUNet(torch.nn.Module):
    def __init__(self, unet, dim):
        super().__init__()
        self.unet = unet
        self.dev_emb = torch.nn.Embedding(3, dim)
        self.grade_emb = torch.nn.Embedding(6, dim)
        self.me_emb = torch.nn.Embedding(3, dim)
        torch.nn.init.zeros_(self.grade_emb.weight)
        torch.nn.init.zeros_(self.me_emb.weight)

    def forward(self, inp, t, labels):              # labels (B,3): camera, grade, oedema
        emb = self.dev_emb(labels[:, 0]) + self.grade_emb(labels[:, 1]) + self.me_emb(labels[:, 2])
        return self.unet(inp, t, class_labels=emb).sample


def build_model(base=128):
    mult = (1, 2, 2, 4, 4)
    unet = UNet2DModel(sample_size=RES, in_channels=5, out_channels=3, layers_per_block=2,
                       block_out_channels=tuple(base * m for m in mult),
                       down_block_types=("DownBlock2D",) * 3 + ("AttnDownBlock2D",) * 2,
                       up_block_types=("AttnUpBlock2D",) * 2 + ("UpBlock2D",) * 3,
                       add_attention=True, attention_head_dim=64, class_embed_type="identity")
    return GradeCondUNet(unet, base * mult[0] * 4)


def init_from_gen_both(model, ckpt):
    """Copy every gen_both convolution; zero the new attention outputs (residual = 0)."""
    sd = dict(torch.load(ckpt, map_location="cpu", weights_only=False)["ema"])
    dev_w = sd.pop("class_embedding.weight")
    missing, unexpected = model.unet.load_state_dict(sd, strict=False)
    if unexpected:
        raise SystemExit(f"gen_both keys with no home in the new model: {unexpected[:5]}")
    bad = [k for k in missing if "attentions" not in k]
    if bad:
        raise SystemExit(f"non-attention weights left uninitialised: {bad[:5]}")
    with torch.no_grad():
        model.dev_emb.weight.copy_(dev_w)
        for n, mod in model.unet.named_modules():
            if n.endswith("attentions.0") or n.endswith("attentions.1") or n.endswith("attentions.2"):
                torch.nn.init.zeros_(mod.to_out[0].weight); torch.nn.init.zeros_(mod.to_out[0].bias)
    return len(sd) - len(unexpected), len(missing)


def load_labels():
    """Per camera: (cache path, grade token per index, oedema token per index)."""
    cache = GEN / "cache"
    mb = pd.read_csv(cache / "brset_train_meta.csv")
    n_b = len(np.load(cache / "brset_train_560.npy", mmap_mode="r"))
    g_b = np.full(n_b, GRADE_NULL, np.int64); e_b = np.full(n_b, ME_NULL, np.int64)
    ok = mb.dr.values == (mb.grade.values >= 1)
    g_b[mb.idx.values] = np.where(ok, mb.grade.values, GRADE_NULL)
    e_b[mb.idx.values] = mb.edema.values
    mm = pd.read_csv(cache / "mbrset_train_meta.csv")
    raw = pd.read_csv(RAW_MBRSET).set_index("file").loc[mm.file]
    n_m = len(np.load(cache / "mbrset_train_560.npy", mmap_mode="r"))
    g_m = np.full(n_m, GRADE_NULL, np.int64); e_m = np.full(n_m, ME_NULL, np.int64)
    g_m[mm.idx.values] = raw.final_icdr.fillna(GRADE_NULL).astype(int).values
    e_m[mm.idx.values] = np.where(raw.final_edema.values == "yes", 1,
                                  np.where(raw.final_edema.values == "no", 0, ME_NULL))
    return {C.DOM_BRSET: (str(cache / "brset_train_560.npy"), g_b, e_b),
            C.DOM_MBRSET: (str(cache / "mbrset_train_560.npy"), g_m, e_m)}


class Frames(Dataset):
    def __init__(self, cams, length, p_pos):
        self.cams, self.length, self.p_pos, self.arrs = cams, length, p_pos, None
        self.keys = sorted(cams)
        self.pos = {k: np.where((cams[k][1] >= 1) & (cams[k][1] <= 4))[0] for k in self.keys}

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.arrs is None:
            self.arrs = {k: np.load(self.cams[k][0], mmap_mode="r") for k in self.keys}
        k = self.keys[int(torch.randint(len(self.keys), ()))]
        if float(torch.rand(())) < self.p_pos:
            j = int(self.pos[k][int(torch.randint(len(self.pos[k]), ()))])
        else:
            j = int(torch.randint(len(self.arrs[k]), ()))
        return (torch.from_numpy(np.array(self.arrs[k][j])),
                torch.tensor([k, self.cams[k][1][j], self.cams[k][2][j]], dtype=torch.long))


def to256(x):                                       # x (B,3,560,560) in [0,1]
    return F.interpolate(x, size=(RES, RES), mode="bilinear", antialias=True, align_corners=False).clamp(0, 1)


@torch.no_grad()
def sample_base(model, mask, labels, steps=50, w=2.0, generator=None):
    """DDIM from pure noise; guidance on (grade, oedema) jointly. Returns (B,3,256,256) in [0,1]."""
    dev = mask.device
    B = mask.shape[0]
    sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
                          prediction_type="v_prediction", clip_sample=False, timestep_spacing="trailing")
    sched.set_timesteps(steps, device=dev)
    x = torch.randn((B, 3, RES, RES), generator=generator, device=dev)
    zero = torch.zeros((B, 1, RES, RES), device=dev)
    unc = labels.clone(); unc[:, 1] = GRADE_NULL; unc[:, 2] = ME_NULL
    for t in sched.timesteps:
        inp = torch.cat([x, zero, mask], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(torch.cat([inp, inp]), t.expand(2 * B), torch.cat([labels, unc])).float()
        vc, vu = v.chunk(2)
        x = sched.step(vc + (w - 1.0) * (vc - vu), t, x).prev_sample
    return ((x + 1) / 2).clamp(0, 1) * mask


@torch.no_grad()
def save_panel(ema, masks, step, out, dev):
    g = torch.Generator(device=dev).manual_seed(0)
    lab = [(1, gr, 0) for gr in range(5)] + [(1, 2, 1), (0, 0, 0), (0, 2, 1)]   # mBRSET grades 0-4, + oedema, BRSET
    rows = []
    for rep in range(2):
        L = torch.tensor(lab, device=dev)
        m = masks[rep * len(lab):(rep + 1) * len(lab)]
        y = sample_base(ema, m, L, generator=g)
        rows.append(torch.cat(list(y), 2))
    img = (torch.cat(rows, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    (out / "panels").mkdir(exist_ok=True)
    Image.fromarray(img).save(out / "panels" / f"base_{step:07d}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init", default=str(GEN / "runs/gen_both/ema_0040000.pt"), help="'' = from scratch")
    ap.add_argument("--batch", type=int, default=10, help="per GPU")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--p_pos", type=float, default=0.25)
    ap.add_argument("--p_drop_label", type=float, default=0.15)
    ap.add_argument("--p_drop_device", type=float, default=0.1)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--keep_every", type=int, default=5000)
    ap.add_argument("--sample_every", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = GEN / "runs" / a.tag; out.mkdir(parents=True, exist_ok=True)

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local, ddp = int(os.environ.get("LOCAL_RANK", 0)), world > 1
    if ddp:
        local = min(local, torch.cuda.device_count() - 1)
        torch.cuda.set_device(local)
        torch.distributed.init_process_group(os.environ.get("GEN_BACKEND", "nccl"))
    main_proc = rank == 0
    torch.manual_seed(a.seed); np.random.seed(a.seed)        # identical init on every rank
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")

    model = build_model()
    if a.init:
        n_ok, n_new = init_from_gen_both(model, a.init)
        if main_proc:
            print(f"init from {a.init}: {n_ok} tensors copied, {n_new} new attention tensors", flush=True)
    model = model.to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = C.train_scheduler()
    cams = load_labels()
    if main_proc:
        print("args:", json.dumps(vars(a)), flush=True)
        print(f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M; GPUs {world}; "
              f"batch {a.batch * world}", flush=True)
        for k, (p, g, e) in cams.items():
            print(f"camera {k}: {len(g)} images, grade counts {np.bincount(g, minlength=6).tolist()} "
                  f"(index 5 = null), oedema {np.bincount(e, minlength=3).tolist()}", flush=True)
    ds = Frames(cams, length=a.steps * a.batch, p_pos=a.p_pos)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True, drop_last=True,
                    persistent_workers=True)
    A_m = np.load(cams[C.DOM_MBRSET][0], mmap_mode="r")
    masks = C.fundus_mask(to256(torch.from_numpy(np.stack([np.array(A_m[i]) for i in range(0, 3200, 200)]))
                                .permute(0, 3, 1, 2).float().div(255).to(dev)))

    step, last = 0, out / "ckpt_last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step = ck["step"]
        if main_proc:
            print(f"resumed from step {step}", flush=True)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if ddp else model
    torch.manual_seed(a.seed * 1000 + rank * 7919 + step)
    if main_proc and step == 0:
        save_panel(ema, masks, 0, out, dev)

    t0, s0, run = time.time(), step, None
    model.train()
    for frames, lab in dl:
        if step >= a.steps:
            break
        with torch.no_grad():
            x = to256(frames.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div(255))
            lab = lab.to(dev)
            fh = torch.rand(x.shape[0], device=dev) < 0.5
            x = torch.where(fh.view(-1, 1, 1, 1), x.flip(3), x)        # left/right eye mirror only
            m = C.fundus_mask(x)
            x0 = x * 2 - 1
            B = x0.shape[0]
            noise = torch.randn_like(x0)
            t = torch.randint(0, 1000, (B,), device=dev)
            xt = sched.add_noise(x0, noise, t)
            target = sched.get_velocity(x0, noise, t)
            drop_l = torch.rand(B, device=dev) < a.p_drop_label
            lab[:, 1] = torch.where(drop_l, torch.full_like(lab[:, 1], GRADE_NULL), lab[:, 1])
            lab[:, 2] = torch.where(drop_l, torch.full_like(lab[:, 2], ME_NULL), lab[:, 2])
            lab[:, 0] = torch.where(torch.rand(B, device=dev) < a.p_drop_device,
                                    torch.full_like(lab[:, 0], DEV_NULL), lab[:, 0])
            inp = torch.cat([xt, torch.zeros_like(m), m], 1)
        lr = a.lr * min(1.0, (step + 1) / a.warmup)
        for gp in opt.param_groups:
            gp["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(inp, t, lab)
        loss = F.mse_loss(pred.float(), target)
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}; restart resumes from the last checkpoint")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if not main_proc:
            continue
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm.detach(), 1 - d)
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 100 == 0:
            r = (step - s0) / max(time.time() - t0, 1e-6)
            print(f"step {step}/{a.steps} loss {run:.4f} lr {lr:.1e} {r:.2f} it/s "
                  f"eta {(a.steps - step) / max(r, 1e-6) / 3600:.1f}h", flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            tmp = out / "ckpt_tmp.pt"
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step, "args": vars(a)}, tmp)
            tmp.replace(last)
        if step % a.keep_every == 0 or step == a.steps:
            torch.save({"ema": ema.state_dict(), "step": step, "args": vars(a)}, out / f"ema_{step:07d}.pt")
            print(f"snapshot ema_{step:07d}.pt", flush=True)
        if step % a.sample_every == 0 or step == a.steps:
            save_panel(ema, masks, step, out, dev)
            print(f"panel at step {step}", flush=True)
    if main_proc:
        print("done", flush=True)
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
