"""Translator-v3: target-judge fine-tuning of the phone renderer (research/codex D095).

Measured 15 Oct (D095 evidence): frozen classifiers trained on REAL mBRSET phone photos call 79-88 %
of healthy BRSET-derived images "DR" -- real BRSET photos, physics-only, old translator and
Translator-v2 alike -- against 3-4 % on real phone photos. So the translations are still OFF the
distribution a real-phone model has learned, and the classifier trained on them learns synthetic-phone
features. This fine-tune keeps everything of scripts/129 (Translator-v2) and adds one reward:

  L_target = mean over target judges j of  BCE( f_j(y), label_smoothed(source) )      (DR, w_me * oedema)

  f_j      frozen classifiers trained on REAL mBRSET train photos (mbo_lebase_s0, s1); RAW image in
           (no colour normalisation -- the phone look is what is being judged)
  label    the BRSET label of the source eye (healthy -> must look healthy to a phone model; DR,
           including grade 1, -> must look DR to it), smoothed by --target_smooth
  guard    a THIRD real-phone classifier (mbo_lebase_s2) is never used in training; the monitor reports
           it next to the two training judges. A gain that appears only for the training judges would be
           adversarial cheating, not realism.

Total reward = lam_cons * L_cons (scripts/129, BRSET-only judge, colour-normalised: keep the source's
disease evidence) + lam_target * L_target; anchor = the ordinary v-prediction loss on real crops of both
cameras. Starts from Translator-v2 (gen_both_dc, 2,000 steps). Sources: BRSET train only (p_mbr 0); the
BRSET DR label is used for over-sampling and as the L_target target.
"""
import argparse, copy, importlib.util, json, math, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).parent
ROOT = HERE.parent
GEN = Path("/data/users3/sthummala2/brset_gen")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _load(name, mod):
    sp = importlib.util.spec_from_file_location(mod, str(HERE / name))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


C = _load("109_gen_common.py", "C109")
from diffusers import DDIMScheduler  # noqa: E402  (path set up by 109)


# ------------------------------------------------------------------ teacher
class Teacher(torch.nn.Module):
    def __init__(self, run_dir, crop=512):
        super().__init__()
        ck = torch.load(Path(run_dir) / "checkpoint-best.pth", map_location="cpu", weights_only=False)
        a = ck["args"]
        self.net = timm.create_model(a["model"], pretrained=False, num_classes=a["nb_classes"])
        self.net.load_state_dict(ck["model"])
        self.net.eval().requires_grad_(False)
        self.crop = crop
        self.register_buffer("mean", IMAGENET_MEAN.clone())
        self.register_buffer("std", IMAGENET_STD.clone())

    def forward(self, x):                         # x (B,3,560,560) in [0,1] -> logits (B,2)
        o = (x.shape[-1] - self.crop) // 2        # the trainer's eval CenterCrop(512)
        x = x[..., o:o + self.crop, o:o + self.crop]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.net((x - self.mean) / self.std).float()


def colour_norm(x, m, ref_mu, ref_sd):
    """Masked per-channel standardisation to fixed reference statistics."""
    w = m.sum((2, 3), keepdim=True).clamp_min(1.0)
    mu = (x * m).sum((2, 3), keepdim=True) / w
    sd = (((x - mu) * m) ** 2).sum((2, 3), keepdim=True).div(w).add(1e-6).sqrt()
    return (((x - mu) / sd * ref_sd + ref_mu) * m).clamp(0, 1)


# ------------------------------------------------------------------ sampler with gradients
def make_sched(steps):
    s = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
                      prediction_type="v_prediction", clip_sample=False, timestep_spacing="trailing")
    s.set_timesteps(steps)
    return s


def ddim_step(sched, v, t, x, t_prev):
    """Deterministic DDIM update for v-prediction (eta 0), written out so it is
    differentiable and stateless; matches DDIMScheduler.step for these settings."""
    ac = sched.alphas_cumprod.to(x.device)
    a_t = ac[t]
    a_p = ac[t_prev] if t_prev >= 0 else sched.final_alpha_cumprod.to(x.device)
    x0 = a_t.sqrt() * x - (1 - a_t).sqrt() * v
    eps = (1 - a_t).sqrt() * x + a_t.sqrt() * v
    return a_p.sqrt() * x0 + (1 - a_p).sqrt() * eps


def sample_with_grad(unet, x_init, s, m, strength, steps, w, k_grad, gen):
    sched = make_sched(steps)
    ts = [int(t) for t in sched.timesteps[int(round(steps * (1.0 - strength))):]]
    B = x_init.shape[0]
    noise = torch.randn(x_init.shape, generator=gen, device=x_init.device)
    a0 = sched.alphas_cumprod.to(x_init.device)[ts[0]]
    x = a0.sqrt() * (x_init * 2 - 1) + (1 - a0).sqrt() * noise
    cls = torch.full((2 * B,), C.DOM_MBRSET, device=x.device, dtype=torch.long)
    ss = torch.cat([s, torch.zeros_like(s)])
    mm = m.repeat(2, 1, 1, 1)

    def guided_v(xx, tt):
        inp = torch.cat([torch.cat([xx, xx]), ss, mm], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = unet(inp, torch.full((2 * B,), tt, device=xx.device), class_labels=cls).sample
        vc, vu = v.float().chunk(2)
        return vc + (w - 1.0) * (vc - vu)

    step_ratio = 1000 // steps
    for i, t in enumerate(ts):
        t_prev = t - step_ratio
        if i >= len(ts) - k_grad:
            v = checkpoint(guided_v, x, t, use_reentrant=False)
        else:
            with torch.no_grad():
                v = guided_v(x, t)
        x = ddim_step(sched, v, t, x, t_prev)
    return ((x + 1) / 2).clamp(0, 1) * m


# ------------------------------------------------------------------ helpers
def frames(arr, ids, dev):
    return torch.from_numpy(np.stack([np.array(arr[i]) for i in ids])).permute(0, 3, 1, 2).float().div(255).to(dev)


def allreduce_grads(params, world):
    if world == 1:
        return
    grads = [p.grad for p in params if p.grad is not None]
    flat = torch.cat([g.reshape(-1) for g in grads])
    torch.distributed.all_reduce(flat)
    flat /= world
    o = 0
    for g in grads:
        n = g.numel(); g.copy_(flat[o:o + n].view_as(g)); o += n


@torch.no_grad()
def monitor(ema, teacher, phys, mon_x, mon_y, ref, a, dev, step, out, judges=None):
    """Held-out BRSET (not used anywhere in training): teacher on source vs synthetic."""
    from sklearn.metrics import roc_auc_score
    ema.eval()
    ps, py, panel = [], [], []
    jy = {k: [] for k in (judges or {})}
    for b0 in range(0, len(mon_x), 4):
        x = mon_x[b0:b0 + 4].to(dev)
        m = C.fundus_mask(x); s = C.structure_map(x, m)
        g = torch.Generator(device=dev).manual_seed(1234 + b0)
        xi = phys(x, generator=g).clamp(0, 1)
        y = C.render(ema, s, m, x_init=xi, strength=a.strength, steps=a.ddim_steps, w_struct=a.w_struct, generator=g)
        ps.append(torch.sigmoid(teacher(colour_norm(x, m, *ref)))[:, 0].cpu())
        py.append(torch.sigmoid(teacher(colour_norm(y, m, *ref)))[:, 0].cpu())
        for k, (jm, _) in (judges or {}).items():
            jy[k].append(torch.sigmoid(jm(y))[:, 0].cpu())
        if b0 < 8:
            for k in range(min(2, x.shape[0])):
                panel.append(torch.cat([x[k], y[k]], 2))
    ps, py = torch.cat(ps).numpy(), torch.cat(py).numpy()
    r = dict(step=step, mean_abs_dp=float(np.abs(ps - py).mean()),
             auroc_source=float(roc_auc_score(mon_y, ps)), auroc_synth=float(roc_auc_score(mon_y, py)),
             mean_p_pos_source=float(ps[mon_y == 1].mean()), mean_p_pos_synth=float(py[mon_y == 1].mean()),
             mean_p_neg_source=float(ps[mon_y == 0].mean()), mean_p_neg_synth=float(py[mon_y == 0].mean()))
    for k, (_, thr) in (judges or {}).items():
        q = torch.cat(jy[k]).numpy()
        r[f"{k}_healthy_mean"] = float(q[mon_y == 0].mean())
        r[f"{k}_healthy_called_DR"] = float((q[mon_y == 0] >= thr).mean())
        r[f"{k}_DRpos_mean"] = float(q[mon_y == 1].mean())
        r[f"{k}_auroc"] = float(roc_auc_score(mon_y, q))
    print("monitor", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    img = (torch.cat(panel, 1).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    (out / "panels").mkdir(exist_ok=True)
    Image.fromarray(img).save(out / "panels" / f"tj_{step:06d}.png")
    with open(out / "monitor.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init_run", default="gen_both_dc")
    ap.add_argument("--init_ckpt", default="ema_0002000.pt")
    ap.add_argument("--target_judges", default="mbo_lebase_s0,mbo_lebase_s1", help="run names under cls_runs")
    ap.add_argument("--heldout_judge", default="mbo_lebase_s2", help="never trained against; monitor only")
    ap.add_argument("--lam_target", type=float, default=1.0)
    ap.add_argument("--target_smooth", type=float, default=0.1)
    ap.add_argument("--teacher", default=str(ROOT / "results/lebase_b0_s0"))
    ap.add_argument("--phys_ckpt", default=str(ROOT / "results/xlate_v2_noartifact/translator_last.pth"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--b_src", type=int, default=2, help="source photos per GPU per step (reward)")
    ap.add_argument("--b_diff", type=int, default=8, help="256 px crops per GPU per step (anchor)")
    ap.add_argument("--k_grad", type=int, default=4, help="DRaFT-K: last K of the 12 sampling steps carry gradients")
    ap.add_argument("--lam_cons", type=float, default=1.0)
    ap.add_argument("--lam_diff", type=float, default=1.0)
    ap.add_argument("--w_me", type=float, default=0.5, help="weight of the oedema output in L_cons")
    ap.add_argument("--p_mbr", type=float, default=0.0, help="share of reward sources drawn from mBRSET train")
    ap.add_argument("--p_pos", type=float, default=0.3, help="share of BRSET reward sources drawn from DR+ eyes")
    ap.add_argument("--strength", type=float, default=0.4)
    ap.add_argument("--ddim_steps", type=int, default=30)
    ap.add_argument("--w_struct", type=float, default=1.5)
    ap.add_argument("--colour_norm", type=int, default=1)
    ap.add_argument("--ckpt_every", type=int, default=100)
    ap.add_argument("--keep_every", type=int, default=500)
    ap.add_argument("--monitor_every", type=int, default=250)
    ap.add_argument("--n_monitor", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", type=int, default=1)
    a = ap.parse_args()
    out = GEN / "runs" / a.tag; out.mkdir(parents=True, exist_ok=True)

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        local = min(local, torch.cuda.device_count() - 1)
        torch.cuda.set_device(local)
        torch.distributed.init_process_group(os.environ.get("GEN_BACKEND", "nccl"))
    main_proc = rank == 0
    dev = torch.device("cuda")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True

    ck0 = torch.load(GEN / "runs" / a.init_run / a.init_ckpt, map_location="cpu", weights_only=False)
    gargs = dict(ck0["args"])
    unet = C.build_unet(base=gargs["base"], cond=gargs.get("cond", "structure")).to(dev)
    unet.load_state_dict(ck0["ema"])
    ema = copy.deepcopy(unet).eval().requires_grad_(False)
    opt = torch.optim.AdamW(unet.parameters(), lr=a.lr, weight_decay=1e-2)
    teacher = Teacher(a.teacher).to(dev)
    CLS = GEN / "cls_runs"
    thr_of = lambda d: json.load(open(d / "metrics_test.json"))["per_label"]["diabetic_retinopathy"]["threshold"]
    tjs = [Teacher(CLS / n).to(dev) for n in a.target_judges.split(",")]
    judges = {f"train_{n}": (j, thr_of(CLS / n)) for n, j in zip(a.target_judges.split(","), tjs)}
    judges[f"HELDOUT_{a.heldout_judge}"] = (Teacher(CLS / a.heldout_judge).to(dev), thr_of(CLS / a.heldout_judge))
    assert a.heldout_judge not in a.target_judges.split(","), "the held-out judge must not be trained against"
    S = _load("88_synth_augment.py", "S88")
    phys = S.load_translator(a.phys_ckpt, device=dev).phys
    sched_train = C.train_scheduler()
    P110 = _load("110_train_diffusion.py", "T110")

    cache = GEN / "cache"
    A_b = np.load(cache / "brset_train_560.npy", mmap_mode="r")
    A_m = np.load(cache / "mbrset_train_560.npy", mmap_mode="r")
    mb = pd.read_csv(cache / "brset_train_meta.csv", usecols=["idx", "dr", "edema"])
    lab_of = {int(i): (float(d), float(e)) for i, d, e in zip(mb.idx.values, mb.dr.values, mb.edema.values)}
    pos_b, all_b = mb[mb.dr == 1].idx.values, mb.idx.values      # DR label: sampling only
    n_m = len(A_m)

    # colour reference: masked channel mean/std averaged over 256 BRSET train photos
    rs = np.random.RandomState(0)
    xr = frames(A_b, np.sort(rs.choice(all_b, 256, replace=False)), dev)
    mr = C.fundus_mask(xr); w = mr.sum((2, 3), keepdim=True)
    mu = (xr * mr).sum((2, 3), keepdim=True) / w
    sd = (((xr - mu) * mr) ** 2).sum((2, 3), keepdim=True).div(w).sqrt()
    ref = (mu.mean(0, keepdim=True), sd.mean(0, keepdim=True))
    del xr, mr
    norm = (lambda x, m: colour_norm(x, m, *ref)) if a.colour_norm else (lambda x, m: x)

    # held-out monitor: BRSET val+test, half DR+ (grades 1-2 first: mild lesions are the ones lost)
    hm = pd.read_csv(cache / "brset_heldout_meta.csv", usecols=["idx", "dr", "grade"])
    A_h = np.load(cache / "brset_heldout_560.npy", mmap_mode="r")
    rh = np.random.RandomState(1)
    hp = rh.permutation(hm[(hm.dr == 1) & hm.grade.isin([1, 2])].idx.values)[: a.n_monitor // 2]
    hn = rh.permutation(hm[hm.dr == 0].idx.values)[: a.n_monitor // 2]
    mon_ids = np.concatenate([hp, hn])
    mon_y = np.r_[np.ones(len(hp)), np.zeros(len(hn))]
    mon_x = frames(A_h, mon_ids, torch.device("cpu"))

    if a.selftest:
        # the gradient sampler with no gradient steps must reproduce scripts/109 render()
        x = frames(A_b, pos_b[:2], dev); m = C.fundus_mask(x); s = C.structure_map(x, m)
        xi = phys(x, generator=torch.Generator(device=dev).manual_seed(5)).clamp(0, 1)
        y1 = C.render(ema, s, m, x_init=xi, strength=a.strength, steps=a.ddim_steps, w_struct=a.w_struct,
                      generator=torch.Generator(device=dev).manual_seed(7))
        with torch.no_grad():
            y2 = sample_with_grad(ema, xi, s, m, a.strength, a.ddim_steps, a.w_struct, 0,
                                  torch.Generator(device=dev).manual_seed(7))
        d = (y1 - y2).abs()
        print(f"selftest: render vs gradient sampler max |d| {d.max().item():.2e} mean {d.mean().item():.2e} "
              f"(0-1 scale; bf16 => ~1e-2 max is expected)", flush=True)
        if d.mean().item() > 2e-3:
            raise SystemExit("gradient sampler does not reproduce the translation sampler")
    step, last = 0, out / "ckpt_last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        unet.load_state_dict(ck["unet"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step = ck["step"]
        if main_proc:
            print(f"resumed from step {step}", flush=True)
    if main_proc:
        print("args:", json.dumps(vars(a)), flush=True)
        print(f"GPUs {world}; reward sources/step {a.b_src * world}; anchor crops/step {a.b_diff * world}; "
              f"teacher {a.teacher}; BRSET DR+ pool {len(pos_b)} / {len(all_b)}; mBRSET {n_m} (no labels)", flush=True)
        if step == 0:
            monitor(ema, teacher, phys, mon_x, mon_y, ref, a, dev, 0, out, judges)
    if world > 1:
        torch.distributed.barrier()

    rng = np.random.RandomState(a.seed * 1000 + rank * 7919 + step)
    gen = torch.Generator(device=dev).manual_seed(a.seed * 1000 + rank * 7919 + step)
    unet.train()
    params = list(unet.parameters())
    t0, run = time.time(), {}
    while step < a.steps:
        lr = a.lr * min(1.0, (step + 1) / a.warmup)
        for gp in opt.param_groups:
            gp["lr"] = lr
        opt.zero_grad(set_to_none=True)

        # ---- anchor: plain v-prediction on real crops of both cameras (no labels)
        n_am = int((rng.rand(a.b_diff) < 0.5).sum())
        parts = ([frames(A_m, rng.randint(0, n_m, n_am), dev)] if n_am else []) + \
                ([frames(A_b, rng.choice(all_b, a.b_diff - n_am), dev)] if n_am < a.b_diff else [])
        xa = torch.cat(parts)
        dom = torch.tensor([C.DOM_MBRSET] * n_am + [C.DOM_BRSET] * (a.b_diff - n_am), device=dev)
        with torch.no_grad():
            mf = C.fundus_mask(xa)
            sf = C.structure_map(C.degrade_for_cond(xa, 1.8, 0.015), mf)
            ys, xs = P110.pick_crops(mf, 256)
            cut = lambda t: torch.stack([t[i, :, ys[i]:ys[i] + 256, xs[i]:xs[i] + 256] for i in range(t.shape[0])])
            x0, s, m = cut(xa) * 2 - 1, cut(sf), cut(mf)
            B = x0.shape[0]
            noise = torch.randn_like(x0)
            t = torch.randint(0, 1000, (B,), device=dev)
            xt = sched_train.add_noise(x0, noise, t)
            target = sched_train.get_velocity(x0, noise, t)
            s = s * (torch.rand(B, 1, 1, 1, device=dev) >= 0.1).float()
            cls = torch.where(torch.rand(B, device=dev) < 0.1, torch.full_like(dom, C.DOM_NULL), dom)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(torch.cat([xt, s, m], 1), t, class_labels=cls).sample
        l_diff = F.mse_loss(pred.float(), target)
        (a.lam_diff * l_diff).backward()

        # ---- reward: disease consistency through the translation sampler
        from_m = rng.rand(a.b_src) < a.p_mbr
        xs_list, labs = [], []
        for fm in from_m:
            if fm:
                xs_list.append(frames(A_m, [rng.randint(n_m)], dev)); labs.append((0.0, 0.0, 0.0))
            else:
                pool = pos_b if rng.rand() < a.p_pos else all_b
                j = int(rng.choice(pool))
                xs_list.append(frames(A_b, [j], dev)); labs.append(lab_of[j] + (1.0,))
        x = torch.cat(xs_list)
        with torch.no_grad():
            m = C.fundus_mask(x); s = C.structure_map(x, m)
            xp = phys(x, generator=gen).clamp(0, 1)
            fm_t = torch.tensor(from_m, device=dev).view(-1, 1, 1, 1)
            x_init = torch.where(fm_t, x, xp)          # mBRSET: source start; BRSET: physics start
            p_src = torch.sigmoid(teacher(norm(x, m)))
        y = sample_with_grad(unet, x_init, s, m, a.strength, a.ddim_steps, a.w_struct, a.k_grad, gen)
        z = teacher(norm(y, m))
        wts = torch.tensor([1.0, a.w_me], device=dev)
        l_cons = (F.binary_cross_entropy_with_logits(z, p_src, reduction="none") * wts).sum(1).mean()
        l_cons_ex = l_cons - (F.binary_cross_entropy(p_src.clamp(1e-6, 1 - 1e-6), p_src, reduction="none")
                              * wts).sum(1).mean()      # excess over the entropy floor: 0 = identical outputs
        L = torch.tensor(labs, device=dev)                        # (B, 3): DR, oedema, has-label
        tgt = L[:, :2] * (1 - a.target_smooth) + 0.5 * a.target_smooth
        l_tgt = torch.stack([(F.binary_cross_entropy_with_logits(jm(y), tgt, reduction="none") * wts).sum(1)
                             for jm in tjs]).mean(0)
        l_tgt = (l_tgt * L[:, 2]).sum() / L[:, 2].sum().clamp_min(1.0)
        (a.lam_cons * l_cons + a.lam_target * l_tgt).backward()
        dp = (torch.sigmoid(z.detach())[:, 0] - p_src[:, 0]).abs().mean()

        if not (torch.isfinite(l_diff) and torch.isfinite(l_cons) and torch.isfinite(l_tgt)):
            raise SystemExit(f"non-finite loss at step {step}; restart resumes from the last checkpoint")
        allreduce_grads(params, world)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), unet.parameters()):
                pe.lerp_(pm.detach(), 1 - min(a.ema, (1 + step) / (10 + step)))
        for k, v in (("diff", l_diff.item()), ("cons_excess", l_cons_ex.item()), ("dp", dp.item()),
                     ("target", l_tgt.item()), ("gnorm", float(gn))):
            run[k] = v if k not in run else 0.95 * run[k] + 0.05 * v
        if main_proc and step % 10 == 0:
            r = step / max(time.time() - t0, 1e-6)
            print(f"step {step}/{a.steps} " + " ".join(f"{k} {v:.4f}" for k, v in run.items())
                  + f" lr {lr:.1e} {r:.3f} it/s", flush=True)
        if main_proc and (step % a.ckpt_every == 0 or step == a.steps):
            tmp = out / "ckpt_tmp.pt"
            torch.save({"unet": unet.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step, "args": gargs | {"tj": vars(a)}}, tmp)
            tmp.replace(last)
        if main_proc and (step % a.keep_every == 0 or step == a.steps):
            torch.save({"ema": ema.state_dict(), "step": step, "args": gargs | {"tj": vars(a)}},
                       out / f"ema_{step:07d}.pt")
            print(f"snapshot ema_{step:07d}.pt", flush=True)
        if main_proc and (step % a.monitor_every == 0 or step == a.steps):
            monitor(ema, teacher, phys, mon_x, mon_y, ref, a, dev, step, out, judges)
            unet.train()
        if world > 1 and (step % a.monitor_every == 0 or step == a.steps):
            torch.distributed.barrier()
    if main_proc:
        print("done", flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
