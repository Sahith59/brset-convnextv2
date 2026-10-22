"""CycleGAN baselines for BRSET <-> mBRSET (research/codex D107).

Closest prior work: Residual-CycleGAN camera adaptation for DR screening (Yang et al., MICCAI 2020).
Two variants, otherwise identical:
  plain     standard CycleGAN (Zhu et al., ICCV 2017): ResNet-9 generators, 70x70 PatchGAN,
            LSGAN loss, cycle weight 10, identity weight 5, image pool 50, Adam 2e-4 (0.5, 0.999),
            constant LR for the first half of training and linear decay to 0 over the second half.
  residual  the generator predicts a residual added to its input (y = clip(x + tanh(r))), last conv
            zero-initialised so training starts at the identity, as in Residual-CycleGAN. The
            camera-oriented feature loss of that paper is NOT reproduced.

Data: the same 560 px caches as the diffusion renderer and the FDA / Reinhard twins
(brset_gen/cache/{brset,mbrset}_train_560.npy; train splits only, no labels). Training uses random
256 px crops (the renderer's crop size); translation runs the fully convolutional generator on the
whole 560 px image and applies the source retina mask, the same output convention as
scripts/123_classic_twins.py (<stem>.png).

Modes
  train            --variant {plain,residual} --iters 50000 --batch 4   (resumes from latest.pt)
  translate_ab     BRSET train -> phone style  -> brset_gen/synth/full_cyclegan_<variant>/<stem>.png
  translate_ba     mBRSET val/test -> tabletop style (Residual-CycleGAN's test-time protocol)
                   -> brset_gen/cgtt/<variant>/{val,test}/<stem>.png ; --identity writes the
                   untranslated 560 px images to brset_gen/cgtt/ident/ (the control arm).
"""
import argparse, json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
G = Path("/data/users3/sthummala2/brset_gen")
RUNS = G / "cyclegan"


# ---------------------------------------------------------------- networks (Zhu et al. 2017)
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c), nn.ReLU(True),
                               nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c))

    def forward(self, x):
        return x + self.b(x)


class Generator(nn.Module):
    def __init__(self, residual=False, ngf=64, n_blocks=9):
        super().__init__()
        self.residual = residual
        L = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        for m in (1, 2):
            L += [nn.Conv2d(ngf * m, ngf * m * 2, 3, 2, 1), nn.InstanceNorm2d(ngf * m * 2), nn.ReLU(True)]
        L += [ResBlock(ngf * 4) for _ in range(n_blocks)]
        for m in (4, 2):
            L += [nn.ConvTranspose2d(ngf * m, ngf * m // 2, 3, 2, 1, output_padding=1),
                  nn.InstanceNorm2d(ngf * m // 2), nn.ReLU(True)]
        self.last = nn.Conv2d(ngf, 3, 7)
        L += [nn.ReflectionPad2d(3), self.last]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        r = torch.tanh(self.net(x))
        return torch.clamp(x + r, -1, 1) if self.residual else r


class PatchD(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), nn.InstanceNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), nn.InstanceNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1), nn.InstanceNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1))

    def forward(self, x):
        return self.net(x)


def init_normal(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_gen(variant):
    g = Generator(residual=(variant == "residual"))
    g.apply(init_normal)
    if variant == "residual":          # start at the identity mapping
        nn.init.zeros_(g.last.weight); nn.init.zeros_(g.last.bias)
    return g


class Pool:
    """Image buffer of Shrivastava et al. / Zhu et al.: D sees a mix of current and past fakes."""
    def __init__(self, size=50):
        self.size, self.items = size, []

    def __call__(self, imgs):
        out = []
        for im in imgs.detach():
            im = im.unsqueeze(0)
            if len(self.items) < self.size:
                self.items.append(im); out.append(im)
            elif random.random() < 0.5:
                k = random.randrange(self.size); out.append(self.items[k].clone()); self.items[k] = im
            else:
                out.append(im)
        return torch.cat(out)


# ---------------------------------------------------------------- data
class Crops(torch.utils.data.IterableDataset):
    """Endless unpaired random 256 px crops (+ horizontal flip) from one 560 px cache."""
    def __init__(self, path, crop=256, seed=0):
        self.path, self.crop, self.seed = str(path), crop, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed * 1000 + (wi.id if wi else 0) + int(time.time()) % 100000)
        arr = np.load(self.path, mmap_mode="r")
        n, h, w, _ = arr.shape
        while True:
            i = rng.integers(n); y = rng.integers(h - self.crop + 1); x = rng.integers(w - self.crop + 1)
            c = np.ascontiguousarray(arr[i, y:y + self.crop, x:x + self.crop])
            if rng.random() < 0.5:
                c = c[:, ::-1]
            t = torch.from_numpy(np.ascontiguousarray(c)).permute(2, 0, 1).float() / 127.5 - 1.0
            yield t


def loader(path, batch, seed, workers):
    return iter(torch.utils.data.DataLoader(Crops(path, seed=seed), batch_size=batch,
                                            num_workers=workers, pin_memory=True, persistent_workers=True))


# ---------------------------------------------------------------- train
def train(a):
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    out = RUNS / (a.run_name or a.variant); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    G_ab, G_ba = build_gen(a.variant).to(dev), build_gen(a.variant).to(dev)   # A = BRSET, B = phone
    D_a, D_b = PatchD().to(dev), PatchD().to(dev)
    D_a.apply(init_normal); D_b.apply(init_normal)
    opt_g = torch.optim.Adam(list(G_ab.parameters()) + list(G_ba.parameters()), lr=a.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(list(D_a.parameters()) + list(D_b.parameters()), lr=a.lr, betas=(0.5, 0.999))
    lr_at = lambda it: a.lr * (1.0 if it < a.iters // 2 else max(0.0, (a.iters - it) / (a.iters - a.iters // 2)))
    start = 0
    if (out / "latest.pt").exists():
        ck = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        for net, k in ((G_ab, "G_ab"), (G_ba, "G_ba"), (D_a, "D_a"), (D_b, "D_b")):
            net.load_state_dict(ck[k])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"]); start = ck["it"]
        print(f"resumed from iteration {start}", flush=True)
    pool_a, pool_b = Pool(), Pool()
    la = loader(G / "cache/brset_train_560.npy", a.batch, a.seed, a.workers)
    lb = loader(G / "cache/mbrset_train_560.npy", a.batch, a.seed + 1, a.workers)
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    print(f"CycleGAN {a.variant}: iters {a.iters}, batch {a.batch}, lr {a.lr}, cycle {a.lam_cyc}, "
          f"identity {a.lam_id}; G params {sum(p.numel() for p in G_ab.parameters())}", flush=True)
    t0, acc = time.time(), {}
    for it in range(start, a.iters):
        for o in (opt_g, opt_d):
            for gp in o.param_groups:
                gp["lr"] = lr_at(it)
        ra, rb = next(la).to(dev, non_blocking=True), next(lb).to(dev, non_blocking=True)
        # generators
        for p in list(D_a.parameters()) + list(D_b.parameters()):
            p.requires_grad_(False)
        fb, fa = G_ab(ra), G_ba(rb)
        pb, pa = D_b(fb), D_a(fa)
        l_gan = mse(pb, torch.ones_like(pb)) + mse(pa, torch.ones_like(pa))
        l_cyc = l1(G_ba(fb), ra) + l1(G_ab(fa), rb)
        l_id = l1(G_ab(rb), rb) + l1(G_ba(ra), ra)
        lg = l_gan + a.lam_cyc * l_cyc + a.lam_id * l_id
        opt_g.zero_grad(set_to_none=True); lg.backward(); opt_g.step()
        # discriminators
        for p in list(D_a.parameters()) + list(D_b.parameters()):
            p.requires_grad_(True)
        qa, qb = pool_a(fa), pool_b(fb)
        da_r, da_f, db_r, db_f = D_a(ra), D_a(qa), D_b(rb), D_b(qb)
        ld = 0.5 * (mse(da_r, torch.ones_like(da_r)) + mse(da_f, torch.zeros_like(da_f))) + \
             0.5 * (mse(db_r, torch.ones_like(db_r)) + mse(db_f, torch.zeros_like(db_f)))
        opt_d.zero_grad(set_to_none=True); ld.backward(); opt_d.step()
        for k, v in (("gan", l_gan), ("cyc", l_cyc), ("id", l_id), ("d", ld)):
            acc[k] = acc.get(k, 0.0) + float(v)
        if not np.isfinite(float(lg)) or not np.isfinite(float(ld)):
            raise SystemExit(f"non-finite loss at iteration {it}")
        if (it + 1) % a.log_every == 0:
            dt = time.time() - t0
            print(f"it {it + 1} " + " ".join(f"{k} {v / a.log_every:.4f}" for k, v in acc.items()) +
                  f" lr {lr_at(it):.2e} {a.log_every / dt:.2f} it/s", flush=True)
            t0, acc = time.time(), {}
        if (it + 1) % a.ckpt_every == 0 or it + 1 == a.iters:
            ck = {"G_ab": G_ab.state_dict(), "G_ba": G_ba.state_dict(), "D_a": D_a.state_dict(),
                  "D_b": D_b.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                  "it": it + 1, "args": vars(a)}
            torch.save(ck, out / ".latest.tmp"); (out / ".latest.tmp").replace(out / "latest.pt")
            torch.save({"G_ab": ck["G_ab"], "G_ba": ck["G_ba"], "it": it + 1, "variant": a.variant},
                       out / f"gen_{it + 1}.pt")
    torch.save({"G_ab": G_ab.state_dict(), "G_ba": G_ba.state_dict(), "it": a.iters, "variant": a.variant},
               out / "gen_final.pt")
    print(f"done: {out / 'gen_final.pt'}", flush=True)


# ---------------------------------------------------------------- translate
def mask_of(x):
    return x.astype(np.float32).mean(-1) > 0.04 * 255          # same rule as scripts/123_classic_twins.py


@torch.no_grad()
def run_gen(g, imgs, dev):
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1.0
    y = g(x).clamp(-1, 1)
    y = ((y + 1.0) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    return [(yi * mask_of(xi)[..., None]).astype(np.uint8) for yi, xi in zip(y, imgs)]


def save_png(arr, p):
    tmp = p.parent / f".{p.stem}.tmp.png"
    Image.fromarray(arr).save(tmp); tmp.replace(p)


def load_gen(variant, key, dev, run_name=""):
    ck = torch.load(RUNS / (run_name or variant) / "gen_final.pt", map_location="cpu", weights_only=False)
    g = Generator(residual=(variant == "residual")); g.load_state_dict(ck[key]); return g.to(dev).eval()


def translate_ab(a):
    dev = torch.device("cuda")
    g = load_gen(a.variant, "G_ab", dev, a.run_name)
    out = G / "synth" / f"full_cyclegan_{a.run_name or a.variant}"; out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(G / "cache/brset_train_meta.csv")
    arr = np.load(G / "cache/brset_train_560.npy", mmap_mode="r")
    todo = [(int(i), Path(str(f)).stem) for i, f in zip(meta.idx, meta.file) if not (out / f"{Path(str(f)).stem}.png").exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"translate_ab {a.variant}: {len(todo)} of {len(meta)} to do -> {out}", flush=True)
    for k in range(0, len(todo), a.batch):
        chunk = todo[k:k + a.batch]
        ys = run_gen(g, [np.asarray(arr[i]) for i, _ in chunk], dev)
        for (_, stem), y in zip(chunk, ys):
            save_png(y, out / f"{stem}.png")
        if k % 2000 < a.batch:
            print(f"  {k}/{len(todo)}", flush=True)
    print(f"translate_ab {a.variant}: {len(list(out.glob('img*.png')))} twins in {out}", flush=True)


def load560(path):
    for k in range(5):
        try:
            return np.asarray(Image.open(path).convert("RGB").resize((560, 560), Image.BILINEAR), np.uint8)
        except (OSError, IOError):
            time.sleep(0.5 * (k + 1))
    raise RuntimeError(f"cannot read {path}")


def translate_ba(a):
    """Phone val/test photos -> tabletop style, resized exactly like the 560 px caches (PIL bilinear),
    for evaluating the BRSET-only classifier at test time. --identity: no generator (control)."""
    dev = torch.device("cuda")
    g = None if a.identity else load_gen(a.variant, "G_ba", dev, a.run_name)
    tag = "ident" if a.identity else (a.run_name or a.variant)
    for sp in ("val", "test"):
        src_dir = ROOT / "data/finetune_joint" / sp
        lab = pd.read_csv(src_dir / "labels.csv")
        out = G / "cgtt" / tag / sp; out.mkdir(parents=True, exist_ok=True)
        files = list(lab.file.astype(str))[:a.limit or None]
        lab = lab.iloc[:len(files)]
        for k in range(0, len(files), a.batch):
            chunk = files[k:k + a.batch]
            xs = [load560(src_dir / f) for f in chunk]
            ys = xs if g is None else run_gen(g, xs, dev)
            for f, y in zip(chunk, ys):
                save_png(y, out / f"{Path(f).stem}.png")
        new = lab.copy(); new["file"] = [str(out / f"{Path(f).stem}.png") for f in files]
        new.to_csv(out / "labels.csv", index=False)
        print(f"translate_ba {tag} {sp}: {len(files)} images -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["train", "translate_ab", "translate_ba"])
    ap.add_argument("--variant", default="plain", choices=["plain", "residual"])
    ap.add_argument("--iters", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam_cyc", type=float, default=10.0)
    ap.add_argument("--lam_id", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--run_name", default="", help="output folder name (default: the variant); smoke tests use their own")
    ap.add_argument("--limit", type=int, default=0, help="translate only the first N images (smoke test)")
    a = ap.parse_args()
    {"train": train, "translate_ab": translate_ab, "translate_ba": translate_ba}[a.mode](a)


if __name__ == "__main__":
    main()
