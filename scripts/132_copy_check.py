"""Copy (memorisation) check for synthetic images (research/codex D091 gate i).

For every query image: its MOST SIMILAR real TRAINING photo (BRSET train + mBRSET train, the
generator's training data, also mirrored, since training used left/right flips), under two
embeddings, cosine similarity:

  pixel  green channel, 64x64 (area), standardised inside the field of view -- catches
         pixel-level copies, the usual memorisation failure of diffusion models on small data
  dino   DINOv2 ViT-L/14 CLS token at 224 px (ImageNet-normalised) -- catches copies that
         differ in colour or texture
  struct the structure map (scripts/109: vessels, disc, lesions as local contrast) average-pooled
         to 140x140 -- the vessel pattern is unique to an eye and survives a change of camera,
         so this catches an eye copied from BRSET and re-rendered as a phone photo (added after
         the first control run: pixel + dino flagged only 5 % of BRSET twins, which ARE derived
         from training photos)

Reference: the same statistic for REAL mBRSET VALIDATION photos -- new patients, never in any
training set -- i.e. how similar a genuinely new phone photo is to the training set. A query
is a COPY if its similarity exceeds the 99th percentile of that reference (either embedding).
Only image files are read; no label column is parsed.

Positive controls (pass --synth_dir with them): the mBRSET re-renderings (near-copies by D089)
must be flagged almost always; BRSET twins are translations of train photos and are expected
to be flagged too. Output: results/copycheck_<tag>.json and a montage of the closest pairs.
"""
import argparse, importlib.util, json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).parent.parent
GEN = Path("/data/users3/sthummala2/brset_gen")
_sp = importlib.util.spec_from_file_location("C109", str(Path(__file__).parent / "109_gen_common.py"))
C = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(C)
EMB = ("pixel", "dino", "struct")


def struct_emb(x):
    m = C.fundus_mask(x)
    s = F.avg_pool2d(C.structure_map(x, m), 4)
    return F.normalize(s.flatten(1), dim=1).half()


def pixel_emb(x):                                  # x (B,3,H,W) in [0,1]
    g = F.interpolate(x[:, 1:2], size=(64, 64), mode="area")
    m = (F.interpolate(x.mean(1, keepdim=True), size=(64, 64), mode="area") > 0.04).float()
    n = m.sum((2, 3), keepdim=True).clamp_min(1)
    mu = (g * m).sum((2, 3), keepdim=True) / n
    sd = (((g - mu) * m) ** 2).sum((2, 3), keepdim=True).div(n).sqrt().clamp_min(1e-4)
    return F.normalize((((g - mu) / sd) * m).flatten(1), dim=1)


class Dino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                                     img_size=224).eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return F.normalize(self.net((x - self.mean) / self.std).float(), dim=1)


@torch.no_grad()
def embed(batches, dino, dev):
    P, D, S = [], [], []
    for x in batches:
        x = x.to(dev)
        P.append(pixel_emb(x).cpu()); D.append(dino(x).cpu()); S.append(struct_emb(x).cpu())
    return torch.cat(P), torch.cat(D), torch.cat(S)


def from_cache(arr, bs=64, flip=False):
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(np.array(arr[i:i + bs])).permute(0, 3, 1, 2).float().div(255)
        yield x.flip(3) if flip else x


def from_files(paths, bs=32):
    for i in range(0, len(paths), bs):
        ims = [np.asarray(Image.open(p).convert("RGB").resize((560, 560), Image.BILINEAR)) for p in paths[i:i + bs]]
        yield torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().div(255)


def nn_max(q, ref, chunk=4096):
    best, idx = torch.full((len(q),), -2.0), torch.zeros(len(q), dtype=torch.long)
    qd = q.cuda().float()
    for s in range(0, len(ref), chunk):
        sim = qd @ ref[s:s + chunk].cuda().float().T
        v, i = sim.max(1)
        upd = v.cpu() > best
        best[upd], idx[upd] = v.cpu()[upd], i.cpu()[upd] + s
    return best, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth_dir", nargs="+", required=True, help="folders of synthetic 560 px PNGs")
    ap.add_argument("--n", type=int, default=0, help="random subset per folder (0 = all)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref", default="all", choices=["all", "mbrset"],
                    help="mbrset: compare only with mBRSET train photos (+ mirrors); thresholds re-derived against that reference")
    a = ap.parse_args()
    dev = torch.device("cuda")
    dino = Dino().to(dev)
    ref_file = GEN / "copycheck_train_emb_v2.pt"
    if ref_file.exists():
        R = torch.load(ref_file)
    else:
        A_b = np.load(GEN / "cache/brset_train_560.npy", mmap_mode="r")
        A_m = np.load(GEN / "cache/mbrset_train_560.npy", mmap_mode="r")
        parts = [embed(from_cache(A, flip=f), dino, dev) for A in (A_b, A_m) for f in (False, True)]
        R = {"pixel": torch.cat([p[0] for p in parts]), "dino": torch.cat([p[1] for p in parts]),
             "struct": torch.cat([p[2] for p in parts]), "n_b": len(A_b), "n_m": len(A_m)}
        torch.save(R, ref_file)
    nb, nm = R["n_b"], R["n_m"]
    off = 0
    if a.ref == "mbrset":            # keep only the mBRSET rows [2*nb, 2*nb + 2*nm); index offset for naming
        off = 2 * nb
        R = {**R, **{e: R[e][off:] for e in EMB}}

    def src_name(i):                    # row of the stacked [BRSET, BRSET flipped, mBRSET, mBRSET flipped]
        i = i + off
        if i < 2 * nb:
            return "brset", int(i % nb), bool(i >= nb)
        j = i - 2 * nb
        return "mbrset", int(j % nm), bool(j >= nm)

    vl = pd.read_csv(ROOT / "data/finetune_joint/val/labels.csv", usecols=["file"])
    vpaths = [ROOT / "data/finetune_joint/val" / f for f in vl.file]
    V = dict(zip(EMB, embed(from_files(vpaths), dino, dev)))
    vs = {e: nn_max(V[e], R[e])[0].numpy() for e in EMB}
    thr = {e: float(np.percentile(vs[e], 99)) for e in EMB}
    # D094: how often the any-embedding rule flags genuinely NEW photos (the calibration a copy rate is read against)
    or_rate = float(np.mean(np.any(np.stack([vs[e] > thr[e] for e in EMB]), axis=0)))
    out = {"reference_val": {"n": len(vpaths), "p99": thr, "or_flag_rate": or_rate,
                             "median": {e: float(np.median(vs[e])) for e in EMB},
                             "max": {e: float(vs[e].max()) for e in EMB}}, "sets": {}}
    print("validation reference", json.dumps(out["reference_val"]), flush=True)
    rng = np.random.RandomState(0)
    for sd in a.synth_dir:
        paths = sorted(p for p in Path(sd).glob("*.png") if not p.name.startswith("."))
        if a.n and len(paths) > a.n:
            paths = [paths[i] for i in np.sort(rng.choice(len(paths), a.n, replace=False))]
        Q = dict(zip(EMB, embed(from_files(paths), dino, dev)))
        NN = {e: nn_max(Q[e], R[e]) for e in EMB}
        flag = {e: NN[e][0] > thr[e] for e in EMB}
        copy = flag["pixel"] | flag["dino"] | flag["struct"]
        s_pix, i_pix = NN["struct"]                     # montage: closest by vessel pattern
        res = {"n": len(paths), "copy_rate": float(copy.float().mean()),
               **{f"copy_rate_{e}": float(flag[e].float().mean()) for e in EMB},
               "median": {e: float(NN[e][0].median()) for e in EMB},
               "max": {e: float(NN[e][0].max()) for e in EMB},
               "copies": [p.name for p, c in zip(paths, copy.tolist()) if c]}
        out["sets"][Path(sd).name] = res
        print(Path(sd).name, json.dumps({k: v for k, v in res.items() if k != "copies"}), flush=True)
        # montage: the 6 synthetic images closest (vessel pattern) to a training photo, next to that photo
        order = torch.argsort(s_pix, descending=True)[:6].tolist()
        A = {"brset": np.load(GEN / "cache/brset_train_560.npy", mmap_mode="r"),
             "mbrset": np.load(GEN / "cache/mbrset_train_560.npy", mmap_mode="r")}
        rows = []
        for k in order:
            ds, j, fl = src_name(int(i_pix[k]))
            r = np.array(A[ds][j]); r = r[:, ::-1] if fl else r
            q = np.asarray(Image.open(paths[k]).convert("RGB"))
            rows.append(np.concatenate([np.asarray(Image.fromarray(q).resize((280, 280))),
                                        np.asarray(Image.fromarray(np.ascontiguousarray(r)).resize((280, 280)))], 1))
        (GEN / "copycheck").mkdir(exist_ok=True)
        Image.fromarray(np.concatenate(rows, 0)).save(GEN / "copycheck" / f"{a.tag}_{Path(sd).name}_closest.png")
    # full record (with per-file copy lists) stays off the public repo; results/ gets the summary only
    (GEN / "copycheck" / f"{a.tag}.json").write_text(json.dumps(out, indent=1))
    summ = {**out, "sets": {k: {kk: vv for kk, vv in v.items() if kk != "copies"} for k, v in out["sets"].items()}}
    Path(ROOT / f"results/copycheck_{a.tag}.json").write_text(json.dumps(summ, indent=1))
    print("->", ROOT / f"results/copycheck_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
