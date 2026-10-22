"""Diagnose WHY the trained translator hurts downstream accuracy.

Measured 10 Oct, zero real target training labels, seed 0:

    arm                     DR AUC    delta     val F1    delta
    baseline, no synthesis  0.9104       -      0.7774       -
    augment p=0.50          0.9044   -0.0060    0.7451   -0.0323
    augment p=0.25          0.8996   -0.0108    0.7333   -0.0441
    replacement p=1.0       0.8888   -0.0216    0.7278   -0.0496
    union (real+synthetic)  0.8652   -0.0452    0.7509   -0.0265

    noise floors: DR AUC 0.0142, DR F1 0.0199

Nothing helps and heavy synthesis clearly harms, with a dose response. This
script tests the two mechanisms that could explain it.

H1. THE TRANSLATOR DESTROYS LESIONS. The earlier sweep measured stage-1
    configurations only, never the trained stage-1-plus-stage-2 model that was
    actually used downstream. Photometric-only stage 1 already lost about 15
    percent of red-lesion candidates. If the trained model loses materially
    more, label-quality damage explains the drop directly.

H2. THE TRANSLATOR DOES NOT ACTUALLY MOVE THE IMAGES TOWARD THE TARGET IN
    FEATURE SPACE. The 13 fitted statistics match to |z| <= 0.32, but
    statistics are not appearance, which this project has now been burned by
    twice. A linear probe on frozen pretrained features is a far more sensitive
    test of whether two image sets are distinguishable. If synthetic BRSET is
    just as separable from real mBRSET as untouched BRSET was, the translator
    closed a gap that was never the operative one, and downstream harm is
    simply the cost of the distortion it introduced.

Both are measured against controls, and both are cheap and CPU-only.
"""
import argparse, importlib.util, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).parent


def _load(name, mod):
    spec = importlib.util.spec_from_file_location(mod, str(HERE / name))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load_imgs(split, n, presize, size, seed):
    d = Path(split)
    files = pd.read_csv(d / "labels.csv").file.values
    files = np.random.default_rng(seed).choice(files, min(n, len(files)), replace=False)
    out = []
    for f in files:
        try:
            im = Image.open(d / f).convert("RGB")
        except Exception:
            continue
        im = im.resize((presize, presize), Image.BILINEAR)
        o = (presize - size) // 2
        im = im.crop((o, o, o + size, o + size))
        out.append(np.asarray(im, dtype=np.float32) / 255.0)
    return out


def to_t(arrs):
    return torch.stack([torch.from_numpy(a).permute(2, 0, 1) for a in arrs])


# ---------------------------------------------------------------- H1
def h1_lesions(L, trans, imgs, args):
    """Red-lesion candidate retention for the TRAINED translator."""
    print("\n" + "=" * 70)
    print("H1: does the trained translator destroy red-lesion candidates?")
    print("=" * 70)
    rows = []
    for i, a in enumerate(imgs):
        m = L.fundus_mask_np(a)
        d = L.disc_diameter(m)
        ll = max(3, int(round(args.line_len_frac * d)))
        tol = max(2.0, args.tol_frac * d)

        def cands(arr):
            mm = L.fundus_mask_np(arr)
            resp = L.lesion_response(arr, mm, ll, args.n_orient)
            sig = L.robust_sigma(resp, mm)
            h = max(args.h_floor, max(args.h_abs, args.k_rel * sig))
            return L.detect(resp, mm, h, args.max_cand)

        c0 = cands(a)
        with torch.no_grad():
            b = trans(to_t([a])).clamp_(0, 1)[0].permute(1, 2, 0).numpy()
        c1 = cands(b)
        cen0 = c0[0] if isinstance(c0, tuple) else c0
        cen1 = c1[0] if isinstance(c1, tuple) else c1
        n0, n1 = len(cen0), len(cen1)
        kept = L.match_counts(cen0, cen1, tol) if n0 and n1 else 0
        kept = int(np.sum(kept)) if isinstance(kept, np.ndarray) else int(kept)
        new = L.match_counts(cen1, cen0, tol) if n0 and n1 else 0
        new = int(np.sum(new)) if isinstance(new, np.ndarray) else int(new)
        rows.append(dict(n_orig=n0, n_trans=n1, kept=kept,
                         spurious=(n1 - new) if n1 else 0))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(imgs)}", flush=True)
    df = pd.DataFrame(rows)
    ret = df.kept.sum() / max(df.n_orig.sum(), 1)
    spur = df.spurious.sum() / max(df.n_trans.sum(), 1)
    print(f"\n  candidates per image: original {df.n_orig.mean():.1f} -> "
          f"translated {df.n_trans.mean():.1f}")
    print(f"  RETENTION {ret:.3f}   SPURIOUS {spur:.3f}")
    print(f"  reference: identity 1.000/0.000, blur sigma3 0.005, chance 0.045")
    print(f"  stage1 photometric-only measured earlier: 0.851 retention")
    return dict(retention=float(ret), spurious=float(spur),
                cand_orig=float(df.n_orig.mean()), cand_trans=float(df.n_trans.mean()))


# ---------------------------------------------------------------- H2
def h2_separability(trans, brset, mbrset, args):
    """How distinguishable is each source set from real mBRSET?

    A logistic probe on frozen ImageNet features. AUC near 1.0 means trivially
    separable; near 0.5 means indistinguishable. The question is whether
    translation moves synthetic BRSET toward 0.5.
    """
    print("\n" + "=" * 70)
    print("H2: does translation actually move images toward the target domain?")
    print("=" * 70)
    import timm
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    enc = timm.create_model("convnextv2_nano.fcmae_ft_in22k_in1k",
                            pretrained=True, num_classes=0).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def feats(arrs, translate=False):
        out = []
        for i in range(0, len(arrs), 8):
            x = to_t(arrs[i:i + 8])
            if translate:
                x = trans(x).clamp_(0, 1)
            out.append(enc((x - mean) / std))
        return torch.cat(out).numpy()

    f_real = feats(mbrset)
    f_src = feats(brset)
    f_syn = feats(brset, translate=True)

    def probe(fa, fb, name):
        X = np.concatenate([fa, fb]); y = np.r_[np.zeros(len(fa)), np.ones(len(fb))]
        auc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0),
                              X, y, cv=5, scoring="roc_auc").mean()
        print(f"  {name:44s} AUC {auc:.4f}")
        return float(auc)

    a_src = probe(f_real, f_src, "real mBRSET vs UNTOUCHED BRSET")
    a_syn = probe(f_real, f_syn, "real mBRSET vs SYNTHETIC BRSET")
    print(f"\n  change: {a_syn - a_src:+.4f}")
    print("  1.0 = trivially separable, 0.5 = indistinguishable")
    if a_syn >= a_src - 0.01:
        print("  -> translation did NOT reduce separability. The statistics it "
              "matched are not the features that distinguish the domains.")
    else:
        print("  -> translation reduced separability, so it did move the images.")
    return dict(auc_untouched=a_src, auc_synthetic=a_syn, delta=a_syn - a_src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/xlate_v2_noartifact/translator_last.pth")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--presize", type=int, default=560)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--line_len_frac", type=float, default=0.017)
    ap.add_argument("--n_orient", type=int, default=8)
    ap.add_argument("--tol_frac", type=float, default=0.0057)
    ap.add_argument("--h_abs", type=float, default=0.012)
    ap.add_argument("--k_rel", type=float, default=3.0)
    ap.add_argument("--h_floor", type=float, default=0.004)
    ap.add_argument("--max_cand", type=int, default=800)
    ap.add_argument("--out", default="results/translator_diagnosis.json")
    a = ap.parse_args()

    S = _load("88_synth_augment.py", "S88")
    L = _load("84_lesion_retention.py", "L84")
    trans = S.load_translator(a.ckpt, device="cpu")

    brset = load_imgs("data/finetune_multilabel/train", a.n, a.presize, a.size, 0)
    mbrset = load_imgs("data/finetune_mbrset_multilabel/train", a.n, a.presize, a.size, 1)
    print(f"loaded {len(brset)} BRSET, {len(mbrset)} mBRSET at {a.size}px")

    res = {"ckpt": a.ckpt, "n": a.n}
    try:
        res["H1_lesions"] = h1_lesions(L, trans, brset, a)
    except Exception as e:
        print(f"[H1 FAILED] {type(e).__name__}: {e}")
        res["H1_lesions"] = {"error": str(e)}
    try:
        res["H2_separability"] = h2_separability(trans, brset, mbrset, a)
    except Exception as e:
        print(f"[H2 FAILED] {type(e).__name__}: {e}")
        res["H2_separability"] = {"error": str(e)}

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
