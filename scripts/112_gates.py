"""Gates 1 and 2 for a translated set. A downstream run is earned, not assumed.

GATE 1, REALISM. The primary question is "did the image cross the DEVICE
boundary": a domain classifier is trained on real BRSET vs real mBRSET (images
disjoint from everything it is later asked about) and then scores the translated
images. Untouched BRSET should score ~0, fresh real mBRSET ~1; the translated
score says how far the images moved. A direct real-vs-translated probe is also
reported but is NOT the pass criterion: it flags any generated image, including
excellent ones, so it measures "is this synthetic", not "is this the right
device". Both are measured on a centre crop INSIDE the retina. The earlier probe scored whole frames, where the two datasets differ in
field-of-view shape (BRSET a clipped ellipse, mBRSET a full circle); the
translator leaves geometry alone by design, so a whole-frame probe reads ~1.0
whatever the appearance does. Whole-frame is still reported, as a secondary.

GATE 2, LABEL PRESERVATION. A translated image inherits its source label, so the
disease evidence has to survive. The judge is a classifier that is competent on
the target device (`brset_mbrset_joint`): if it can still recover the BRSET
labels from the translated images, per ICDR grade, the label-bearing content
arrived intact. This replaces the red-lesion detector, which could not separate
grade 4 from grade 0.

The judge is VALIDATED HERE before it is trusted, which the detector never was:
  - it must separate grades on the untouched images (per-grade AUC reported)
  - a deliberately lesion-destroying control (Gaussian blur, sigma 4 px) must
    make it drop. If the control does not move it, the gate is void.

The judge was trained with target labels. It is a measuring instrument only and
never enters generator training.
"""
import argparse, importlib.util, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parent
GEN = Path("/data/users3/sthummala2/brset_gen")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _load(name, mod):
    sp = importlib.util.spec_from_file_location(mod, str(HERE / name))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


C = _load("109_gen_common.py", "C109")


def crop(x, size):
    o = (x.shape[-1] - size) // 2
    return x[..., o:o + size, o:o + size]


@torch.no_grad()
def run(model, x, size, dev, bs=16):
    out = []
    for i in range(0, len(x), bs):
        xb = crop(x[i:i + bs].to(dev).float().div(255), size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model((xb - MEAN.to(dev)) / STD.to(dev)).float().cpu())
    return torch.cat(out)


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth_tag", required=True)
    ap.add_argument("--n_real", type=int, default=640)
    ap.add_argument("--judge", default="brset_mbrset_joint")
    ap.add_argument("--pool", default="heldout", choices=["train", "heldout"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    import timm
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    meta = pd.read_csv(GEN / f"cache/brset_{a.pool}_meta.csv")
    meta["stem"] = meta.file.map(lambda f: Path(str(f)).stem)
    sdir = GEN / "synth" / a.synth_tag
    have = {p.stem for p in sdir.glob("*.png")}
    sub = meta[meta.stem.isin(have)].reset_index(drop=True)
    if len(sub) < 20:
        raise SystemExit(f"only {len(sub)} translated images in {sdir}")
    A_b = np.load(GEN / f"cache/brset_{a.pool}_560.npy", mmap_mode="r")
    A_m = np.load(GEN / "cache/mbrset_train_560.npy", mmap_mode="r")
    t = lambda arrs: torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
    x_src = t([np.array(A_b[i]) for i in sub.idx])
    x_syn = t([np.asarray(Image.open(sdir / f"{s}.png").convert("RGB")) for s in sub.stem])
    rng = np.random.RandomState(a.seed)
    perm = rng.permutation(len(A_m))
    x_real = t([np.array(A_m[i]) for i in np.sort(perm[:a.n_real])])
    # disjoint sets that TRAIN the domain classifier: other real mBRSET, other BRSET
    x_real2 = t([np.array(A_m[i]) for i in np.sort(perm[a.n_real:2 * a.n_real])])
    other = np.setdiff1d(meta.idx.values, sub.idx.values)
    x_src2 = t([np.array(A_b[i]) for i in np.sort(rng.permutation(other)[:a.n_real])])
    y, grade = sub.dr.values.astype(int), sub.grade.values
    print(f"translated set '{a.synth_tag}': {len(sub)} images "
          f"(DR+ {int(y.sum())}; grades {dict(pd.Series(grade).value_counts().sort_index())})")
    res = {"synth_tag": a.synth_tag, "n": int(len(sub)), "n_real": int(len(x_real))}

    # ------------------------------------------------------------ gate 1
    print("\n" + "=" * 72 + "\nGATE 1  realism: probe AUC against real mBRSET (0.5 = indistinguishable)\n" + "=" * 72)
    enc = timm.create_model("convnextv2_nano.fcmae_ft_in22k_in1k", pretrained=True,
                            num_classes=0).to(dev).eval()

    def probe(fa, fb):
        X = np.concatenate([fa, fb]); yy = np.r_[np.zeros(len(fa)), np.ones(len(fb))]
        return float(cross_val_score(LogisticRegression(max_iter=3000, C=1.0), X, yy,
                                     cv=5, scoring="roc_auc").mean())

    res["gate1"] = {}
    for name, size in (("inside-retina crop 256 (PRIMARY)", 256), ("whole frame 512 (geometry-dominated)", 512)):
        f_real, f_src, f_syn, f_real2, f_src2 = (
            run(enc, x, size, dev).numpy() for x in (x_real, x_src, x_syn, x_real2, x_src2))
        h = len(f_real) // 2
        r = dict(floor_real_vs_real=probe(f_real[:h], f_real[h:]),
                 untouched=probe(f_real, f_src), translated=probe(f_real, f_syn))
        # device classifier: trained on images it is never asked about
        dc = LogisticRegression(max_iter=3000, C=1.0).fit(
            np.concatenate([f_src2, f_real2]), np.r_[np.zeros(len(f_src2)), np.ones(len(f_real2))])
        pm = lambda f: dc.predict_proba(f)[:, 1]
        gap = lambda f: float(np.linalg.norm(f.mean(0) - f_real.mean(0)))
        r.update(p_target_untouched=float(pm(f_src).mean()), p_target_real=float(pm(f_real).mean()),
                 p_target_translated=float(pm(f_syn).mean()),
                 frac_called_target=float((pm(f_syn) > 0.5).mean()),
                 feature_gap_remaining=gap(f_syn) / max(gap(f_src), 1e-9))
        res["gate1"][str(size)] = r
        print(f"  {name}")
        print(f"     DEVICE CLASSIFIER, mean P(mBRSET):  untouched BRSET {r['p_target_untouched']:.3f}"
              f"   real mBRSET {r['p_target_real']:.3f}   TRANSLATED {r['p_target_translated']:.3f}")
        print(f"     translated images called mBRSET:    {100 * r['frac_called_target']:.1f}%")
        print(f"     feature-mean gap remaining:         {100 * r['feature_gap_remaining']:.1f}%"
              f"   (100% = did not move, 0% = at the mBRSET mean)")
        print("     direct real-vs-set probe (flags ANY synthetic image; not the criterion):")
        print(f"     real mBRSET vs real mBRSET (floor)   {r['floor_real_vs_real']:.4f}")
        print(f"     real mBRSET vs UNTOUCHED BRSET       {r['untouched']:.4f}")
        print(f"     real mBRSET vs TRANSLATED BRSET      {r['translated']:.4f}   "
              f"({r['translated'] - r['untouched']:+.4f})")
    del enc

    # ------------------------------------------------------------ gate 2
    print("\n" + "=" * 72 + f"\nGATE 2  label preservation, judge = {a.judge}\n" + "=" * 72)
    ck = torch.load(ROOT / "results" / a.judge / "checkpoint-best.pth", map_location="cpu",
                    weights_only=False)
    judge = timm.create_model("convnextv2_large.fcmae_ft_in22k_in1k_384", pretrained=False,
                              num_classes=2).to(dev).eval()
    judge.load_state_dict(ck["model"])
    x_blur = (C.gblur(x_src.float().div(255), 4.0) * 255).round().byte()   # positive control
    P = {k: torch.sigmoid(run(judge, v, 512, dev)[:, 0]).numpy()
         for k, v in (("untouched", x_src), ("translated", x_syn), ("blur4_control", x_blur))}
    neg = y == 0
    rows = []
    for g in ("all", 1, 2, 3, 4):
        m = np.ones(len(y), bool) if g == "all" else (neg | (grade == g))
        rows.append(dict(grade=g, n_pos=int((y[m] == 1).sum()),
                         **{k: auc(y[m], P[k][m]) for k in P}))
    df = pd.DataFrame(rows)
    df["kept"] = df.translated / df.untouched
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    res["gate2"] = dict(rows=json.loads(df.to_json(orient="records")),
                        mean_abs_dp=float(np.abs(P["translated"] - P["untouched"]).mean()),
                        mean_p_pos=dict(untouched=float(P["untouched"][~neg].mean()),
                                        translated=float(P["translated"][~neg].mean())),
                        mean_p_neg=dict(untouched=float(P["untouched"][neg].mean()),
                                        translated=float(P["translated"][neg].mean())))
    a_all = df[df.grade == "all"].iloc[0]
    ok_judge = a_all.untouched - a_all.blur4_control > 0.05
    print(f"\n  judge validity: blur control drops AUC by {a_all.untouched - a_all.blur4_control:+.4f} "
          f"-> {'judge is sensitive, gate is valid' if ok_judge else 'JUDGE IS NOT SENSITIVE, GATE VOID'}")
    print(f"  mean judge score on positives: untouched {res['gate2']['mean_p_pos']['untouched']:.3f} "
          f"-> translated {res['gate2']['mean_p_pos']['translated']:.3f}")
    print(f"  mean judge score on negatives: untouched {res['gate2']['mean_p_neg']['untouched']:.3f} "
          f"-> translated {res['gate2']['mean_p_neg']['translated']:.3f}   (a rise = invented disease)")
    res["judge_valid"] = bool(ok_judge)
    out = ROOT / "results" / f"gates_{a.synth_tag}.json"
    out.write_text(json.dumps(res, indent=2)); print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
