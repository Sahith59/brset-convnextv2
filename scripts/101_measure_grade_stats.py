"""Calibration for lesion synthesis: what does a REAL grade-1 image look like
to the red-lesion detector?

Synthesis has to hit a target, and the target cannot be invented. ICDR grade 1
is DEFINED as "microaneurysms only", so a healthy retina plus a few
microaneurysms is a grade-1 image by the grading standard itself -- which is
what makes the synthetic label correct by construction rather than assumed.

What has to be measured before writing the synthesiser:
  - how many candidates a real grade-0 image already shows (the floor: the
    detector fires on vessel crossings and noise, not only on lesions)
  - how many a real grade-1 image shows (the target)
  - the CONTRAST and AREA of those candidates (so pasted lesions are the right
    size and darkness, not merely the right count)

Detector parameters are the ones validated in 90_diagnose_translator.py.
"""
import argparse, importlib.util as ilu, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
BRSET_CSV = ("/data/users4/nshaik3/Datasets/BRSET/physionet.org/files/"
             "brazilian-ophthalmological/1.0.1/labels_brset.csv")
IMG_DIR = ("/data/users4/nshaik3/Datasets/BRSET/physionet.org/files/"
           "brazilian-ophthalmological/1.0.1/fundus_photos")


def _load(name):
    sp = ilu.spec_from_file_location(name.split(".")[0], str(ROOT / "scripts" / name))
    m = ilu.module_from_spec(sp); sp.loader.exec_module(m); return m


def load_img(path, presize, size):
    im = Image.open(path).convert("RGB")
    im = im.resize((presize, presize), Image.BILINEAR)
    o = (presize - size) // 2
    im = im.crop((o, o, o + size, o + size))
    return np.asarray(im, np.float32) / 255.0


def candidates(L, arr, a):
    """Return (centres, peaks, areas) with the validated detector settings."""
    m = L.fundus_mask_np(arr)
    d = L.disc_diameter(m)
    ll = max(3, int(round(a.line_len_frac * d)))
    resp = L.lesion_response(arr, m, ll, a.n_orient)
    sig = L.robust_sigma(resp, m)
    h = max(a.h_floor, max(a.h_abs, a.k_rel * sig))
    out = L.detect(resp, m, h, a.max_cand)
    cen, peak, area = out[0], out[1], out[2]
    return cen, peak, area, m, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="images per grade")
    ap.add_argument("--grades", default="0,1,2")
    ap.add_argument("--presize", type=int, default=560)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--line_len_frac", type=float, default=0.017)
    ap.add_argument("--n_orient", type=int, default=8)
    ap.add_argument("--h_abs", type=float, default=0.012)
    ap.add_argument("--k_rel", type=float, default=3.0)
    ap.add_argument("--h_floor", type=float, default=0.004)
    ap.add_argument("--max_cand", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/grade_candidate_stats.json"))
    a = ap.parse_args()

    L = _load("84_lesion_retention.py")
    df = pd.read_csv(BRSET_CSV)
    idc = df.columns[0]          # BOM on 'image_id', so take it positionally
    rng = np.random.RandomState(a.seed)

    rows, per_grade = [], {}
    for g in [int(x) for x in a.grades.split(",")]:
        ids = df.loc[df.DR_ICDR == g, idc].astype(str).values
        ids = rng.permutation(ids)[:a.n]
        pk, ar, nc = [], [], []
        for i, iid in enumerate(ids):
            p = Path(IMG_DIR) / f"{iid}.jpg"
            if not p.exists():
                continue
            arr = load_img(p, a.presize, a.size)
            cen, peak, area, _, _ = candidates(L, arr, a)
            nc.append(len(cen)); pk += list(np.asarray(peak).ravel()); ar += list(np.asarray(area).ravel())
            rows.append(dict(grade=g, image=iid, n_cand=len(cen)))
            if (i + 1) % 10 == 0:
                print(f"  grade {g}: {i+1}/{len(ids)}", flush=True)
        pk, ar, nc = np.asarray(pk), np.asarray(ar), np.asarray(nc)
        per_grade[g] = dict(
            n_images=int(len(nc)),
            cand_mean=float(nc.mean()), cand_sd=float(nc.std()),
            cand_p25=float(np.percentile(nc, 25)), cand_p50=float(np.percentile(nc, 50)),
            cand_p75=float(np.percentile(nc, 75)),
            peak_p50=float(np.median(pk)), peak_p25=float(np.percentile(pk, 25)),
            peak_p75=float(np.percentile(pk, 75)),
            area_p50=float(np.median(ar)), area_p25=float(np.percentile(ar, 25)),
            area_p75=float(np.percentile(ar, 75)))
        print(f"\ngrade {g}: n={len(nc)} images")
        print(f"  candidates/image  mean {nc.mean():6.1f}  median {np.median(nc):6.1f}"
              f"  IQR [{np.percentile(nc,25):.0f}, {np.percentile(nc,75):.0f}]")
        print(f"  peak contrast     median {np.median(pk):.4f}"
              f"  IQR [{np.percentile(pk,25):.4f}, {np.percentile(pk,75):.4f}]")
        print(f"  area (px)         median {np.median(ar):.1f}"
              f"  IQR [{np.percentile(ar,25):.0f}, {np.percentile(ar,75):.0f}]")

    print("\n" + "=" * 68)
    if 0 in per_grade and 1 in per_grade:
        d = per_grade[1]["cand_mean"] - per_grade[0]["cand_mean"]
        print(f"grade 1 minus grade 0 = {d:+.1f} candidates per image")
        print("That difference is the synthesis budget: how many microaneurysms to")
        print("add to a healthy retina so the detector sees a grade-1 image.")
        if d <= 0:
            print("\nWARNING: the detector does NOT separate grade 0 from grade 1 by")
            print("count. Count alone cannot be the synthesis target -- contrast and")
            print("area have to carry it, or the detector is too coarse for grade 1.")
    import json
    Path(a.out).write_text(json.dumps(
        dict(per_grade={str(k): v for k, v in per_grade.items()},
             detector=vars(a)), indent=2))
    pd.DataFrame(rows).to_csv(str(a.out).replace(".json", ".csv"), index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
