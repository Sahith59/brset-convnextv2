"""EXPERIMENT 3, step 3a: measure the appearance gap between BRSET and mBRSET.

Uses image appearance only. No diagnostic labels are read. This is what the
degradation parameters are later fitted to match.
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

_s = importlib.util.spec_from_file_location("ops", str(Path(__file__).parent / "55_degradation_ops.py"))
O = importlib.util.module_from_spec(_s); _s.loader.exec_module(O)

N = 400
SIZE = 512


def load(path, mode):
    """mode 'squash' reproduces the training pipeline, Resize((S,S)), which
    ignores aspect ratio. mode 'preserve' keeps geometry: resize the short side
    then centre crop. The two differ enormously for BRSET, whose images are all
    about 1.29 aspect, and not at all for mBRSET, whose images are all square."""
    im = Image.open(path).convert("RGB")
    if mode == "squash":
        return im.resize((SIZE, SIZE))
    w, h = im.size
    sc = SIZE / min(w, h)
    im = im.resize((max(SIZE, int(round(w * sc))), max(SIZE, int(round(h * sc)))))
    w, h = im.size
    l, t = (w - SIZE) // 2, (h - SIZE) // 2
    return im.crop((l, t, l + SIZE, t + SIZE))


def sample(split_dir, n, seed=0, mode="squash"):
    df = pd.read_csv(Path(split_dir) / "labels.csv")
    rng = np.random.default_rng(seed)
    files = df.file.values
    pick = rng.choice(files, size=min(n, len(files)), replace=False)
    rows = []
    for f in pick:
        try:
            im = load(Path(split_dir) / f, mode)
        except Exception:
            continue
        st = O.image_stats(im)
        if st: rows.append(st)
    return pd.DataFrame(rows)


def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "squash"
    b = sample("data/finetune_multilabel/train", N, mode=mode)
    m = sample("data/finetune_mbrset_multilabel/train", N, mode=mode)
    print(f"resize mode: {mode}   BRSET n={len(b)}   mBRSET n={len(m)}\n")
    print(f"{'statistic':12s} {'BRSET mean':>12} {'BRSET std':>11} {'mBRSET mean':>13} "
          f"{'mBRSET std':>11} {'gap':>10}")
    out = {}
    for c in ["sharpness", "brightness", "contrast", "falloff", "saturation"]:
        bm, bs = b[c].mean(), b[c].std()
        mm, ms = m[c].mean(), m[c].std()
        print(f"{c:12s} {bm:12.5f} {bs:11.5f} {mm:13.5f} {ms:11.5f} {mm-bm:+10.5f}")
        out[c] = {"brset_mean": float(bm), "brset_std": float(bs),
                  "mbrset_mean": float(mm), "mbrset_std": float(ms)}
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open(f"results/dataset_appearance_stats_{mode}.json", "w"), indent=2)
    b.to_csv(f"results/brset_appearance_{mode}.csv", index=False)
    m.to_csv(f"results/mbrset_appearance_{mode}.csv", index=False)
    print(f"\nwrote results/dataset_appearance_stats_{mode}.json")
    print("\nDIRECTION OF THE GAP (what degradation must reproduce):")
    for c in ["sharpness", "brightness", "contrast", "falloff", "saturation"]:
        d = out[c]["mbrset_mean"] - out[c]["brset_mean"]
        rel = d / max(abs(out[c]["brset_mean"]), 1e-9)
        word = "HIGHER" if d > 0 else "LOWER"
        print(f"  mBRSET is {word:6s} on {c:11s} by {abs(rel)*100:6.1f} percent")


if __name__ == "__main__":
    main()
