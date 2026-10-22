"""Cache the generator's training images as uint8 arrays at the classifier's own
geometry (Resize((560,560)), bilinear, exactly what 30_train_strong_baseline.py
does before RandomCrop(512)). The generator then works in the pixel grid the
classifier consumes, so a synthetic image needs no further resampling.

ONLY the TRAIN split is cached. The generator never sees mBRSET val or test, and
it never sees an mBRSET label: the mBRSET meta file holds file names only.
"""
import argparse, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
GEN = Path("/data/users3/sthummala2/brset_gen")
BRSET_CSV = ("/data/users4/nshaik3/Datasets/BRSET/physionet.org/files/"
             "brazilian-ophthalmological/1.0.1/labels_brset.csv")


def load_one(job):
    path, S = job
    last = None
    for k in range(5):
        try:
            im = Image.open(path).convert("RGB").resize((S, S), Image.BILINEAR)
            return np.asarray(im, np.uint8)
        except (OSError, IOError) as e:      # NFS reads fail transiently here
            last = e; time.sleep(0.5 * (k + 1))
    raise RuntimeError(f"cannot read {path}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=560)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--heldout", action="store_true",
                    help="cache BRSET val+test instead. The gate-2 judge was trained "
                         "on BRSET TRAIN, so label preservation has to be measured on "
                         "BRSET images it has never seen.")
    a = ap.parse_args()
    if a.heldout:
        b = pd.read_csv(BRSET_CSV); idc = b.columns[0]
        grade = dict(zip(b[idc].astype(str), b["DR_ICDR"]))
        parts = []
        for sp in ("val", "test"):
            d = pd.read_csv(ROOT / f"data/finetune_multilabel/{sp}/labels.csv"); d["split"] = sp
            parts.append(d)
        d = pd.concat(parts).reset_index(drop=True)
        out = GEN / "cache" / f"brset_heldout_{a.size}.npy"
        mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8,
                                       shape=(len(d), a.size, a.size, 3))
        jobs = [(str(ROOT / f"data/finetune_multilabel/{sp}" / f), a.size)
                for sp, f in zip(d.split, d.file)]
        with Pool(a.workers) as p:
            for i, arr in enumerate(p.imap(load_one, jobs, chunksize=8)):
                mm[i] = arr
        mm.flush(); del mm
        pd.DataFrame({"idx": np.arange(len(d)), "file": d.file, "split": d.split,
                      "dr": d.diabetic_retinopathy.values, "edema": d.macular_edema.values,
                      "grade": d.file.astype(str).str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)
                                .map(grade).values}
                     ).to_csv(GEN / "cache" / "brset_heldout_meta.csv", index=False)
        print(f"wrote {out}  shape ({len(d)},{a.size},{a.size},3)", flush=True)
        return
    (GEN / "cache").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/finetune_joint/train/labels.csv")
    b = pd.read_csv(BRSET_CSV); idc = b.columns[0]       # BOM on image_id
    grade = dict(zip(b[idc].astype(str), b["DR_ICDR"]))
    for dom, name in ((1, "mbrset"), (0, "brset")):
        d = df[df.domain == dom].reset_index(drop=True)
        if a.limit:
            # keep some positives in a smoke cache so the panels have lesions
            d = pd.concat([d[d.diabetic_retinopathy == 1].head(a.limit // 2),
                           d[d.diabetic_retinopathy == 0].head(a.limit // 2)]).reset_index(drop=True)
        out = GEN / "cache" / f"{a.tag}{name}_train_{a.size}.npy"
        mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8,
                                       shape=(len(d), a.size, a.size, 3))
        jobs = [(str(ROOT / "data/finetune_joint/train" / f), a.size) for f in d.file]
        t0 = time.time()
        with Pool(a.workers) as p:
            for i, arr in enumerate(p.imap(load_one, jobs, chunksize=8)):
                mm[i] = arr
                if (i + 1) % 1000 == 0:
                    print(f"  {name} {i+1}/{len(d)}  {time.time()-t0:.0f}s", flush=True)
        mm.flush(); del mm
        meta = pd.DataFrame({"idx": np.arange(len(d)), "file": d.file})
        if dom == 0:                                    # source labels are ours to use
            meta["dr"] = d.diabetic_retinopathy.values
            meta["edema"] = d.macular_edema.values
            meta["grade"] = (d.file.astype(str).str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)
                             .map(grade).values)
        meta.to_csv(GEN / "cache" / f"{a.tag}{name}_train_meta.csv", index=False)
        print(f"wrote {out}  shape ({len(d)},{a.size},{a.size},3)  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
