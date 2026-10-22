"""D105 experiment B (X-mild / X-nat): batch-matched selection, preprocessing and CPU kill tests for the external
DR photos (DDR + Messidor-2) that are translated into phone style and added to the C1-prev synthetic set.

Selection (D105 amendment, validity red team): sampling is done INSIDE each acquisition batch at C1-prev's DR
prevalence P = 4748/15372, so no batch predicts the label:
  DDR      : only the '007-*' filename family of the 512x512 bucket (the timestamp batches are healthy-only)
  Messidor2: per family = IM* (Brest extension) and each Messidor-1 acquisition month (filename prefix YYYYMM)
Per batch f: X-mild positives = grade 1 first, then grade 2, k_f = min(mild available, round(n0_f * P / (1 - P)));
negatives = min(n0_f, round(k_f * (1 - P) / P)); X-nat = the SAME negatives + k_f positives drawn from the batch's
natural grade 1-4 mix. Near-duplicates (64-bit DCT pHash, Hamming <= 2, PAIRWISE, no transitive grouping) within and across sources:
a pair with conflicting DR status drops both, otherwise one member is kept.
Preprocessing 'matfix': anti-aliased resize of the field of view to the BRSET-cache ellipse + per-channel FOV
mean/std matched to BRSET (prep2 'mataa', agents_wf2/external/prep2.py), then ONE Gaussian blur per SOURCE (median of
the per-image Laplacian-variance-matching sigma over 200 grade-0 photos of that source), not per image, so the blur
level cannot carry the label (red-team finding on per-image 'matlv').
Outputs (off the public repo): brset_gen/cache/ext2_brset_train_560.npy + ext2_brset_train_meta.csv (read by
scripts/111_translate.py with --cache_tag ext2_), brset_gen/ext2/{xmild,xnat}.csv; aggregate counts only to results/.

Modes: manifest | prep --shard k --n_shards n | merge | kb_cpu
"""
import argparse, importlib.util, json, re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).parent.parent
G = Path("/data/users3/sthummala2/brset_gen")
E = G / "ext2"
DS = Path("/data/users4/nshaik3/Datasets")
P = 4748 / 15372
SV = json.loads((G / "agents_wf2/external/survey.json").read_text())
ELL = dict(cx=0.5098, cy=0.4924, a=0.4848, b=0.6273)                     # BRSET-cache FOV ellipse (prep2.py)
BMU, BSD = np.array(SV["brset_cache"]["rgb_mean"]), np.array(SV["brset_cache"]["rgb_std"])


def _load(name, mod):
    sp = importlib.util.spec_from_file_location(mod, str(ROOT / "scripts" / name))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def candidates():
    ddr = pd.read_csv(DS / "DDR/DR_grading.csv")
    ddr = ddr[ddr.diagnosis.isin([0, 1, 2, 3, 4])].copy()
    ddr["path"] = [str(DS / "DDR/DR_grading/DR_grading" / i) for i in ddr.id_code]
    ddr = ddr[[bool(re.match(r"^\d{3}-", i)) for i in ddr.id_code]]
    ddr["source"], ddr["family"] = "ddr", "007"
    ms = pd.read_csv(DS / "Messidor2/messidor_data.csv")
    ms = ms[ms.diagnosis.isin([0, 1, 2, 3, 4]) & (ms.adjudicated_gradable == 1)].copy()
    ms["path"] = [str(DS / "Messidor2/messidor-2/messidor-2/preprocess" / i) for i in ms.id_code]
    ms["source"] = "messidor2"
    ms["family"] = ["IM" if i.upper().startswith("IM") else "M1_" + i[:6] for i in ms.id_code]
    c = pd.concat([ddr, ms], ignore_index=True)
    c["grade"] = c.diagnosis.astype(int); c["dr"] = (c.grade >= 1).astype(int)
    c["name"] = [f"{s}__{Path(i).stem}" for s, i in zip(c.source, c.id_code)]
    return c[["name", "path", "source", "family", "grade", "dr"]]


def size_ok(p):
    with Image.open(p) as im:
        return im.size == (512, 512)


def phash(p):
    g = cv2.resize(np.asarray(Image.open(p).convert("L"), dtype=np.float32), (32, 32), interpolation=cv2.INTER_AREA)
    d = cv2.dct(g)[:8, :8].flatten()
    return np.packbits(d > np.median(d[1:]))


# ---- preprocessing (geom / mataa copied from agents_wf2/external/prep2.py; blur fixed per source) ----
def geom(a):
    m = a.mean(2) > 10; rows = np.where(m.any(1))[0]; cols = np.where(m.any(0))[0]; H, W = m.shape
    widths = m.sum(1).astype(float); cx = (cols[0] + cols[-1]) / 2; amax = widths.max() / 2
    ct, cb = rows[0] <= 1, rows[-1] >= H - 2
    if not (ct or cb):
        cy = (rows[0] + rows[-1]) / 2; b = (rows[-1] - rows[0] + 1) / 2
    else:
        cy = float(np.argmax(widths)) if not (ct and cb) else float(np.mean(np.where(widths >= widths.max() - 2)[0]))
        edge = rows[0] if ct else rows[-1]; r = min(widths[edge] / 2 / amax, 0.999)
        b = abs(cy - edge) / np.sqrt(1 - r * r) if r < 0.999 else max(abs(cy - edge), 1)
    return cx, cy, amax, b


def mataa(a):
    cx, cy, ah, bh = geom(a); H, W = a.shape[:2]
    sx = ELL["a"] * 560 / ah; sy = ELL["b"] * 560 / bh
    nw, nh = max(8, int(round(W * sx))), max(8, int(round(H * sy)))
    r = np.asarray(Image.fromarray(a).resize((nw, nh), Image.BILINEAR))
    tx = int(round(ELL["cx"] * 560 - cx * nw / W)); ty = int(round(ELL["cy"] * 560 - cy * nh / H))
    y = np.zeros((560, 560, 3), np.uint8)
    x0, y0 = max(0, tx), max(0, ty); x1, y1 = min(560, tx + nw), min(560, ty + nh)
    y[y0:y1, x0:x1] = r[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
    m = y.mean(2) > 10; yf = y.astype(np.float32)
    for c in range(3):
        v = yf[..., c][m]; yf[..., c][m] = (v - v.mean()) / (v.std() + 1e-6) * BSD[c] + BMU[c]
    yf[~m] = 0
    return np.clip(yf, 0, 255).round().astype(np.uint8)


def blur(a, s):
    if s <= 0:
        return a
    m = a.mean(2) > 10; b = cv2.GaussianBlur(a, (0, 0), s); b[~m] = 0
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["manifest", "prep", "merge", "kb_cpu"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    a = ap.parse_args()
    E.mkdir(exist_ok=True)
    if a.mode == "manifest":
        c = candidates()
        c = c[[size_ok(p) for p in c.path]].reset_index(drop=True)
        H = np.stack([phash(p) for p in c.path])                              # (N, 8) uint8
        bits = np.unpackbits(H, axis=1).astype(np.uint8)
        N = len(c); hist = np.zeros(65, int); pairs = []
        for s in range(0, N, 512):
            d = (bits[s:s + 512, None, :] != bits[None, :, :]).sum(2)        # Hamming distances
            ii, jj = np.where(d <= 8)
            for i, j in zip(ii + s, jj):
                if i < j:
                    hist[d[i - s, j]] += 1
                    if d[i - s, j] <= 2: pairs.append((i, j))
        # PAIRWISE near-duplicate removal (no transitive grouping: fundus photos chain together at loose thresholds,
        # first attempt dropped 2,858 photos in 104 giant 'groups'). Pair with conflicting DR -> drop both; else drop j.
        drop = set(); conflict = 0
        for i, j in pairs:
            if c.dr[i] != c.dr[j]:
                drop |= {i, j}; conflict += 1
            else:
                drop.add(j)
        keep = c.drop(index=sorted(drop)).reset_index(drop=True)
        dedup = dict(candidates=N, pair_hist_hamming_0_to_8={k: int(hist[k]) for k in range(9)}, dup_pairs_le2=len(pairs),
                     conflicting_pairs=conflict, dropped=len(drop), kept=len(keep))
        print(json.dumps(dedup), flush=True)
        rng = np.random.RandomState(0); xm, xn = [], []
        for f, d in keep.groupby(["source", "family"]):
            n0 = d[d.grade == 0]; mild = pd.concat([d[d.grade == 1].sample(frac=1, random_state=rng),
                                                     d[d.grade == 2].sample(frac=1, random_state=rng)])
            k = int(min(len(mild), round(len(n0) * P / (1 - P))))
            nneg = int(min(len(n0), round(k * (1 - P) / P)))
            neg = n0.sample(nneg, random_state=rng); pos_m = mild.iloc[:k]
            pos_n = d[d.grade >= 1].sample(k, random_state=rng)
            xm.append(pd.concat([neg, pos_m])); xn.append(pd.concat([neg, pos_n]))
        xm, xn = pd.concat(xm, ignore_index=True), pd.concat(xn, ignore_index=True)
        xm.to_csv(E / "xmild.csv", index=False); xn.to_csv(E / "xnat.csv", index=False)
        union = pd.concat([xm, xn]).drop_duplicates("name").reset_index(drop=True)
        # per-source fixed blur: median per-image sigma that matches BRSET green Laplacian variance, 200 grade-0 photos
        M = _load("76_measure_chromatic.py", "M76")
        GLV = json.loads((ROOT / "results/chromatic_stats_sq512.json").read_text())["g_lapvar"]["brset_mean"]
        glv = lambda x: M.stats(Image.fromarray(x).crop((24, 24, 536, 536)))["g_lapvar"]
        sig = {}
        for s, d in keep[keep.grade == 0].groupby("source"):
            ss = []
            for p in d.sample(min(200, len(d)), random_state=0).path:
                y = mataa(np.asarray(Image.open(p).convert("RGB")))
                best = (abs(np.log(glv(y) / GLV)), 0.0)
                for t in (0.3, 0.45, 0.6, 0.75, 0.9, 1.1, 1.3):
                    v = abs(np.log(glv(blur(y, t)) / GLV))
                    if v < best[0]: best = (v, t)
                ss.append(best[1])
            sig[s] = float(np.median(ss))
        union["sigma"] = union.source.map(sig)
        union.to_csv(E / "union.csv", index=False)
        summ = dict(dedup=dedup, sigma_by_source=sig, n_union=len(union), P=P)
        for nm, x in (("xmild", xm), ("xnat", xn)):
            summ[nm] = dict(rows=len(x), dr_pos=int(x.dr.sum()),
                            grades=x.groupby(["source", "grade"]).size().unstack(fill_value=0).to_dict("index"),
                            prevalence_by_family=x.groupby(["source", "family"]).dr.mean().round(3).to_dict())
        summ["xmild"]["prevalence_by_family"] = {f"{k[0]}/{k[1]}": v for k, v in summ["xmild"]["prevalence_by_family"].items()}
        summ["xnat"]["prevalence_by_family"] = {f"{k[0]}/{k[1]}": v for k, v in summ["xnat"]["prevalence_by_family"].items()}
        (ROOT / "results/ext2_manifest_summary.json").write_text(json.dumps(summ, indent=1, default=str))
        print(json.dumps(summ, indent=1, default=str), flush=True)
        return
    if a.mode == "prep":
        u = pd.read_csv(E / "union.csv")
        ids = list(range(len(u)))[a.shard::a.n_shards]
        out = np.zeros((len(ids), 560, 560, 3), np.uint8)
        for j, i in enumerate(ids):
            out[j] = blur(mataa(np.asarray(Image.open(u.path[i]).convert("RGB"))), u.sigma[i])
        np.save(E / f"prep_shard{a.shard:02d}.npy", out)
        print(f"shard {a.shard}: {len(ids)} images", flush=True)
        return
    if a.mode == "merge":
        u = pd.read_csv(E / "union.csv")
        arr = np.lib.format.open_memmap(G / "cache/ext2_brset_train_560.npy", mode="w+", dtype=np.uint8, shape=(len(u), 560, 560, 3))
        shards = sorted(E.glob("prep_shard*.npy")); n = len(shards)
        for k, f in enumerate(shards):
            arr[list(range(len(u)))[k::n]] = np.load(f)
        arr.flush()
        meta = u.assign(idx=np.arange(len(u)), file=[f"{s}.png" for s in u.name])
        meta[["idx", "file", "name", "path", "source", "family", "grade", "dr", "sigma"]].to_csv(
            G / "cache/ext2_brset_train_meta.csv", index=False)
        print(f"merged {n} shards -> {len(u)} images", flush=True)
        return
    # ---- kb_cpu: batch shortcut on the FINAL manifests + pre-normalisation-statistics DR probe
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    M = _load("76_measure_chromatic.py", "M76")
    res = {}
    for nm in ("xmild", "xnat"):
        x = pd.read_csv(E / f"{nm}.csv")
        for s, d in x.groupby("source"):
            if d.family.nunique() < 2:
                res[f"{nm}_{s}_batch_auroc"] = 0.5; continue
            fam_rate = d.groupby("family").dr.transform("mean")     # how well 'which batch' predicts DR
            res[f"{nm}_{s}_batch_auroc"] = float(roc_auc_score(d.dr, fam_rate))
    keys = None
    def feats(paths):
        nonlocal keys
        rows = []
        for p in paths:
            s = M.stats(Image.open(p).convert("RGB").resize((560, 560), Image.BILINEAR).crop((24, 24, 536, 536)))
            if s is None:
                rows.append(None); continue
            keys = keys or sorted(k for k, v in s.items() if isinstance(v, float))
            rows.append([s[k] for k in keys])
        return rows
    def probe_rows(rows, y):
        ok = [r is not None for r in rows]
        return probe(np.array([r for r in rows if r is not None]), np.asarray(y)[ok])
    probe = lambda X, y: float(roc_auc_score(y, cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                                                   X, y, cv=5, method="predict_proba")[:, 1]))
    x = pd.read_csv(E / "xmild.csv")
    for s, d in x.groupby("source"):
        res[f"rawstats_dr_probe_{s}"] = probe_rows(feats(d.path), d.dr.values)
    bm = pd.read_csv(G / "cache/brset_train_meta.csv")
    bm = pd.concat([bm[bm.dr == 1].sample(400, random_state=0), bm[bm.dr == 0].sample(900, random_state=0)])
    A = np.load(G / "cache/brset_train_560.npy", mmap_mode="r")
    rows = []
    for i in bm.index:
        s = M.stats(Image.fromarray(np.array(A[i])).crop((24, 24, 536, 536)))
        rows.append(None if s is None else [s[k] for k in keys])
    res["rawstats_dr_probe_BRSET_reference"] = probe_rows(rows, bm.dr.values)
    lim = max(0.60, res["rawstats_dr_probe_BRSET_reference"] + 0.05)
    ok = all(v <= 0.55 for k, v in res.items() if k.endswith("batch_auroc")) and \
        all(res[f"rawstats_dr_probe_{s}"] <= lim for s in x.source.unique())
    res.update(rawstats_limit=lim, pass_=ok)
    (ROOT / "results/ext2_kb_cpu.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()
