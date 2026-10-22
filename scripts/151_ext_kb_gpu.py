"""D105a experiment B, GPU kill tests on the TRANSLATED external images (synth/ext2_dc_matfix_s0.4_w1.5).

KB2  lesions survive translation, WITHIN batch: frozen C1 seed-0 classifier (cls_runs/syn_both_dc_b0_s0_p1.0, diagnostic
     only, never selected on), grade-1-vs-grade-0 AUROC >= 0.70 inside DDR '007' and inside Messidor-1 (all M1 months).
KB3  healthy eyes not read as disease: per source, C1-s0 mean P(DR) on translated healthy images <= max(0.20, 2 x ref),
     ref = C1-s0 mean P(DR) on 1,000 HELD-OUT BRSET grade-0 translations (synth/heldout_gen_both_e40k_phys_s0.4_w1.5).
KB1a acquisition batch visible in pixels: 5-fold logistic probe on DINOv2 ViT-L/14 embeddings (as scripts/132) of translated
     grade-0 images predicting the batch (DDR-007 / Messidor IM / Messidor-1), macro one-vs-rest AUROC -- REPORTED only.
Exit 0 if KB2 and KB3 pass, 3 otherwise. Reads no mBRSET label and no test file.
"""
import importlib.util, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
G = Path("/data/users3/sthummala2/brset_gen")
TW = G / "synth/ext2_dc_matfix_s0.4_w1.5"


def _load(name, mod):
    sp = importlib.util.spec_from_file_location(mod, str(ROOT / "scripts" / name))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def main():
    PR = _load("120_predict.py", "P120")
    u = pd.read_csv(G / "cache/ext2_brset_train_meta.csv")
    u["twin"] = [str(TW / f) for f in u.file]
    assert all(Path(p).exists() for p in u.twin), "missing translated images"
    run = G / "cls_runs/syn_both_dc_b0_s0_p1.0"
    u["p"] = PR.score(run, u.twin.tolist())[:, 0]
    hm = pd.read_csv(G / "cache/brset_heldout_meta.csv")
    h0 = hm[hm.dr == 0].sample(1000, random_state=0)
    ref = float(PR.score(run, [str(G / "synth/heldout_gen_both_e40k_phys_s0.4_w1.5" / (Path(f).stem + ".png")) for f in h0.file])[:, 0].mean())
    res = {"KB3_ref_heldout_grade0_meanP": ref}
    ok = True
    for nm, d in (("ddr_007", u[u.source == "ddr"]), ("messidor1", u[u.family.str.startswith("M1_")])):
        g = d[d.grade.isin([0, 1])]
        auc = float(roc_auc_score(g.grade == 1, g.p))
        res[f"KB2_{nm}_g1v0_auroc"] = auc; ok &= auc >= 0.70
    lim = max(0.20, 2 * ref)
    for s, d in u.groupby("source"):
        mp = float(d[d.grade == 0].p.mean())
        res[f"KB3_{s}_healthy_meanP"] = mp; ok &= mp <= lim
    res["KB3_limit"] = lim
    res["diag_meanP_by_source_grade"] = {f"{s}/{g}": float(v) for (s, g), v in u.groupby(["source", "grade"]).p.mean().items()}
    # KB1a (reported): batch visible in DINOv2 space among translated grade-0 images
    CC = _load("132_copy_check.py", "C132")
    dino = CC.Dino().cuda()
    z = u[u.grade == 0].copy()
    z["batch"] = np.where(z.source == "ddr", "ddr007", np.where(z.family == "IM", "messidor_IM", "messidor_M1"))
    emb = torch.cat([dino(x.cuda()).cpu() for x in CC.from_files(z.twin.tolist())]).numpy()
    pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), emb, z.batch.values, cv=5,
                           method="predict_proba")
    cls = sorted(z.batch.unique())
    res["KB1a_batch_probe_macro_auroc"] = float(np.mean([roc_auc_score(z.batch.values == c, pr[:, k]) for k, c in enumerate(cls)]))
    res["pass"] = bool(ok)
    (ROOT / "results/ext2_kb_gpu.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()
