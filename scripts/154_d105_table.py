"""D105 registered test read for experiment A (and later B). Same metric code as scripts/119_full_table.py:
test DR AUROC and F1 at each run's own validation-chosen cut-off (stored in test_predictions.npz['thresholds']),
3-seed means, paired patient-cluster bootstrap over the 193 test patients (2,000 resamples, rng seed 1),
plus validation DR AUROC and grade-1-vs-healthy AUROC (validation only).
CIs are patient-only (seed variance not included); Holm adjustment across the 5 'C1 beats X' contrasts (D105a).
Usage: --exp A | B   (B adds X-mild, X-nat, C1-prev).  Writes results/d105_<exp>_table.json.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/users/sthummala2/brset-convnextv2"); CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
ARMS = {"MBO": "mbo_lebase_s{s}", "C1": "syn_both_dc_b0_s{s}_p1.0", "SYN1": "syn_both_b0_s{s}_p1.0", "B0": "lebase_b0_s{s}",
        "FDA1": "syn_fda1_b0_s{s}_p1.0", "RH1": "syn_rh1_b0_s{s}_p1.0", "PH1": "syn_ph1_b0_s{s}_p1.0",
        "CC1": "syn_cc1_b0_s{s}_p1.0", "SRC1": "syn_src1_b0_s{s}_p1.0", "REV1": "syn_rev1_b0_s{s}_p1.0",
        "C1PREV": "syn_c1prev_s{s}", "XMILD": "syn_xmild2_s{s}", "XNAT": "syn_xnat2_s{s}"}
PAIRS = {"A": [("C1", "FDA1", "holm"), ("C1", "RH1", "holm"), ("C1", "PH1", "holm"), ("C1", "CC1", "holm"), ("C1", "SRC1", "holm"),
               ("REV1", "SRC1", "order"), ("SYN1", "REV1", "order"), ("SYN1", "SRC1", "order"),
               ("C1", "MBO", "target"), ("FDA1", "MBO", "target"), ("REV1", "MBO", "target"), ("C1", "B0", "context")],
         "B": [("XMILD", "C1PREV", "primary"), ("XMILD", "C1", "headline"), ("XMILD", "XNAT", "mechanism"),
               ("XNAT", "C1PREV", "control"), ("XMILD", "MBO", "target"), ("XNAT", "MBO", "target")]}
rd = lambda t: (CLS if t.startswith(("syn_", "pst_", "mbo_")) else ROOT / "results") / t


def ld(t, sp):
    d = np.load(rd(t) / f"{sp}_predictions.npz"); return d["y_true"][:, 0].astype(bool), d["y_prob"][:, 0]


def f1(y, q):
    tp = (q & y).sum(); return 2 * tp / max(2 * tp + (q & ~y).sum() + (~q & y).sum(), 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True, choices=["A", "B"]); a = ap.parse_args()
    raw = pd.read_csv("/home/users/sthummala2/mbrset-retfound/data/raw/labels_mbrset.csv").set_index("file")
    vl = pd.read_csv(ROOT / "data/finetune_joint/val/labels.csv"); gv = raw.loc[vl.file, "final_icdr"].astype(int).values
    tl = pd.read_csv(ROOT / "data/finetune_joint/test/labels.csv"); pid = tl.file.astype(str).str.split(".").str[0].values
    pats = np.unique(pid); ix = {p: np.where(pid == p)[0] for p in pats}; yt = ld("lebase_b0_s0", "test")[0]
    rng = np.random.default_rng(1); BOOT = [np.concatenate([ix[p] for p in rng.choice(pats, len(pats), replace=True)]) for _ in range(2000)]
    need = sorted({x for p in PAIRS[a.exp] for x in p[:2]})
    R = {}
    for k in need:
        t = ARMS[k]
        V = [ld(t.format(s=s), "val") for s in (0, 1, 2)]; T = [ld(t.format(s=s), "test") for s in (0, 1, 2)]
        assert all((tt[0] == yt).all() for tt in T)
        thr = [float(np.load(rd(t.format(s=s)) / "test_predictions.npz")["thresholds"][0]) for s in (0, 1, 2)]
        m = gv <= 1
        R[k] = dict(val_auc=[roc_auc_score(v[0], v[1]) for v in V], val_g1=[roc_auc_score(gv[m] == 1, v[1][m]) for v in V],
                    auc=[roc_auc_score(yt, tt[1]) for tt in T], f1=[f1(yt, tt[1] >= c) for tt, c in zip(T, thr)], thr=thr,
                    P=[tt[1] for tt in T], Q=[tt[1] >= c for tt, c in zip(T, thr)])
    out = {"arms": {}, "pairs": []}
    for k, r in R.items():
        out["arms"][k] = {"run": ARMS[k], "val_auc": float(np.mean(r["val_auc"])), "val_g1_vs_healthy": float(np.mean(r["val_g1"])),
                          "test_auc": float(np.mean(r["auc"])), "test_f1": float(np.mean(r["f1"])),
                          "test_auc_seeds": [float(v) for v in r["auc"]], "test_f1_seeds": [float(v) for v in r["f1"]],
                          "cutoffs": r["thr"]}
        print(f"{k:7s} val {np.mean(r['val_auc']):.4f} g1 {np.mean(r['val_g1']):.3f} | test AUROC {np.mean(r['auc']):.4f} "
              f"({' / '.join(f'{v:.4f}' for v in r['auc'])}) F1 {np.mean(r['f1']):.4f} ({' / '.join(f'{v:.4f}' for v in r['f1'])})", flush=True)
    for x, y, kind in PAIRS[a.exp]:
        dA = np.array([np.mean([roc_auc_score(yt[s], p[s]) for p in R[x]["P"]]) - np.mean([roc_auc_score(yt[s], p[s]) for p in R[y]["P"]]) for s in BOOT])
        dF = np.array([np.mean([f1(yt[s], q[s]) for q in R[x]["Q"]]) - np.mean([f1(yt[s], q[s]) for q in R[y]["Q"]]) for s in BOOT])
        pA = float(min(1.0, 2 * min((dA <= 0).mean(), (dA >= 0).mean()))); pF = float(min(1.0, 2 * min((dF <= 0).mean(), (dF >= 0).mean())))
        out["pairs"].append(dict(x=x, y=y, kind=kind, dauc=float(np.mean(R[x]["auc"]) - np.mean(R[y]["auc"])),
                                 aucCI=[float(np.percentile(dA, 2.5)), float(np.percentile(dA, 97.5))], p_auc=pA,
                                 df1=float(np.mean(R[x]["f1"]) - np.mean(R[y]["f1"])),
                                 f1CI=[float(np.percentile(dF, 2.5)), float(np.percentile(dF, 97.5))], p_f1=pF,
                                 seeds_x_gt_y=int(sum(u > v for u, v in zip(R[x]["auc"], R[y]["auc"])))))
    hp = [p for p in out["pairs"] if p["kind"] == "holm"]
    for key in ("p_auc", "p_f1"):                       # Holm step-down across the 'C1 beats X' family
        order = sorted(range(len(hp)), key=lambda i: hp[i][key]); run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (len(hp) - rank) * hp[i][key])); hp[i][key + "_holm"] = run
    for p in out["pairs"]:
        print(f"{p['x']:6s} - {p['y']:6s} [{p['kind']}] AUROC {p['dauc']:+.4f} [{p['aucCI'][0]:+.4f}, {p['aucCI'][1]:+.4f}] p {p['p_auc']:.3f}"
              + (f" holm {p['p_auc_holm']:.3f}" if 'p_auc_holm' in p else "")
              + f" | F1 {p['df1']:+.4f} [{p['f1CI'][0]:+.4f}, {p['f1CI'][1]:+.4f}] p {p['p_f1']:.3f}"
              + (f" holm {p['p_f1_holm']:.3f}" if 'p_f1_holm' in p else "") + f" | seeds {p['seeds_x_gt_y']}/3", flush=True)
    (ROOT / f"results/d105_{a.exp}_table.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
