"""D107 registered read (CycleGAN baselines). Same metric code and bootstrap as scripts/154 and 156:
test DR AUROC, F1 at each run's stored validation-chosen cut-off, sens@spec90; 3-seed means; paired patient-cluster
bootstrap over the 193 test patients (2,000 resamples, rng seed 1).
Step 1 (validation only): pick the CycleGAN variant (plain vs residual) by mean validation DR AUROC, separately for the
25 % mix, the synthetic-only (swap 1.0) and the test-time arms.
Step 2 (single test read): primary S11 - CG25; CG25 - B0; Table-5 Holm family recomputed with CycleGAN added
(S11 minus FDA, Reinhard, class-conditional, physics only, diffusion only, no physics start, CycleGAN);
test-time: CGTT - CGTT-ident, S11 - CGTT, CGTT-ident - B0 (sanity, should be ~0); synthetic-only CG100 - C1.
--val_only prints step 1 and stops. Writes results/d107_table.json.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path("/home/users/sthummala2/brset-convnextv2"); CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
ARMS = {"S11": "syn_both_b0_s{s}_p0.25", "B0": "lebase_b0_s{s}", "C1": "syn_both_dc_b0_s{s}_p1.0",
        "CGP25": "syn_cyclegan_plain_b0_s{s}_p0.25", "CGR25": "syn_cyclegan_residual_b0_s{s}_p0.25",
        "CGP100": "syn_cyclegan_plain_b0_s{s}_p1.0", "CGR100": "syn_cyclegan_residual_b0_s{s}_p1.0",
        "CGTTP": "cgtt_plain_s{s}", "CGTTR": "cgtt_residual_s{s}", "CGTTI": "cgtt_ident_s{s}",
        "BFDA": "syn_fda_b0_s{s}_p0.25", "BRH": "syn_reinhard_b0_s{s}_p0.25", "BCC": "syn_classcond_b0_s{s}_p0.25",
        "AQ1": "syn_physonly_b0_s{s}_p0.25", "AQ2": "syn_bothnone_b0_s{s}_p0.25", "AQ3": "syn_bothsrc_b0_s{s}_p0.25"}
CHOICES = {"CG25": ("CGP25", "CGR25"), "CG100": ("CGP100", "CGR100"), "CGTT": ("CGTTP", "CGTTR")}
rd = lambda t: (CLS if t.startswith(("syn_", "pst_", "mbo_", "cgtt_")) else ROOT / "results") / t


def ld(t, sp):
    d = np.load(rd(t) / f"{sp}_predictions.npz"); return d["y_true"][:, 0].astype(bool), d["y_prob"][:, 0]


def f1(y, q):
    tp = (q & y).sum(); return 2 * tp / max(2 * tp + (q & ~y).sum() + (~q & y).sum(), 1)


def s90(y, p):
    fpr, tpr, _ = roc_curve(y, p); return float(np.interp(0.10, fpr, tpr))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--val_only", action="store_true"); a = ap.parse_args()
    val = {k: [roc_auc_score(*ld(ARMS[k].format(s=s), "val")) for s in (0, 1, 2)] for pair in CHOICES.values() for k in pair}
    chosen = {}
    for name, (p, r) in CHOICES.items():
        chosen[name] = p if np.mean(val[p]) >= np.mean(val[r]) else r
        print(f"VALIDATION {name}: plain {np.mean(val[p]):.4f} ({' / '.join(f'{v:.4f}' for v in val[p])}) vs residual "
              f"{np.mean(val[r]):.4f} ({' / '.join(f'{v:.4f}' for v in val[r])}) -> {chosen[name]}", flush=True)
    print(f"validation reference: S11 0.9550, C1 0.9397, B0 0.9377", flush=True)
    if a.val_only:
        return
    tl = pd.read_csv(ROOT / "data/finetune_joint/test/labels.csv"); pid = tl.file.astype(str).str.split(".").str[0].values
    pats = np.unique(pid); ix = {p: np.where(pid == p)[0] for p in pats}; yt = ld("lebase_b0_s0", "test")[0]
    rng = np.random.default_rng(1); BOOT = [np.concatenate([ix[p] for p in rng.choice(pats, len(pats), replace=True)]) for _ in range(2000)]
    A = {}
    for k, t in ARMS.items():
        T = [ld(t.format(s=s), "test") for s in (0, 1, 2)]
        assert all((tt[0] == yt).all() for tt in T), k
        thr = [float(np.load(rd(t.format(s=s)) / "test_predictions.npz")["thresholds"][0]) for s in (0, 1, 2)]
        P = [tt[1] for tt in T]
        A[k] = dict(P=P, Q=[p >= c for p, c in zip(P, thr)], thr=thr)
        A[k]["auc"] = [roc_auc_score(yt, p) for p in P]; A[k]["f1"] = [f1(yt, q) for q in A[k]["Q"]]
        A[k]["s90"] = [s90(yt, p) for p in P]
        A[k]["val"] = [roc_auc_score(*ld(t.format(s=s), "val")) for s in (0, 1, 2)]
        print(f"{k:7s} val {np.mean(A[k]['val']):.4f} | test AUROC {np.mean(A[k]['auc']):.4f} "
              f"({' / '.join(f'{v:.4f}' for v in A[k]['auc'])}) F1 {np.mean(A[k]['f1']):.4f} sens@spec90 {np.mean(A[k]['s90']):.3f}", flush=True)
    c25, c100, ctt = chosen["CG25"], chosen["CG100"], chosen["CGTT"]
    pairs = [("S11", c25, "primary"), (c25, "B0", "vs real BRSET")] + \
            [("S11", y, "holm7") for y in ("BFDA", "BRH", "BCC", "AQ1", "AQ2", "AQ3", c25)] + \
            [(ctt, "CGTTI", "test-time vs control"), ("S11", ctt, "ours vs test-time"), ("CGTTI", "B0", "sanity"),
             (c100, "C1", "synthetic-only")]
    out = {"val_choice": chosen, "val": {k: v for k, v in val.items()},
           "arms": {k: {"run": ARMS[k], "val_auc": float(np.mean(a_["val"])), "auc": float(np.mean(a_["auc"])),
                        "f1": float(np.mean(a_["f1"])), "s90": float(np.mean(a_["s90"])), "auc_seeds": a_["auc"],
                        "f1_seeds": a_["f1"], "cutoffs": a_["thr"]} for k, a_ in A.items()}, "pairs": []}
    for x, y, kind in pairs:
        dA = np.array([np.mean([roc_auc_score(yt[s], p[s]) for p in A[x]["P"]]) - np.mean([roc_auc_score(yt[s], p[s]) for p in A[y]["P"]]) for s in BOOT])
        dF = np.array([np.mean([f1(yt[s], q[s]) for q in A[x]["Q"]]) - np.mean([f1(yt[s], q[s]) for q in A[y]["Q"]]) for s in BOOT])
        out["pairs"].append(dict(x=x, y=y, kind=kind, dauc=float(np.mean(A[x]["auc"]) - np.mean(A[y]["auc"])),
                                 aucCI=[float(np.percentile(dA, 2.5)), float(np.percentile(dA, 97.5))],
                                 p_auc=float(min(1, 2 * min((dA <= 0).mean(), (dA >= 0).mean()))),
                                 df1=float(np.mean(A[x]["f1"]) - np.mean(A[y]["f1"])),
                                 f1CI=[float(np.percentile(dF, 2.5)), float(np.percentile(dF, 97.5))],
                                 p_f1=float(min(1, 2 * min((dF <= 0).mean(), (dF >= 0).mean()))),
                                 seeds_x_gt_y=int(sum(u > v for u, v in zip(A[x]["auc"], A[y]["auc"])))))
    hp = [p for p in out["pairs"] if p["kind"] == "holm7"]
    for key in ("p_auc", "p_f1"):
        order = sorted(range(len(hp)), key=lambda i: hp[i][key]); run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (len(hp) - rank) * hp[i][key])); hp[i][key + "_holm"] = run
    for p in out["pairs"]:
        print(f"{p['x']} - {p['y']} [{p['kind']}] AUROC {p['dauc']:+.4f} [{p['aucCI'][0]:+.4f}, {p['aucCI'][1]:+.4f}]"
              + (f" Holm p {p['p_auc_holm']:.3f}" if "p_auc_holm" in p else f" p {p['p_auc']:.3f}")
              + f" | F1 {p['df1']:+.4f} [{p['f1CI'][0]:+.4f}, {p['f1CI'][1]:+.4f}]"
              + (f" Holm p {p['p_f1_holm']:.3f}" if "p_f1_holm" in p else "") + f" | seeds {p['seeds_x_gt_y']}/3", flush=True)
    (ROOT / "results/d107_table.json").write_text(json.dumps(out, indent=1))
    print("wrote results/d107_table.json", flush=True)


if __name__ == "__main__":
    main()
