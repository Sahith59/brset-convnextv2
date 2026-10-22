"""D106: Translator-v2 twins in the zero-label 25 % mix (S11DC) vs the headline S11 (Translator-v1).
Step 1 (validation only): mean validation DR AUROC of S11DC vs S11 (0.9550) -> registered selection rule.
Step 2 (registered single test read, after all 3 runs): 3-seed test AUROC / F1 / sensitivity at 90 % specificity, paired
patient-cluster bootstrap (2,000, same resamples/metric code as scripts/154 and 119): S11DC - S11, S11DC - B0, and
S11DC - FDA / Reinhard / class-conditional with Holm across those three. Writes results/d106_table.json.
"""
import importlib.util, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sp = importlib.util.spec_from_file_location("T154", str(Path(__file__).parent / "154_d105_table.py"))
T = importlib.util.module_from_spec(sp); sp.loader.exec_module(T)
ARMS = {"S11DC": "syn_both_dc_b0_s{s}_p0.25", "S11": "syn_both_b0_s{s}_p0.25", "B0": "lebase_b0_s{s}",
        "BFDA": "syn_fda_b0_s{s}_p0.25", "BRH": "syn_reinhard_b0_s{s}_p0.25", "BCC": "syn_classcond_b0_s{s}_p0.25"}
PAIRS = [("S11DC", "S11", "primary"), ("S11DC", "B0", "vs real BRSET"), ("S11DC", "BFDA", "holm"), ("S11DC", "BRH", "holm"), ("S11DC", "BCC", "holm")]


def s90(y, p):
    fpr, tpr, _ = roc_curve(y, p); return float(np.interp(0.10, fpr, tpr))


def main():
    R = T.ROOT
    val = {k: [roc_auc_score(*T.ld(t.format(s=s), "val")) for s in (0, 1, 2)] for k, t in ARMS.items() if k in ("S11DC", "S11")}
    sel = float(np.mean(val["S11DC"])) > float(np.mean(val["S11"]))
    print(f"VALIDATION DR AUROC: S11DC {np.mean(val['S11DC']):.4f} ({' / '.join(f'{v:.4f}' for v in val['S11DC'])}) vs S11 "
          f"{np.mean(val['S11']):.4f} -> {'S11DC becomes the headline' if sel else 'S11 stays the headline'}", flush=True)
    tl = pd.read_csv(R / "data/finetune_joint/test/labels.csv"); pid = tl.file.astype(str).str.split(".").str[0].values
    pats = np.unique(pid); ix = {p: np.where(pid == p)[0] for p in pats}; yt = T.ld("lebase_b0_s0", "test")[0]
    rng = np.random.default_rng(1); BOOT = [np.concatenate([ix[p] for p in rng.choice(pats, len(pats), replace=True)]) for _ in range(2000)]
    A = {}
    for k, t in ARMS.items():
        P = [T.ld(t.format(s=s), "test")[1] for s in (0, 1, 2)]
        thr = [float(np.load(T.rd(t.format(s=s)) / "test_predictions.npz")["thresholds"][0]) for s in (0, 1, 2)]
        A[k] = dict(P=P, Q=[p >= c for p, c in zip(P, thr)])
        A[k]["auc"] = [roc_auc_score(yt, p) for p in P]; A[k]["f1"] = [T.f1(yt, q) for q in A[k]["Q"]]; A[k]["s90"] = [s90(yt, p) for p in P]
        print(f"{k:6s} test AUROC {np.mean(A[k]['auc']):.4f} ({' / '.join(f'{v:.4f}' for v in A[k]['auc'])}) F1 {np.mean(A[k]['f1']):.4f} "
              f"sens@spec90 {np.mean(A[k]['s90']):.3f}", flush=True)
    out = {"val": {k: v for k, v in val.items()}, "selected_S11DC": sel,
           "arms": {k: {"auc": float(np.mean(a["auc"])), "f1": float(np.mean(a["f1"])), "s90": float(np.mean(a["s90"])),
                        "auc_seeds": a["auc"], "f1_seeds": a["f1"]} for k, a in A.items()}, "pairs": []}
    for x, y, kind in PAIRS:
        dA = np.array([np.mean([roc_auc_score(yt[s], p[s]) for p in A[x]["P"]]) - np.mean([roc_auc_score(yt[s], p[s]) for p in A[y]["P"]]) for s in BOOT])
        dF = np.array([np.mean([T.f1(yt[s], q[s]) for q in A[x]["Q"]]) - np.mean([T.f1(yt[s], q[s]) for q in A[y]["Q"]]) for s in BOOT])
        out["pairs"].append(dict(x=x, y=y, kind=kind, dauc=float(np.mean(A[x]["auc"]) - np.mean(A[y]["auc"])),
                                 aucCI=[float(np.percentile(dA, 2.5)), float(np.percentile(dA, 97.5))],
                                 p_auc=float(min(1, 2 * min((dA <= 0).mean(), (dA >= 0).mean()))),
                                 df1=float(np.mean(A[x]["f1"]) - np.mean(A[y]["f1"])),
                                 f1CI=[float(np.percentile(dF, 2.5)), float(np.percentile(dF, 97.5))],
                                 seeds_x_gt_y=int(sum(u > v for u, v in zip(A[x]["auc"], A[y]["auc"])))))
    hp = [p for p in out["pairs"] if p["kind"] == "holm"]
    order = sorted(range(len(hp)), key=lambda i: hp[i]["p_auc"]); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(hp) - rank) * hp[i]["p_auc"])); hp[i]["p_auc_holm"] = run
    for p in out["pairs"]:
        print(f"{p['x']} - {p['y']} [{p['kind']}] AUROC {p['dauc']:+.4f} [{p['aucCI'][0]:+.4f}, {p['aucCI'][1]:+.4f}]"
              + (f" Holm p {p['p_auc_holm']:.3f}" if 'p_auc_holm' in p else f" p {p['p_auc']:.3f}")
              + f" | F1 {p['df1']:+.4f} [{p['f1CI'][0]:+.4f}, {p['f1CI'][1]:+.4f}] | seeds {p['seeds_x_gt_y']}/3", flush=True)
    (R / "results/d106_table.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
