"""Reviewer-requested statistics (CPU only, saved predictions only, no training; 20 Oct 2026).
(1) Two-level bootstrap for every Table-1 contrast (ours S11 minus X): each of 2,000 iterations resamples the 193 test
    patients (shared by both arms, same resamples as scripts/154/156/159: rng seed 1) AND, independently for each arm,
    draws 3 of its 3 seeds with replacement (rng seed 2), so the interval includes training-seed variation as well as
    patient variation. Patient-only intervals are recomputed alongside as a consistency check (must match table5_holm /
    d106 / pair_stf_minus_s11). Holm across the Table-1 family (six contrasts; seven when the D107 CycleGAN arm exists).
(2) Label budget: S11 minus the 250-, 500- and 1,000-label models, patient-only and two-level.
(3) The 12 zero-label candidate configurations that validation chose among: validation and test AUROC/F1 of each.
Writes results/extra_stats_2026-10-20.json (or ..._withcg.json when the CycleGAN arm is included).
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/users/sthummala2/brset-convnextv2"); CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
rd = lambda t: (CLS if t.startswith(("syn_", "pst_", "mbo_", "cgtt_")) else ROOT / "results") / t
ARMS = {"S11": "syn_both_b0_s{s}_p0.25", "B0": "lebase_b0_s{s}", "BFDA": "syn_fda_b0_s{s}_p0.25",
        "BRH": "syn_reinhard_b0_s{s}_p0.25", "BCC": "syn_classcond_b0_s{s}_p0.25", "AQ1": "syn_physonly_b0_s{s}_p0.25",
        "AQ2": "syn_bothnone_b0_s{s}_p0.25", "AQ3": "syn_bothsrc_b0_s{s}_p0.25", "S11DC": "syn_both_dc_b0_s{s}_p0.25",
        "STF": "pst_F_base1loose_notwins_s{s}", "B250": "lebase_b250_s{s}", "B500": "lebase_b500_s{s}", "B1000": "lebase_b1000_s{s}"}
import os
if os.environ.get("D113_FDA_RUN"):   # D113: FOV-restricted FDA replaces whole-frame FDA in the family; default unchanged
    ARMS["BFDA"] = os.environ["D113_FDA_RUN"]
FAMILY = ["BFDA", "BRH", "BCC", "AQ1", "AQ2", "AQ3"]
OTHER = ["B0", "S11DC", "STF", "B250", "B500", "B1000"]
CANDIDATES = ["S1", "S2", "S3", "S4", "S5", "S9", "S11", "S12", "S13", "S14", "S15", "S16"]   # zero-label rows of full_table
NB = 2000


def ld(t, sp):
    d = np.load(rd(t) / f"{sp}_predictions.npz"); return d["y_true"][:, 0].astype(bool), d["y_prob"][:, 0]


def f1(y, q):
    tp = (q & y).sum(); return 2 * tp / max(2 * tp + (q & ~y).sum() + (~q & y).sum(), 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0,1,2"); a = ap.parse_args()
    SEEDS = [int(x) for x in a.seeds.split(",")]; NS = len(SEEDS)
    d107 = ROOT / "results/d107_table.json"
    if d107.exists():
        ch = json.loads(d107.read_text())["val_choice"]["CG25"]
        ARMS["CG25"] = {"CGP25": "syn_cyclegan_plain_b0_s{s}_p0.25", "CGR25": "syn_cyclegan_residual_b0_s{s}_p0.25"}[ch]
        FAMILY.append("CG25"); print(f"D107 read found: CycleGAN arm {ch} added to the Holm family", flush=True)
    tl = pd.read_csv(ROOT / "data/finetune_joint/test/labels.csv"); pid = tl.file.astype(str).str.split(".").str[0].values
    pats = np.unique(pid); ix = {p: np.where(pid == p)[0] for p in pats}; yt = ld("lebase_b0_s0", "test")[0]
    rng = np.random.default_rng(1); BOOT = [np.concatenate([ix[p] for p in rng.choice(pats, len(pats), replace=True)]) for _ in range(NB)]
    rng2 = np.random.default_rng(2)
    # per arm, per seed: AUROC and F1 on every patient resample (computed once)
    A = {}
    for k, t in ARMS.items():
        P, Q = [], []
        SK = [s for s in SEEDS if (rd(t.format(s=s)) / "test_predictions.npz").exists()]   # arms outside D109 keep 3 seeds
        assert len(SK) >= 3, (k, SK); NK = len(SK)
        for s in SK:
            y, p = ld(t.format(s=s), "test"); assert (y == yt).all(), k
            thr = float(np.load(rd(t.format(s=s)) / "test_predictions.npz")["thresholds"][0])
            P.append(p); Q.append(p >= thr)
        auc = np.array([[roc_auc_score(yt[b], P[s][b]) for b in BOOT] for s in range(NK)])     # NK x NB
        ff = np.array([[f1(yt[b], Q[s][b]) for b in BOOT] for s in range(NK)])
        A[k] = dict(auc=auc, f1=ff, full_auc=[roc_auc_score(yt, P[s]) for s in range(NK)], full_f1=[f1(yt, Q[s]) for s in range(NK)],
                    seeds=rng2.integers(0, NK, size=(NB, NK)), n_seeds=NK)                      # independent seed draws per arm
        print(f"{k:6s} test AUROC {np.mean(A[k]['full_auc']):.4f} F1 {np.mean(A[k]['full_f1']):.4f} ({NK} seeds) per seed {[round(v, 4) for v in A[k]['full_auc']]}", flush=True)

    def contrast(x, y):
        out = {"x": x, "y": y, "dauc": float(np.mean(A[x]["full_auc"]) - np.mean(A[y]["full_auc"])),
               "df1": float(np.mean(A[x]["full_f1"]) - np.mean(A[y]["full_f1"]))}
        for m in ("auc", "f1"):
            pat = A[x][m].mean(0) - A[y][m].mean(0)
            two = np.array([A[x][m][A[x]["seeds"][i], i].mean() - A[y][m][A[y]["seeds"][i], i].mean() for i in range(NB)])
            for name, d in (("patient", pat), ("two_level", two)):
                out[f"{m}_{name}_CI"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
                out[f"{m}_{name}_p"] = float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        return out

    res = {"family": [contrast("S11", y) for y in FAMILY], "other": [contrast("S11", y) for y in OTHER]}
    for lev in ("patient", "two_level"):
        for m in ("auc", "f1"):
            fam = res["family"]; order = sorted(range(len(fam)), key=lambda i: fam[i][f"{m}_{lev}_p"]); run = 0.0
            for rank, i in enumerate(order):
                run = max(run, min(1.0, (len(fam) - rank) * fam[i][f"{m}_{lev}_p"])); fam[i][f"{m}_{lev}_p_holm"] = run
    for grp in ("family", "other"):
        for c in res[grp]:
            print(f"S11 - {c['y']:6s} AUROC {c['dauc']:+.4f} patient [{c['auc_patient_CI'][0]:+.4f}, {c['auc_patient_CI'][1]:+.4f}]"
                  f"{' Holm p %.3f' % c['auc_patient_p_holm'] if 'auc_patient_p_holm' in c else ' p %.3f' % c['auc_patient_p']}"
                  f" | two-level [{c['auc_two_level_CI'][0]:+.4f}, {c['auc_two_level_CI'][1]:+.4f}]"
                  f"{' Holm p %.3f' % c['auc_two_level_p_holm'] if 'auc_two_level_p_holm' in c else ' p %.3f' % c['auc_two_level_p']}"
                  f" | F1 {c['df1']:+.4f} two-level [{c['f1_two_level_CI'][0]:+.4f}, {c['f1_two_level_CI'][1]:+.4f}]", flush=True)
    ft = json.loads((ROOT / "results/full_table_2026-10-16_1421.json").read_text()); ft = ft.get("arms", ft)
    cand = {k: {"gen": ft[k]["gen"], "steps": ft[k]["steps"], "swap": ft[k]["swap"], "val_auc": float(ft[k]["val"]),
                "test_auc": float(np.mean(ft[k]["auc"])), "test_f1": float(np.mean(ft[k]["f1"]))} for k in CANDIDATES}
    ta = [v["test_auc"] for v in cand.values()]; tf = [v["test_f1"] for v in cand.values()]
    b0 = float(np.mean(A["B0"]["full_auc"]))
    res["candidates"] = {"arms": cand, "test_auc_range": [min(ta), max(ta)], "test_f1_range": [min(tf), max(tf)],
                         "n_above_B0_mean": int(sum(v > b0 for v in ta)), "n": len(ta), "B0_test_auc": b0,
                         "chosen": max(cand, key=lambda k: cand[k]["val_auc"])}
    print(f"12 candidates: test AUROC {min(ta):.4f}-{max(ta):.4f}, F1 {min(tf):.4f}-{max(tf):.4f}; "
          f"{res['candidates']['n_above_B0_mean']}/{len(ta)} above BRSET-only {b0:.4f}; validation chose {res['candidates']['chosen']}", flush=True)
    res["seeds"] = SEEDS
    res["arms"] = {k: {"run": ARMS[k], "n_seeds": A[k]["n_seeds"], "auc": float(np.mean(A[k]["full_auc"])), "f1": float(np.mean(A[k]["full_f1"])),
                       "auc_seeds": [float(v) for v in A[k]["full_auc"]], "f1_seeds": [float(v) for v in A[k]["full_f1"]]} for k in A}
    out = ROOT / ("results/extra_stats_2026-10-20" + ("_withcg" if "CG25" in ARMS else "") + (f"_{NS}seed" if NS != 3 else "") + ("_fdafov" if os.environ.get("D113_FDA_RUN") else "") + ".json")
    out.write_text(json.dumps(res, indent=1)); print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
