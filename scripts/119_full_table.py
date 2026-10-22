"""Every joint-training arm in one table, straight from saved predictions.
Δ and CI are against the baseline with the SAME number of real mBRSET labels,
patient-cluster bootstrap over the 193 test patients (2,000 resamples)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path("/home/users/sthummala2/brset-convnextv2"); CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
GRID = np.arange(0.05, 0.95, 0.01)
# id, generator, gen steps, swap %, real labels, template, baseline id
A = [("B0", "none", "-", 0, 0, "lebase_b0_s{s}", None),
     ("B250", "none", "-", 0, 250, "lebase_b250_s{s}", None),
     ("B500", "none", "-", 0, 500, "lebase_b500_s{s}", None),
     ("B1000", "none", "-", 0, 1000, "lebase_b1000_s{s}", None),
     ("B2000", "none", "-", 0, 2000, "lebase_b2000_s{s}", None),
     ("S1", "gen_tgt", "40k", 25, 0, "syn_tgt_b0_s{s}_p0.25", "B0"),
     ("S2", "gen_tgt", "40k", 50, 0, "syn_tgt_b0_s{s}", "B0"),
     ("S3", "gen_tgt", "40k", 75, 0, "syn_tgt_b0_s{s}_p0.75", "B0"),
     ("S4", "gen_both", "40k", 50, 0, "syn_both_b0_s{s}", "B0"),
     ("S5", "gen_both", "60k", 50, 0, "syn_bothld60_b0_s{s}", "B0"),
     ("S6", "gen_tgt", "40k", 50, 250, "syn_tgt_b250_s{s}", "B250"),
     ("S7", "gen_both", "40k", 50, 250, "syn_both_b250_s{s}", "B250"),
     ("S8", "gen_both", "60k", 50, 250, "syn_bothld60_b250_s{s}", "B250"),
     ("S9", "gen_tgt", "60k", 50, 0, "syn_tgtld60_b0_s{s}", "B0"),
     ("S10", "gen_tgt", "60k", 50, 250, "syn_tgtld60_b250_s{s}", "B250"),
     ("S11", "gen_both", "40k", 25, 0, "syn_both_b0_s{s}_p0.25", "B0"),
     ("S12", "gen_both", "40k", 75, 0, "syn_both_b0_s{s}_p0.75", "B0"),
     ("AB1", "physics only", "-", 50, 0, "syn_physonly_b0_s{s}", "B0"),
     ("AB2", "diffusion only (gen_tgt)", "40k", 50, 0, "syn_tgtnone_b0_s{s}", "B0"),
     ("AB3", "no physics start (gen_tgt)", "40k", 50, 0, "syn_tgtsrc_b0_s{s}", "B0"),
     ("S13", "gen_both", "60k", 25, 0, "syn_bothld60_b0_s{s}_p0.25", "B0"),
     ("S14", "gen_tgt", "60k", 25, 0, "syn_tgtld60_b0_s{s}_p0.25", "B0"),
     ("S15", "gen_both", "100k", 50, 0, "syn_bothld100_b0_s{s}", "B0"),
     ("S16", "gen_both", "100k", 25, 0, "syn_bothld100_b0_s{s}_p0.25", "B0"),
     # D073/D075 self-training on unlabelled mBRSET (teacher pseudo-labels)
     ("STA", "self-train A: syn teacher x3 strict + twins", "40k", 25, 0, "pst_A_syn3strict_s{s}", "B0"),
     ("STB", "self-train B: syn teacher x3 loose + twins", "40k", 25, 0, "pst_B_syn3loose_s{s}", "B0"),
     ("STC", "self-train C: BRSET-only teacher, no twins", "-", 0, 0, "pst_C_base1strict_s{s}", "B0"),
     ("STD", "self-train D: syn teacher x1 strict + twins", "40k", 25, 0, "pst_D_syn1strict_s{s}", "B0"),
     ("STE", "self-train E: syn teacher x3 strict, no twins", "-", 0, 0, "pst_E_syn3strict_notwins_s{s}", "B0"),
     # D076 baselines and final-configuration ablations (all 25 % swap)
     ("BFDA", "FDA twins (beta 0.01)", "-", 25, 0, "syn_fda_b0_s{s}_p0.25", "B0"),
     ("BRH", "Reinhard colour-transfer twins", "-", 25, 0, "syn_reinhard_b0_s{s}_p0.25", "B0"),
     ("BCC", "RetSyn-style class-conditioned generator", "40k", 25, 0, "syn_classcond_b0_s{s}_p0.25", "B0"),
     ("AQ1", "physics only", "-", 25, 0, "syn_physonly_b0_s{s}_p0.25", "B0"),
     ("AQ2", "diffusion only (gen_both)", "40k", 25, 0, "syn_bothnone_b0_s{s}_p0.25", "B0"),
     ("AQ3", "no physics start (gen_both)", "40k", 25, 0, "syn_bothsrc_b0_s{s}_p0.25", "B0"),
     # D078 matched self-training (single teacher, loose cut-off), D079 eye-level teacher, D080 same-eye consistency
     ("STF", "self-train F: BRSET teacher x1 loose, no twins", "-", 0, 0, "pst_F_base1loose_notwins_s{s}", "B0"),
     ("STG", "self-train G: syn teacher x1 loose + twins", "40k", 25, 0, "pst_G_syn1loose_twins_s{s}", "B0"),
     ("STH", "self-train H: syn teacher x1 loose, no twins", "-", 0, 0, "pst_H_syn1loose_notwins_s{s}", "B0"),
     ("STI", "self-train I: syn teacher x3 eye-level agree + twins", "40k", 25, 0, "pst_I_syn3eyeagree_twins_s{s}", "B0"),
     ("MV1", "twins + same-eye consistency w1", "40k", 25, 0, "syn_both_b0_s{s}_p0.25_mv1", "B0"),
     ("MV5", "twins + same-eye consistency w5", "40k", 25, 0, "syn_both_b0_s{s}_p0.25_mv5", "B0"),
     # D082 mBRSET -> mBRSET target and synthetic-from-mBRSET (trained on synthetic only), D081 synthetic-only from BRSET
     ("MBO", "mBRSET->mBRSET (real photos, 3,402 labels)", "-", 0, 3402, "mbo_lebase_s{s}", None),
     ("MBS", "SYNTHETIC from mBRSET, re-drawn (s0.4), synthetic only", "40k", 100, 3402, "mbo_synsrc_s{s}", "MBO"),
     ("MBN", "SYNTHETIC from mBRSET, from structure map (noise), synthetic only", "40k", 100, 3402, "mbo_synnone_s{s}", "MBO"),
     ("SYN1", "SYNTHETIC from BRSET only (swap 100 %), 0 phone labels", "40k", 100, 0, "syn_both_b0_s{s}_p1.0", "B0"),
     ("SYN1K3", "SYNTHETIC from BRSET only, 3 versions, 0 phone labels", "40k", 100, 0, "syn_both_b0_s{s}_p1.0_k3", "B0"),
     # D083-D085 synthetic ONLY from both datasets, compared with the real mBRSET->mBRSET target
     ("ALLK1", "SYNTHETIC ONLY from BRSET+mBRSET, 1 version", "40k", 100, 3402, "syn_allsyn_k1_s{s}", "MBO"),
     ("ALLK3", "SYNTHETIC ONLY from BRSET+mBRSET, 3 versions", "40k", 100, 3402, "syn_allsyn_k3_s{s}", "MBO"),
     ("ALL025", "SYNTHETIC ONLY from BRSET+mBRSET, mBRSET re-drawn at 0.25", "40k", 100, 3402, "syn_allsyn_k1_s025_s{s}", "MBO"),
     # reference only (real images): joint BRSET + mBRSET. D086 real+synthetic arms withdrawn (D087), not listed.
     ("JREAL", "REAL BRSET + all mBRSET (joint), no synthetic", "-", 0, 3402, "mbo_jointreal_s{s}", "MBO"),
     # D090/D091 synthetic ONLY, no image derived from a labelled mBRSET photo
     ("C1DC", "SYNTHETIC ONLY: disease-preserving BRSET twins (DC), 0 phone labels", "dc", 100, 0, "syn_both_dc_b0_s{s}_p1.0", "B0"),
     ("C5TJ", "SYNTHETIC ONLY: Translator-v3 (phone-judge) BRSET twins, 0 phone labels", "tj", 100, 0, "syn_both_tj_b0_s{s}_p1.0", "B0"),
     ("C2A", "SYNTHETIC ONLY: new eyes, cascade A (w4.0 s0.4)", "cascade", 100, 3402, "syn_c2_A_s{s}", "MBO"),
     ("C3A", "SYNTHETIC ONLY: DC twins + new eyes A", "cascade", 100, 3402, "syn_c3_A_s{s}", "MBO"),
     ("C2B", "SYNTHETIC ONLY: new eyes, cascade B (w2.5 s0.6)", "cascade", 100, 3402, "syn_c2_B_s{s}", "MBO"),
     ("C3B", "SYNTHETIC ONLY: DC twins + new eyes B", "cascade", 100, 3402, "syn_c3_B_s{s}", "MBO")]
import sys as _sys
_have = lambda tpl: all((((CLS if tpl.startswith(("syn_", "pst_", "mbo_")) else ROOT / "results") / tpl.format(s=s)) / "test_predictions.npz").exists() for s in (0, 1, 2))
A = [a for a in A if _have(a[5])]
# head-to-head comparisons fixed before reading (D070: full method vs stage 1 alone)
PAIRS = [("S2", "AB1", "full method (gen_tgt 40k, 50%) minus physics only (50%)"),
         ("S2", "AB2", "full method minus diffusion only (no physics start, pure noise)"),
         ("S2", "AB3", "full method minus diffusion started from the plain BRSET image"),
         ("S11", "AQ1", "FINAL (gen_both 40k 25%) minus physics only 25%"),
         ("S11", "AQ2", "FINAL minus gen_both diffusion only 25%"),
         ("S11", "AQ3", "FINAL minus gen_both no physics start 25%"),
         ("S11", "BFDA", "FINAL minus FDA twins"),
         ("S11", "BRH", "FINAL minus Reinhard twins"),
         ("S11", "BCC", "FINAL minus RetSyn-style class-conditioned generator"),
         ("STA", "S11", "self-train A minus FINAL (does self-training add to synthesis?)"),
         ("STA", "STC", "self-train A minus standard self-training C"),
         ("STD", "STC", "self-train D minus C (synthetic teacher + twins vs BRSET-only teacher, both single teacher)"),
         ("STE", "STC", "self-train E minus C (synthetic teacher x3 vs BRSET-only teacher, neither with twins)"),
         ("STA", "STE", "self-train A minus E (do the twins add once the teacher is synthetic?)"),
         ("STG", "STF", "D078 primary: our pipeline (syn teacher + twins) minus standard self-training, matched"),
         ("STG", "STH", "D078: twins effect inside self-training (same synthetic teacher)"),
         ("STH", "STF", "D078: teacher effect (synthetic vs BRSET-only teacher, no twins)"),
         ("STI", "S11", "D079: eye-level agreement self-training minus FINAL twins-only"),
         ("MV1", "S11", "D080: same-eye consistency w1 minus FINAL"),
         ("MV5", "S11", "D080: same-eye consistency w5 minus FINAL"),
         ("MBS", "MBO", "D082: synthetic-from-mBRSET (re-drawn) minus REAL mBRSET->mBRSET"),
         ("MBN", "MBO", "D082: synthetic-from-mBRSET (structure map only) minus REAL mBRSET->mBRSET"),
         ("SYN1", "MBO", "D082: synthetic-only from BRSET (0 phone labels) minus REAL mBRSET->mBRSET"),
         ("STB", "MBO", "D082: ST-B (0 phone labels) minus REAL mBRSET->mBRSET"),
         ("S11", "MBO", "D082: twins-only final pick (0 phone labels) minus REAL mBRSET->mBRSET"),
         ("STG", "MBO", "D082: ST-G (0 phone labels) minus REAL mBRSET->mBRSET"),
         ("STI", "MBO", "D082: ST-I (0 phone labels) minus REAL mBRSET->mBRSET"),
         ("MV1", "MBO", "D082: twins + same-eye consistency w1 (0 phone labels) minus REAL mBRSET->mBRSET"),
         ("ALLK1", "MBO", "D083: SYNTHETIC ONLY from both (1 version) minus REAL mBRSET->mBRSET"),
         ("ALLK3", "MBO", "D084: SYNTHETIC ONLY from both (3 versions) minus REAL mBRSET->mBRSET"),
         ("ALL025", "MBO", "D085: SYNTHETIC ONLY from both (mBRSET at 0.25) minus REAL mBRSET->mBRSET"),
         ("SYN1K3", "MBO", "D084: SYNTHETIC ONLY from BRSET, 3 versions (0 phone labels) minus REAL mBRSET->mBRSET"),
         ("ALLK1", "MBS", "D083: adding BRSET twins to the synthetic-from-mBRSET set"),
         ("ALLK3", "ALLK1", "D084: 3 versions minus 1 version (both datasets)"),
         ("ALLK1", "B0", "synthetic-only from both minus BRSET->mBRSET zero-shot (NOT like-for-like: uses mBRSET labels)"),
         ("JREAL", "MBO", "reference: real joint BRSET + mBRSET minus real mBRSET->mBRSET"),
         ("C1DC", "SYN1", "D090: disease-preserving twins minus old twins (both synthetic only, 0 phone labels)"),
         ("C1DC", "B0", "D090: DC twins only minus REAL BRSET->mBRSET zero-shot"),
         ("C1DC", "MBO", "D091 C1: DC twins only minus REAL mBRSET->mBRSET"),
         ("C5TJ", "MBO", "D095 C5: Translator-v3 twins only minus REAL mBRSET->mBRSET"),
         ("C5TJ", "C1DC", "D095: Translator-v3 minus Translator-v2 (both synthetic only)"),
         ("C2A", "MBO", "D091 C2 (A): new eyes only minus REAL mBRSET->mBRSET"),
         ("C3A", "MBO", "D091 C3 (A): DC twins + new eyes minus REAL mBRSET->mBRSET"),
         ("C2B", "MBO", "D091 C2 (B): new eyes only minus REAL mBRSET->mBRSET"),
         ("C3B", "MBO", "D091 C3 (B): DC twins + new eyes minus REAL mBRSET->mBRSET"),
         ("C3A", "C1DC", "D091: adding new eyes (A) to the DC twins"),
         ("C3B", "C1DC", "D091: adding new eyes (B) to the DC twins"),
         # added 14 Oct 12:40 AFTER the D073 validation rule selected ST-B (mean val DR F1 0.866)
         ("STB", "S11", "[post-selection] self-train B minus FINAL"),
         ("STB", "STC", "[post-selection] self-train B minus standard self-training C"),
         ("STB", "B500", "[post-selection] self-train B minus 500 real labels"),
         ("STB", "B250", "[post-selection] self-train B minus 250 real labels")]
rd = lambda t: (CLS if t.startswith(("syn_", "pst_", "mbo_")) else ROOT / "results") / t
def ld(t, sp):
    d = np.load(rd(t) / f"{sp}_predictions.npz"); return d["y_true"][:, 0].astype(bool), d["y_prob"][:, 0]
def f1(y, q): tp = (q & y).sum(); return 2 * tp / max(2 * tp + (q & ~y).sum() + (~q & y).sum(), 1)
def f1c(y, p):
    P = p[None] >= GRID[:, None]; tp = (P & y).sum(1); return 2 * tp / np.maximum(2 * tp + (P & ~y).sum(1) + (y.sum() - tp), 1)
def bmed(y, p):
    r = np.random.default_rng(0); ts = [GRID[int(np.argmax(f1c(y[i], p[i])))] for i in (r.integers(0, len(y), len(y)) for _ in range(200)) if y[i].sum() >= 2]
    return float(np.median(ts))
tl = pd.read_csv(ROOT / "data/finetune_joint/test/labels.csv"); pid = tl.file.astype(str).str.split(".").str[0].values
pats = np.unique(pid); ix = {p: np.where(pid == p)[0] for p in pats}; yt = ld("lebase_b0_s0", "test")[0]
rng = np.random.default_rng(1); BOOT = [np.concatenate([ix[p] for p in rng.choice(pats, len(pats), replace=True)]) for _ in range(2000)]
R = {}
for aid, gen, st, sw, lab, tpl, base in A:
    V = [ld(tpl.format(s=s), "val") for s in (0, 1, 2)]; T = [ld(tpl.format(s=s), "test") for s in (0, 1, 2)]
    thr = [float(np.load(rd(tpl.format(s=s)) / "test_predictions.npz")["thresholds"][0]) for s in (0, 1, 2)]
    valf1 = [f1(v[0], v[1] >= c) for v, c in zip(V, thr)]      # validation DR F1 at each run's own cut-off
    from sklearn.metrics import roc_curve as _roc
    def _s90(y, p):
        fpr, tpr, _ = _roc(y, p); return float(np.interp(0.10, fpr, tpr))
    sens90 = [_s90(yt, tt[1]) for tt in T]                      # D079 co-primary: sensitivity at 90 % specificity
    pv, pt = np.mean([v[1] for v in V], 0), np.mean([t[1] for t in T], 0)
    R[aid] = dict(gen=gen, steps=st, swap=sw, labels=lab, tpl=tpl.replace("_s{s}", "_s*"), base=base, valf1=valf1, sens90=sens90,
                  val=np.mean([roc_auc_score(V[0][0], v[1]) for v in V]), auc=[roc_auc_score(yt, t[1]) for t in T],
                  f1=[f1(yt, t[1] >= c) for t, c in zip(T, thr)], ensf1=f1(yt, pt >= bmed(V[0][0], pv)),
                  P=[t[1] for t in T], Q=[t[1] >= c for t, c in zip(T, thr)])
for aid, r in R.items():
    if r["base"]:
        b = R[r["base"]]
        dA = [np.mean([roc_auc_score(yt[s], p[s]) for p in r["P"]]) - np.mean([roc_auc_score(yt[s], p[s]) for p in b["P"]]) for s in BOOT]
        dF = [np.mean([f1(yt[s], q[s]) for q in r["Q"]]) - np.mean([f1(yt[s], q[s]) for q in b["Q"]]) for s in BOOT]
        r["aucCI"] = list(np.percentile(dA, [2.5, 97.5])); r["f1CI"] = list(np.percentile(dF, [2.5, 97.5]))
        r["up"] = sum(x > y for x, y in zip(r["auc"], b["auc"]))
print(f"{'ID':<6}{'generator':<10}{'gen steps':>9}{'swap %':>7}{'real labels':>12}{'val AUROC':>10}{'test AUROC':>11}{'Δ AUROC [95% CI]':>27}{'seeds up':>9}{'test F1':>8}{'Δ F1 [95% CI]':>26}{'ens F1':>8}")
for aid, r in R.items():
    a, f = np.mean(r["auc"]), np.mean(r["f1"])
    if r["base"]:
        b = R[r["base"]]; da, df = a - np.mean(b["auc"]), f - np.mean(b["f1"])
        ca = f"{da:+.4f} [{r['aucCI'][0]:+.3f},{r['aucCI'][1]:+.3f}]"; cf = f"{df:+.4f} [{r['f1CI'][0]:+.3f},{r['f1CI'][1]:+.3f}]"; up = f"{r['up']}/3"
    else:
        ca = cf = "baseline"; up = ""
    print(f"{aid:<6}{r['gen']:<10}{r['steps']:>9}{r['swap']:>7}{r['labels']:>12}{r['val']:>10.4f}{a:>11.4f}{ca:>27}{up:>9}{f:>8.4f}{cf:>26}{r['ensf1']:>8.4f}")
print("\nper seed (test AUROC / test F1):")
for aid, r in R.items():
    print(f"  {aid:<5}{r['tpl']:<28}" + "   ".join(f"s{s}: {a:.4f} / {f:.4f}" for s, (a, f) in enumerate(zip(r["auc"], r["f1"]))))
print("\nhead-to-head (paired patient bootstrap, same 2,000 resamples):")
PAIR_OUT = []
for x, y, what in PAIRS:
    if x in R and y in R:
        dA = [np.mean([roc_auc_score(yt[s], p[s]) for p in R[x]["P"]]) - np.mean([roc_auc_score(yt[s], p[s]) for p in R[y]["P"]]) for s in BOOT]
        dF = [np.mean([f1(yt[s], q[s]) for q in R[x]["Q"]]) - np.mean([f1(yt[s], q[s]) for q in R[y]["Q"]]) for s in BOOT]
        dS = [np.mean([_s90(yt[s], p[s]) for p in R[x]["P"]]) - np.mean([_s90(yt[s], p[s]) for p in R[y]["P"]]) for s in BOOT]
        print(f"  {what}: val AUROC {R[x]['val'] - R[y]['val']:+.4f} | test AUROC {np.mean(R[x]['auc']) - np.mean(R[y]['auc']):+.4f} "
              f"[{np.percentile(dA, 2.5):+.4f}, {np.percentile(dA, 97.5):+.4f}] | test F1 {np.mean(R[x]['f1']) - np.mean(R[y]['f1']):+.4f} "
              f"[{np.percentile(dF, 2.5):+.4f}, {np.percentile(dF, 97.5):+.4f}] | seeds x>y on AUROC {sum(a > b for a, b in zip(R[x]['auc'], R[y]['auc']))}/3")
        PAIR_OUT.append(dict(x=x, y=y, what=what, dval=R[x]["val"] - R[y]["val"], dauc=float(np.mean(R[x]["auc"]) - np.mean(R[y]["auc"])),
                             aucCI=[float(np.percentile(dA, 2.5)), float(np.percentile(dA, 97.5))],
                             df1=float(np.mean(R[x]["f1"]) - np.mean(R[y]["f1"])), f1CI=[float(np.percentile(dF, 2.5)), float(np.percentile(dF, 97.5))],
                             seeds=int(sum(a > b for a, b in zip(R[x]["auc"], R[y]["auc"]))),
                             dsens=float(np.mean(R[x]["sens90"]) - np.mean(R[y]["sens90"])),
                             sensCI=[float(np.percentile(dS, 2.5)), float(np.percentile(dS, 97.5))]))
        print(f"      sens@spec90 {np.mean(R[x]['sens90']) - np.mean(R[y]['sens90']):+.4f} [{np.percentile(dS, 2.5):+.4f}, {np.percentile(dS, 97.5):+.4f}]")
for r in R.values():
    r.pop("P"); r.pop("Q")
R["_pairs"] = PAIR_OUT
(ROOT / ("results/full_table_" + __import__("time").strftime("%Y-%m-%d_%H%M") + ".json")).write_text(json.dumps(R, indent=1, default=float))
