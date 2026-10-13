#!/usr/bin/env python
"""Model forensics: WHERE and HOW the BRSET->mBRSET classifier fails.

CPU-only. Reads nothing but cached prediction vectors
(results/<task>/test_predictions.npz, results/<task>/val_predictions.npz) plus
the test label table, and writes a single aggregate JSON.

DUA NOTE. Nothing patient-identifying is written out. The JSON contains only
aggregate statistics and *row indices* into the 732-row ordered test manifest
(0..731). No filenames, no patient identifiers, no per-patient label rows.
Patients appear only as counts and as anonymous ordinals assigned by sorting.

ROW-ORDERING ASSUMPTION (verified, see `ordering_verification` in the output).
scripts/30_train_strong_baseline.py builds the test set with
BRSETMultiLabel(data_path/"test") which reads labels.csv once and indexes it
with .iloc[idx]; the test DataLoader is constructed with shuffle=False, and
run_inference concatenates batches in order. Therefore npz row i corresponds to
labels.csv data row i. This script additionally asserts, for every run it uses,
that the stored y_true is bit-identical to the label columns read from the CSV
in CSV order. Caveat recorded honestly: the value check alone would also pass
under a permutation that only exchanges rows carrying an identical (DR, ME)
label pair; the code-path argument above is what rules that out.

Usage:
  python scripts/93_model_forensics.py                       # all sections
  python scripts/93_model_forensics.py --sections 1,6
  python scripts/93_model_forensics.py --label-index 1        # macular edema
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
LABEL_COLS = ["diabetic_retinopathy", "macular_edema"]
THR_GRID = np.arange(0.05, 0.95, 0.01)   # exactly the trainer's grid
THR_BOOT = 200                           # exactly the trainer's --thr_bootstrap
AUC_NOISE = 0.0142                       # seed-only noise floor, DR AUC
F1_NOISE = 0.0199                        # seed-only noise floor, DR F1

# ---------------------------------------------------------------------------
# Model registry. Eligibility is read off the launch scripts, not guessed:
#   scripts/71_label_efficiency_baseline.slurm  --target_budget $BUDGET
#   scripts/73_label_balance_control.slurm      --target_budget 0 --target_pos_rate .234
#   scripts/89_synth_downstream.slurm           --target_budget 0 --translator_ckpt ...
#   scripts/68..70_aspect_preserve_seed*.slurm  no --target_budget  => default -1
#   scripts/51_ceiling_brset_mbrset_joint.slurm / 66 / 67  => default -1
# apply_target_budget() in 30_train_strong_baseline.py: -1 keeps ALL target images.
# "zero-shot eligible" = ZERO real target images in TRAINING. Note the flag in
# `caveats`: every run still tuned its decision threshold on the 725-image real
# target validation set, so these arms are zero-shot in training data only.
ZERO_SHOT = [
    "lebase_b0_s0", "lebase_b0_s1", "lebase_b0_s2",
    "synth_b0_s0", "synth_b0_s0_p0.5", "synth_b0_s0_p0.25", "synth_b0_s0_union1.0",
    "balctl_s0", "balctl_s1",
]
SYNTH = ["synth_b0_s0", "synth_b0_s0_p0.5", "synth_b0_s0_p0.25", "synth_b0_s0_union1.0"]
POOLED = ["brset_mbrset_joint", "joint_seed1", "joint_seed2"]
BUDGET_CURVE = {
    0: ["lebase_b0_s0", "lebase_b0_s1", "lebase_b0_s2"],
    250: ["lebase_b250_s0", "lebase_b250_s1", "lebase_b250_s2"],
    500: ["lebase_b500_s0", "lebase_b500_s1", "lebase_b500_s2"],
    1000: ["lebase_b1000_s0", "lebase_b1000_s1", "lebase_b1000_s2"],
    2000: ["lebase_b2000_s0", "lebase_b2000_s1", "lebase_b2000_s2"],
    3402: POOLED,
}
TARGET_TRAINED = [
    m for b, ms in BUDGET_CURVE.items() if b > 0 for m in ms
] + [
    "aspect_preserve_seed0", "aspect_preserve_seed1", "aspect_preserve_seed2",
    "mbrset_ft_lr2e5", "mbrset_ft_lr5e6", "mbrset_scratch_stable",
    "mbrset_part1recipe_scratch", "mbrset_finetune_from_brset",
    "convnextv2_large_mBRSET_multilabel_512",
    "convnextv2_large_mBRSET_multilabel_512_regularized",
    "convnextv2_large_mBRSET_multilabel_512_distilled_dr",
]
BASE = "lebase_b0_s0"


# ---------------------------------------------------------------------------
# loading / verification
def load_manifest(labels_csv: Path):
    rows = list(csv.DictReader(open(labels_csv)))
    y = np.array([[int(r[c]) for c in LABEL_COLS] for r in rows], dtype=float)
    files = [r["file"] for r in rows]
    patients = [f.split(".")[0] for f in files]
    return y, files, patients


def load_runs(results_dir: Path, y_ref: np.ndarray, want: list[str] | None = None):
    """Return {task: {'test': prob, 'thr': array|None, 'val': prob|None, 'val_y': ..}}."""
    runs, rejected = {}, {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        tp = d / "test_predictions.npz"
        if not tp.exists():
            continue
        if want is not None and d.name not in want:
            continue
        z = np.load(tp)
        yt = z["y_true"]
        if yt.shape != y_ref.shape:
            rejected[d.name] = f"shape {tuple(yt.shape)} != {tuple(y_ref.shape)} (not the mBRSET test set)"
            continue
        if not np.array_equal(yt, y_ref):
            rejected[d.name] = "y_true does not match the ordered label manifest"
            continue
        rec = {"test": z["y_prob"].astype(np.float64),
               "thr": z["thresholds"].astype(np.float64) if "thresholds" in z else None}
        vp = d / "val_predictions.npz"
        if vp.exists():
            v = np.load(vp)
            if v["y_true"].shape[0] == 725:
                rec["val"] = v["y_prob"].astype(np.float64)
                rec["val_y"] = v["y_true"].astype(np.float64)
        runs[d.name] = rec
    return runs, rejected


# ---------------------------------------------------------------------------
# metric helpers
def f1_from_counts(tp, fp, fn):
    den = 2.0 * tp + fp + fn
    return np.where(den > 0, 2.0 * tp / np.maximum(den, 1e-12), 0.0)


def f1_at(y, p, thr):
    pred = (p >= thr)
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum((~pred) & (y == 1)))
    return float(f1_from_counts(tp, fp, fn)), tp, fp, fn


def grid_f1_curve(y, p, grid=THR_GRID):
    B = (p[:, None] >= grid[None, :])
    tp = (y[:, None] * B).sum(0)
    fp = ((1 - y)[:, None] * B).sum(0)
    fn = y.sum() - tp
    return f1_from_counts(tp, fp, fn)


def best_grid_threshold(y, p, grid=THR_GRID):
    f1 = grid_f1_curve(y, p, grid)
    k = int(np.argmax(f1))          # first maximum, matches the trainer's loop
    return float(grid[k]), float(f1[k])


def oracle_f1(y, p):
    """Max F1 over ALL cut points, not just the 0.01 grid. This is the ceiling
    for any monotone recalibration of p, because monotone maps cannot reorder."""
    o = np.argsort(-p, kind="mergesort")
    ys = y[o]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    f1 = f1_from_counts(tp, fp, fn)
    k = int(np.argmax(f1))
    thr = float(p[o][k])
    return float(f1[k]), thr, int(k + 1)


def bootstrap_thresholds(val_y, val_p, n_boot=THR_BOOT, seed=0, grid=THR_GRID):
    """Vectorised replica of tune_thresholds(..., n_boot, seed) for ONE label.
    The trainer consumes `n_boot` draws per label in label order, so label 0
    (DR) uses the first n_boot draws of default_rng(seed)."""
    n = len(val_y)
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.integers(0, n, n) for _ in range(n_boot)])
    C = np.stack([np.bincount(r, minlength=n) for r in idx]).astype(np.float64)
    Cy = C * val_y[None, :]
    Cn = C * (1.0 - val_y)[None, :]
    npos = Cy.sum(1)
    B = (val_p[:, None] >= grid[None, :]).astype(np.float64)
    tp = Cy @ B
    fp = Cn @ B
    fn = npos[:, None] - tp
    f1 = f1_from_counts(tp, fp, fn)
    keep = npos >= 2
    if not keep.any():
        return best_grid_threshold(val_y, val_p, grid)[0]
    chosen = grid[np.argmax(f1[keep], axis=1)]
    return float(np.median(chosen))


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for k in range(bins):
        m = b == k
        if m.sum() == 0:
            continue
        e += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(e)


def overlap_coefficient(a, b, bins):
    edges = np.linspace(0.0, 1.0, bins + 1)
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    return float(np.minimum(ha, hb).sum())


def per_positive_auc_contrib(y, p):
    """c_i = P(score_i > score_neg) + .5 P(=), so AUC == mean_i c_i exactly."""
    neg = np.sort(p[y == 0])
    nneg = len(neg)
    pos = p[y == 1]
    lo = np.searchsorted(neg, pos, side="left")
    hi = np.searchsorted(neg, pos, side="right")
    return (lo + 0.5 * (hi - lo)) / nneg


def patient_bootstrap(y, pa, pb, patients, stat, n=2000, seed=0):
    """Clustered bootstrap over patients for stat(y, p) difference b - a."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(patients)))
    idx_by_pat = {u: np.flatnonzero(np.array(patients) == u) for u in uniq}
    out = []
    for _ in range(n):
        draw = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by_pat[uniq[d]] for d in draw])
        ys = y[sel]
        if ys.sum() < 2 or (1 - ys).sum() < 2:
            continue
        try:
            out.append(stat(ys, pb[sel]) - stat(ys, pa[sel]))
        except ValueError:
            continue
    out = np.array(out)
    return {"n_reps": int(len(out)), "mean": float(out.mean()),
            "ci_lo": float(np.percentile(out, 2.5)), "ci_hi": float(np.percentile(out, 97.5)),
            "frac_gt_0": float((out > 0).mean())}


def verify_stored_thresholds(runs, li, seeds=(0, 1, 2)):
    """Independent check that the cached `thresholds` really are the trainer's
    bootstrap-median val-tuned thresholds. Reproducing them from
    val_predictions.npz with the same grid, n_boot and RNG confirms that the
    val and test vectors in a run belong together, which the whole
    threshold-based analysis depends on."""
    rep = {}
    for t, r in sorted(runs.items()):
        if r["thr"] is None or "val" not in r:
            continue
        stored = float(r["thr"][li])
        got = {s: bootstrap_thresholds(r["val_y"][:, li], r["val"][:, li], seed=s)
               for s in seeds}
        match = [s for s, v in got.items() if abs(v - stored) < 1e-9]
        rep[t] = {"stored": stored,
                  "reproduced_by_seed": {str(k): v for k, v in got.items()},
                  "matching_seeds": match,
                  "exact_match": bool(match)}
    n = len(rep)
    return {"n_checked": n,
            "n_exact_match_for_some_seed": sum(1 for v in rep.values() if v["exact_match"]),
            "per_run": rep}


# ---------------------------------------------------------------------------
# sections
def section1(y, runs, li):
    """Ranking versus calibration."""
    out = {"_question": "How much of the zero-shot deficit is bad ordering (AUC) "
                        "versus a badly placed threshold?"}
    def block(task):
        p = runs[task]["test"][:, li]
        thr_stored = float(runs[task]["thr"][li]) if runs[task]["thr"] is not None else None
        d = {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p))}
        if thr_stored is not None:
            f1v, tp, fp, fn = f1_at(y, p, thr_stored)
            d.update(val_threshold=thr_stored, f1_at_val_threshold=f1v,
                     missed=int(fn), false_positives=int(fp), detected=int(tp),
                     recall=float(tp / (tp + fn)), precision=float(tp / max(tp + fp, 1)))
        of1, othr, ok = oracle_f1(y, p)
        gthr, gf1 = best_grid_threshold(y, p)
        d.update(oracle_f1_any_cutpoint=of1, oracle_threshold=othr, oracle_n_predicted_pos=ok,
                 oracle_f1_on_001_grid=gf1, oracle_threshold_on_grid=gthr)
        if thr_stored is not None:
            d["threshold_cost_f1"] = of1 - d["f1_at_val_threshold"]
        # prevalence-matched cut: predict exactly as many positives as there are
        k = int(y.sum())
        pm = float(np.sort(p)[-k])
        d["f1_at_prevalence_matched_cut"] = f1_at(y, p, pm)[0]
        d["ece_15bin"] = ece(y, p)
        d["brier"] = float(np.mean((p - y) ** 2))
        d["mean_prob_positives"] = float(p[y == 1].mean())
        d["mean_prob_negatives"] = float(p[y == 0].mean())
        # how flat is the F1/threshold curve?
        curve = grid_f1_curve(y, p)
        d["f1_curve_grid_max"] = float(curve.max())
        d["f1_within_0.01_of_max_threshold_width"] = float(
            (curve >= curve.max() - 0.01).sum() * 0.01)
        return d

    out["zero_shot"] = block(BASE)
    out["pooled_seeds"] = {t: block(t) for t in POOLED}
    pm = out["pooled_seeds"]
    out["pooled_mean"] = {k: float(np.mean([pm[t][k] for t in POOLED]))
                          for k in ["auc", "f1_at_val_threshold", "oracle_f1_any_cutpoint",
                                    "threshold_cost_f1", "ece_15bin", "brier"]}
    zs, po = out["zero_shot"], out["pooled_mean"]
    total = po["f1_at_val_threshold"] - zs["f1_at_val_threshold"]
    thr_part = zs["threshold_cost_f1"]
    rank_part = po["oracle_f1_any_cutpoint"] - zs["oracle_f1_any_cutpoint"]
    out["decomposition_f1"] = {
        "total_gap_val_threshold_f1": total,
        "recoverable_by_rethresholding_zero_shot": thr_part,
        "irreducible_ranking_gap_oracle_vs_oracle": rank_part,
        "pooled_own_threshold_loss": po["oracle_f1_any_cutpoint"] - po["f1_at_val_threshold"],
        "share_of_gap_from_threshold": float(thr_part / total) if total else None,
        "share_of_gap_from_ranking": float(rank_part / total) if total else None,
        "note": "oracle_f1 is the ceiling of ANY monotone recalibration (Platt, "
                "temperature, isotonic): a monotone map cannot change the ranking, "
                "so it cannot beat the best cut point. Residual beyond that is "
                "representation, not calibration.",
    }
    # Platt scaling fitted on the real target validation set, applied to test.
    if "val" in runs[BASE]:
        vy = runs[BASE]["val_y"][:, li]
        vp = runs[BASE]["val"][:, li]
        eps = 1e-6
        Xv = np.log(np.clip(vp, eps, 1 - eps) / (1 - np.clip(vp, eps, 1 - eps)))
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(Xv[:, None], vy)
        p = runs[BASE]["test"][:, li]
        Xt = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
        pc = lr.predict_proba(Xt[:, None])[:, 1]
        thr_c = bootstrap_thresholds(vy, lr.predict_proba(Xv[:, None])[:, 1])
        out["platt_recalibration_zero_shot"] = {
            "slope": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0]),
            "auc_after": float(roc_auc_score(y, pc)),
            "ece_after": ece(y, pc), "brier_after": float(np.mean((pc - y) ** 2)),
            "f1_at_0.5_after": f1_at(y, pc, 0.5)[0],
            "f1_at_val_tuned_threshold_after": f1_at(y, pc, thr_c)[0],
            "val_tuned_threshold_after": thr_c,
        }
    return out


def section2(y, runs, li, patients):
    """Which cases are missed, and what is the hard core."""
    pos = np.flatnonzero(y == 1)
    pb = runs[BASE]["test"][:, li]
    thr_b = float(runs[BASE]["thr"][li])
    missed_base = np.flatnonzero((y == 1) & (pb < thr_b))

    thr_models = [t for t, r in runs.items() if r["thr"] is not None]
    zs_models = [t for t in ZERO_SHOT if t in thr_models]
    tt_models = [t for t in TARGET_TRAINED if t in thr_models]

    def detected_mask(task):
        r = runs[task]
        return r["test"][:, li] >= float(r["thr"][li])

    det = {t: detected_mask(t) for t in thr_models}

    def recovery(task):
        m = det[task][missed_base]
        return {"n_recovered": int(m.sum()),
                "frac_recovered": float(m.mean()),
                "recovered_rows": [int(i) for i in missed_base[m]]}

    out = {
        "_question": "Which positives does the zero-shot model miss, who recovers "
                     "them, and how big is the core nobody gets?",
        "_unit": "IMAGES, not patients. 159 DR-positive images come from 59 "
                 "patients; an image false negative is not a patient miss.",
        "n_positive_images": int(y.sum()),
        "n_models_with_val_thresholds": len(thr_models),
        "models_excluded_from_threshold_analysis": sorted(
            t for t, r in runs.items() if r["thr"] is None),
        "zero_shot_missed": {"n": int(len(missed_base)),
                             "rows": [int(i) for i in missed_base]},
        "recovery_by_pooled": {t: recovery(t) for t in POOLED if t in det},
        "recovery_by_synthesis": {t: recovery(t) for t in SYNTH if t in det},
        "recovery_by_budget": {},
        "recovery_by_other_target_trained": {
            t: {k: v for k, v in recovery(t).items() if k != "recovered_rows"}
            for t in tt_models if t not in POOLED and not t.startswith("lebase_")},
    }
    for b, ms in BUDGET_CURVE.items():
        if b == 0:
            continue
        vals = [recovery(t)["n_recovered"] for t in ms if t in det]
        out["recovery_by_budget"][str(b)] = {
            "per_seed_n_recovered": vals,
            "mean": float(np.mean(vals)) if vals else None}

    # hard cores
    def core(model_list, label):
        if not model_list:
            return None
        D = np.stack([det[t] for t in model_list])          # (M, 732)
        never = np.flatnonzero((y == 1) & (~D.any(0)))
        always = np.flatnonzero((y == 1) & D.all(0))
        hit_counts = D[:, pos].sum(0)
        return {"n_models": len(model_list), "models": model_list,
                "n_never_detected": int(len(never)),
                "never_detected_rows": [int(i) for i in never],
                "frac_of_positives": float(len(never) / y.sum()),
                "n_always_detected": int(len(always)),
                "detector_count_histogram": {
                    str(k): int((hit_counts == k).sum())
                    for k in range(len(model_list) + 1) if (hit_counts == k).sum()},
                "_label": label}

    out["hard_core"] = {
        "all_models": core(thr_models, "every run with a val-tuned threshold"),
        "zero_shot_eligible_only": core(zs_models, "no real target training images"),
        "target_trained_only": core(tt_models, "trained with real target images"),
    }
    # threshold-free version: ranked below the median negative by EVERY model
    allm = sorted(runs)
    C = np.stack([per_positive_auc_contrib(y, runs[t]["test"][:, li]) for t in allm])
    best = C.max(0)
    out["hard_core_threshold_free"] = {
        "_definition": "per-positive AUC contribution c = fraction of the 573 "
                       "negatives it outranks; max taken over ALL 732-row runs "
                       "including ones without thresholds",
        "n_models": len(allm),
        "n_below_median_negative_for_every_model": int((best < 0.5).sum()),
        "rows_below_median_negative_for_every_model":
            [int(pos[i]) for i in np.flatnonzero(best < 0.5)],
        "n_below_0.9_for_every_model": int((best < 0.9).sum()),
        "mean_best_c": float(best.mean()),
        "median_best_c": float(np.median(best)),
    }
    # patient level, counts only (anonymous ordinals)
    pat = np.array(patients)
    pos_pat = sorted(set(pat[y == 1]))
    ordinal = {p: i for i, p in enumerate(sorted(set(pat)))}
    def pat_missed(task):
        d = det[task]
        return sorted(ordinal[p] for p in pos_pat
                      if not d[(pat == p) & (y == 1)].any())
    base_pm = pat_missed(BASE)
    core_models = thr_models
    never_pat = set(ordinal[p] for p in pos_pat)
    for t in core_models:
        never_pat &= set(pat_missed(t))
    out["patient_level"] = {
        "_definition": "a DR-positive patient is detected if ANY of their "
                       "DR-positive images is predicted positive; identifiers "
                       "replaced by anonymous ordinals",
        "n_dr_positive_patients": len(pos_pat),
        "zero_shot_patients_fully_missed": len(base_pm),
        "pooled_patients_fully_missed": {t: len(pat_missed(t)) for t in POOLED if t in det},
        "synthesis_patients_fully_missed": {t: len(pat_missed(t)) for t in SYNTH if t in det},
        "n_patients_missed_by_every_model": len(never_pat),
        "patient_ordinals_missed_by_every_model": sorted(never_pat),
    }
    return out


def section3(y, runs, li, patients):
    """Score distribution: shift or overlap?"""
    def block(task):
        p = runs[task]["test"][:, li]
        pp, pn = p[y == 1], p[y == 0]
        eps = 1e-6
        lg = lambda v: np.log(np.clip(v, eps, 1 - eps) / (1 - np.clip(v, eps, 1 - eps)))
        d = {
            "pos_mean": float(pp.mean()), "pos_std": float(pp.std(ddof=1)),
            "pos_median": float(np.median(pp)),
            "pos_q10": float(np.percentile(pp, 10)), "pos_q90": float(np.percentile(pp, 90)),
            "neg_mean": float(pn.mean()), "neg_std": float(pn.std(ddof=1)),
            "neg_median": float(np.median(pn)),
            "neg_q90": float(np.percentile(pn, 90)), "neg_q99": float(np.percentile(pn, 99)),
            "mean_gap": float(pp.mean() - pn.mean()),
            "cohens_d": float((pp.mean() - pn.mean()) /
                              np.sqrt(((len(pp) - 1) * pp.var(ddof=1) +
                                       (len(pn) - 1) * pn.var(ddof=1)) /
                                      (len(pp) + len(pn) - 2))),
            "logit_pos_mean": float(lg(pp).mean()), "logit_neg_mean": float(lg(pn).mean()),
            "logit_cohens_d": float((lg(pp).mean() - lg(pn).mean()) /
                                    np.sqrt(0.5 * (lg(pp).var(ddof=1) + lg(pn).var(ddof=1)))),
            "overlap_coef_50bin": overlap_coefficient(pp, pn, 50),
            "overlap_coef_20bin": overlap_coefficient(pp, pn, 20),
            "auc": float(roc_auc_score(y, p)),
            "frac_pos_below_0.5": float((pp < 0.5).mean()),
            "frac_pos_below_0.29": float((pp < 0.29).mean()),
            "frac_neg_above_0.29": float((pn > 0.29).mean()),
        }
        return d

    out = {"_question": "Is the zero-shot model under-confident on positives (a "
                        "shift a threshold could fix) or does it genuinely fail "
                        "to separate them (overlap)?",
           "zero_shot": block(BASE),
           "pooled_seeds": {t: block(t) for t in POOLED}}
    # best single additive logit shift: can a pure shift reproduce the pooled F1?
    pb = runs[BASE]["test"][:, li]
    f1_shift, thr_shift, _ = oracle_f1(y, pb)
    out["pure_shift_ceiling"] = {
        "f1": f1_shift,
        "note": "an additive logit shift is monotone, so its best F1 equals the "
                "oracle cut-point F1 reported here; any gap to pooled is overlap",
    }
    # AUC restricted to subsets
    pos = np.flatnonzero(y == 1)
    cb = per_positive_auc_contrib(y, pb)
    hard = pos[np.argsort(cb)[:int(0.25 * len(pos))]]       # worst-ranked quartile
    easy = pos[np.argsort(-cb)[:int(0.25 * len(pos))]]
    neg = np.flatnonzero(y == 0)
    def restricted(task, keep_pos):
        p = runs[task]["test"][:, li]
        sel = np.concatenate([keep_pos, neg])
        return float(roc_auc_score(y[sel], p[sel]))
    out["auc_restricted"] = {
        "_definition": "positives split by the ZERO-SHOT model's per-positive AUC "
                       "contribution (worst/best quartile), negatives kept whole",
        "hard_quartile_positives": {t: restricted(t, hard)
                                    for t in [BASE] + POOLED + SYNTH},
        "easy_quartile_positives": {t: restricted(t, easy)
                                    for t in [BASE] + POOLED + SYNTH},
        "n_hard": int(len(hard)), "n_easy": int(len(easy)),
    }
    # does the pooled model separate the zero-shot hard quartile?
    out["auc_restricted"]["delta_pooled_minus_zeroshot_hard"] = float(
        np.mean([out["auc_restricted"]["hard_quartile_positives"][t] for t in POOLED])
        - out["auc_restricted"]["hard_quartile_positives"][BASE])
    # co-label structure: positives that are also ME-positive
    return out


def section4(y, runs, li, patients, y_full, quality):
    """Does real target data fix what synthesis cannot?"""
    pb = runs[BASE]["test"][:, li]
    thr_b = float(runs[BASE]["thr"][li])
    missed = np.flatnonzero((y == 1) & (pb < thr_b))

    def det(task):
        r = runs[task]
        return r["test"][:, li] >= float(r["thr"][li])

    pooled_votes = np.sum([det(t)[missed] for t in POOLED if t in runs], axis=0)
    synth_votes = np.sum([det(t)[missed] for t in SYNTH if t in runs], axis=0)
    real_fix = pooled_votes >= 2                 # majority of 3 pooled seeds
    synth_fix = synth_votes >= 1                 # ANY synthesis arm, generous
    synth_fix_maj = synth_votes >= 2

    tab = {
        "real_yes_synth_yes": int(np.sum(real_fix & synth_fix)),
        "real_yes_synth_no": int(np.sum(real_fix & ~synth_fix)),
        "real_no_synth_yes": int(np.sum(~real_fix & synth_fix)),
        "real_no_synth_no": int(np.sum(~real_fix & ~synth_fix)),
    }
    pooled_mean = np.mean([runs[t]["test"][:, li] for t in POOLED], axis=0)
    synth_mean = np.mean([runs[t]["test"][:, li] for t in SYNTH], axis=0)

    def characterise(mask, name):
        rows = missed[mask]
        if len(rows) == 0:
            return {"_label": name, "n": 0}
        d = {"_label": name, "n": int(len(rows)), "rows": [int(i) for i in rows],
             "zero_shot_prob_mean": float(pb[rows].mean()),
             "zero_shot_prob_max": float(pb[rows].max()),
             "pooled_prob_mean": float(pooled_mean[rows].mean()),
             "synth_prob_mean": float(synth_mean[rows].mean()),
             "pooled_minus_zeroshot_prob": float((pooled_mean - pb)[rows].mean()),
             "synth_minus_zeroshot_prob": float((synth_mean - pb)[rows].mean()),
             "also_macular_edema_positive": int(y_full[rows, 1].sum()),
             "n_distinct_patients": len(set(np.array(patients)[rows])),
             "zero_shot_auc_contrib_mean": float(
                 per_positive_auc_contrib(y, pb)[
                     np.searchsorted(np.flatnonzero(y == 1), rows)].mean()),
             }
        if quality is not None:
            for k in ["sharpness", "brightness", "falloff"]:
                d[f"{k}_mean"] = float(quality[k][rows].mean())
        return d

    out = {
        "_question": "If real target images recover a specific set and synthesis "
                     "recovers none of it, what is that set like?",
        "n_zero_shot_missed": int(len(missed)),
        "definitions": {
            "real_fix": "detected by >=2 of 3 pooled (3402 real target) seeds",
            "synth_fix": "detected by >=1 of 4 synthesis arms (deliberately generous)",
        },
        "contingency": tab,
        "mcnemar_real_vs_synth_on_missed_set": None,
        "sets": {
            "real_only": characterise(real_fix & ~synth_fix, "fixed by real data only"),
            "synth_only": characterise(~real_fix & synth_fix, "fixed by synthesis only"),
            "both": characterise(real_fix & synth_fix, "fixed by both"),
            "neither": characterise(~real_fix & ~synth_fix, "fixed by neither"),
        },
        "synth_majority_contingency": {
            "real_yes_synth_yes": int(np.sum(real_fix & synth_fix_maj)),
            "real_yes_synth_no": int(np.sum(real_fix & ~synth_fix_maj)),
            "real_no_synth_yes": int(np.sum(~real_fix & synth_fix_maj)),
            "real_no_synth_no": int(np.sum(~real_fix & ~synth_fix_maj)),
        },
    }
    if quality is not None:
        ref = {}
        for name, m in (("all_positives", y == 1), ("all_negatives", y == 0),
                        ("detected_positives_zero_shot",
                         (y == 1) & (pb >= thr_b))):
            ref[name] = {"n": int(m.sum()),
                         **{f"{k}_mean": float(quality[k][m].mean())
                            for k in ["sharpness", "brightness", "falloff"]}}
        out["reference_quality"] = ref
        out["_quality_note"] = ("image appearance was already ruled out as the "
                                "cause of the gap; these columns are here only so "
                                "the recovered/unrecovered subsets can be checked "
                                "against the whole-set means rather than read in "
                                "isolation")
    b, c = tab["real_yes_synth_no"], tab["real_no_synth_yes"]
    if b + c > 0:
        out["mcnemar_real_vs_synth_on_missed_set"] = {
            "b_real_only": b, "c_synth_only": c,
            "exact_binomial_p": float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)}
    # whole-test-set view, not just the zero-shot misses
    out["whole_test_set"] = {}
    for t in POOLED + SYNTH:
        d = det(t)
        db = det(BASE)
        out["whole_test_set"][t] = {
            "fixed_fn": int(np.sum((y == 1) & ~db & d)),
            "broke_tp": int(np.sum((y == 1) & db & ~d)),
            "fixed_fp": int(np.sum((y == 0) & db & ~d)),
            "broke_tn": int(np.sum((y == 0) & ~db & d)),
        }
    return out


def section5(y, runs, li):
    """Agreement structure."""
    models = [m for m in ZERO_SHOT + POOLED if m in runs]
    P = np.stack([runs[m]["test"][:, li] for m in models])
    rho = np.zeros((len(models), len(models)))
    for i in range(len(models)):
        for j in range(len(models)):
            rho[i, j] = stats.spearmanr(P[i], P[j]).statistic
    out = {"_question": "Are the synthesis arms making the baseline's errors plus "
                        "extra ones, or different errors?",
           "models": models,
           "spearman_matrix": [[float(v) for v in r] for r in rho]}
    idx = {m: i for i, m in enumerate(models)}
    out["spearman_vs_zero_shot"] = {m: float(rho[idx[BASE], idx[m]]) for m in models}
    out["mean_spearman_within_synth"] = float(np.mean(
        [rho[idx[a], idx[b]] for a, b in itertools.combinations(
            [m for m in SYNTH if m in idx], 2)]))
    out["mean_spearman_within_pooled"] = float(np.mean(
        [rho[idx[a], idx[b]] for a, b in itertools.combinations(
            [m for m in POOLED if m in idx], 2)]))
    out["mean_spearman_within_lebase_b0_seeds"] = float(np.mean(
        [rho[idx[a], idx[b]] for a, b in itertools.combinations(
            [m for m in ["lebase_b0_s0", "lebase_b0_s1", "lebase_b0_s2"] if m in idx], 2)]))

    db = runs[BASE]["test"][:, li] >= float(runs[BASE]["thr"][li])
    eb = db != (y == 1)
    out["error_set_overlap_vs_zero_shot"] = {}
    for m in models:
        if m == BASE:
            continue
        d = runs[m]["test"][:, li] >= float(runs[m]["thr"][li])
        e = d != (y == 1)
        inherited = int(np.sum(eb & e))
        new = int(np.sum(~eb & e))
        fixed = int(np.sum(eb & ~e))
        p = float(stats.binomtest(min(new, fixed), new + fixed, 0.5).pvalue) if new + fixed else None
        out["error_set_overlap_vs_zero_shot"][m] = {
            "n_errors_baseline": int(eb.sum()), "n_errors_model": int(e.sum()),
            "inherited": inherited, "new": new, "fixed": fixed,
            "jaccard": float(inherited / max(np.sum(eb | e), 1)),
            "strict_superset_of_baseline_errors": bool(fixed == 0 and new > 0),
            "mcnemar_exact_p": p,
        }
    # Is a synthesis arm's score a noisy version of the baseline's, or a shift?
    out["prob_delta_vs_zero_shot"] = {}
    pb = runs[BASE]["test"][:, li]
    for m in models:
        if m == BASE:
            continue
        p = runs[m]["test"][:, li]
        out["prob_delta_vs_zero_shot"][m] = {
            "mean_delta_positives": float((p - pb)[y == 1].mean()),
            "mean_delta_negatives": float((p - pb)[y == 0].mean()),
            "std_delta_positives": float((p - pb)[y == 1].std(ddof=1)),
            "kendall_tau_vs_zero_shot": float(stats.kendalltau(pb, p).statistic),
        }
    return out


def section6(y, runs, li, patients, max_subset=None, n_boot_ci=2000):
    """Is there a free ensemble win among the zero-shot-eligible models?"""
    elig = [m for m in ZERO_SHOT if m in runs and "val" in runs[m]]
    vy = runs[elig[0]]["val_y"][:, li]
    for m in elig:
        assert np.array_equal(runs[m]["val_y"][:, li], vy), f"val labels differ for {m}"
    TP = np.stack([runs[m]["test"][:, li] for m in elig])
    VP = np.stack([runs[m]["val"][:, li] for m in elig])
    TR = np.stack([stats.rankdata(r) / len(r) for r in TP])
    VR = np.stack([stats.rankdata(r) / len(r) for r in VP])

    results = []
    n = len(elig)
    for k in range(1, n + 1):
        if max_subset and k > max_subset:
            break
        for comb in itertools.combinations(range(n), k):
            c = list(comb)
            for rule, tp_, vp_ in (("mean_prob", TP[c].mean(0), VP[c].mean(0)),
                                   ("mean_rank", TR[c].mean(0), VR[c].mean(0))):
                thr = bootstrap_thresholds(vy, vp_)
                results.append({
                    "members": [elig[i] for i in c], "k": k, "rule": rule,
                    "val_auc": float(roc_auc_score(vy, vp_)),
                    "val_f1_at_tuned": f1_at(vy, vp_, thr)[0],
                    "test_auc": float(roc_auc_score(y, tp_)),
                    "test_f1_at_val_threshold": f1_at(y, tp_, thr)[0],
                    "test_oracle_f1": oracle_f1(y, tp_)[0],
                    "threshold": thr,
                })
    singles = [r for r in results if r["k"] == 1 and r["rule"] == "mean_prob"]
    best_single_test_auc = max(singles, key=lambda r: r["test_auc"])
    best_single_test_f1 = max(singles, key=lambda r: r["test_f1_at_val_threshold"])
    best_single_val_auc = max(singles, key=lambda r: r["val_auc"])
    ens = [r for r in results if r["k"] > 1]
    best_ens_test_auc = max(ens, key=lambda r: r["test_auc"])
    best_ens_test_f1 = max(ens, key=lambda r: r["test_f1_at_val_threshold"])
    val_sel_auc = max(ens, key=lambda r: r["val_auc"])
    val_sel_f1 = max(ens, key=lambda r: r["val_f1_at_tuned"])

    out = {
        "_question": "Does averaging zero-shot-eligible models clear the noise floor?",
        "_protocol_note": "Picking the subset by TEST score is selection on the "
                          "evaluation set and is reported only as an upper bound. "
                          "The honest number is the subset chosen on VALIDATION.",
        "eligible_models": elig,
        "n_subsets_scored": len(results),
        "best_single": {"by_test_auc": best_single_test_auc,
                        "by_test_f1": best_single_test_f1,
                        "by_val_auc": best_single_val_auc},
        "best_ensemble_selected_on_test_UPPER_BOUND": {
            "by_auc": best_ens_test_auc, "by_f1": best_ens_test_f1},
        "ensemble_selected_on_val_HONEST": {"by_val_auc": val_sel_auc,
                                            "by_val_f1": val_sel_f1},
        "all_models_mean_prob": next(
            r for r in results if r["k"] == n and r["rule"] == "mean_prob"),
        "all_models_mean_rank": next(
            r for r in results if r["k"] == n and r["rule"] == "mean_rank"),
        "top10_by_test_auc": sorted(results, key=lambda r: -r["test_auc"])[:10],
        "top10_by_test_f1": sorted(results, key=lambda r: -r["test_f1_at_val_threshold"])[:10],
        "noise_floors": {"auc": AUC_NOISE, "f1": F1_NOISE},
    }
    for name, r in (("val_selected_by_auc", val_sel_auc),
                    ("val_selected_by_f1", val_sel_f1),
                    ("test_selected_by_auc", best_ens_test_auc)):
        out.setdefault("verdicts", {})[name] = {
            "members": r["members"], "rule": r["rule"],
            "delta_auc_vs_best_single_test_auc": r["test_auc"] - best_single_test_auc["test_auc"],
            "delta_auc_vs_zero_shot": r["test_auc"] - float(roc_auc_score(y, runs[BASE]["test"][:, li])),
            "delta_f1_vs_best_single_test_f1": r["test_f1_at_val_threshold"] - best_single_test_f1["test_f1_at_val_threshold"],
            "clears_auc_noise_floor_vs_best_single": bool(
                r["test_auc"] - best_single_test_auc["test_auc"] > AUC_NOISE),
            "clears_f1_noise_floor_vs_best_single": bool(
                r["test_f1_at_val_threshold"] - best_single_test_f1["test_f1_at_val_threshold"] > F1_NOISE),
        }
    # clustered bootstrap on the honest pick vs the best single
    c = [elig.index(m) for m in val_sel_auc["members"]]
    ens_p = (TP[c].mean(0) if val_sel_auc["rule"] == "mean_prob" else TR[c].mean(0))
    base_p = runs[best_single_val_auc["members"][0]]["test"][:, li]
    out["patient_bootstrap_val_selected_auc_vs_best_single_by_val"] = patient_bootstrap(
        y, base_p, ens_p, patients, lambda yy, pp: roc_auc_score(yy, pp), n=n_boot_ci)
    return out


def section7(y, runs, li, patients):
    """How concentrated is the pooled model's advantage?"""
    pb = runs[BASE]["test"][:, li]
    cb = per_positive_auc_contrib(y, pb)
    auc_b = float(roc_auc_score(y, pb))
    npos = int(y.sum())
    pos = np.flatnonzero(y == 1)
    out = {"_question": "Is the pooled model's advantage concentrated in a few "
                        "cases or spread evenly?",
           "_identity": "AUC == mean over positives of c_i, where c_i is the "
                        "fraction of negatives that positive outranks. The AUC "
                        "gap therefore splits exactly across positives.",
           "zero_shot_auc": auc_b, "n_positives": npos,
           "n_negatives": int((y == 0).sum()), "per_seed": {}}
    curves = []
    for t in POOLED:
        if t not in runs:
            continue
        p = runs[t]["test"][:, li]
        c = per_positive_auc_contrib(y, p)
        auc_t = float(roc_auc_score(y, p))
        delta = c - cb
        order = np.argsort(-delta)
        cum = np.cumsum(delta[order]) / npos
        gap = auc_t - auc_b
        def k_for(frac):
            if gap <= 0:
                return None
            hit = np.flatnonzero(cum >= frac * gap)
            return int(hit[0] + 1) if len(hit) else None
        out["per_seed"][t] = {
            "pooled_auc": auc_t, "auc_gap": gap,
            "n_positives_improved": int((delta > 0).sum()),
            "n_positives_worsened": int((delta < 0).sum()),
            "n_positives_unchanged": int((delta == 0).sum()),
            "top1_share_of_gap": float(cum[0] / gap) if gap > 0 else None,
            "top5_share_of_gap": float(cum[4] / gap) if gap > 0 else None,
            "top10_share_of_gap": float(cum[9] / gap) if gap > 0 else None,
            "top20_share_of_gap": float(cum[19] / gap) if gap > 0 else None,
            "k_positives_for_50pct_of_gap": k_for(0.5),
            "k_positives_for_80pct_of_gap": k_for(0.8),
            "k_positives_for_100pct_of_gap": k_for(1.0),
            "gini_of_positive_gains": float(_gini(np.maximum(delta, 0))),
            "largest_gain_rows": [int(pos[i]) for i in order[:10]],
            "mean_delta_c": float(delta.mean()),
        }
        curves.append(cum / gap if gap > 0 else cum)
    # Is the concentrated gain set the SAME cases across seeds, or seed noise?
    tops = {}
    for t in list(out["per_seed"]):
        p = runs[t]["test"][:, li]
        delta = per_positive_auc_contrib(y, p) - cb
        order = np.argsort(-delta)
        tops[t] = {k: set(int(pos[i]) for i in order[:k]) for k in (10, 25)}
    for k in (10, 25):
        inter = set.intersection(*[tops[t][k] for t in tops])
        union = set.union(*[tops[t][k] for t in tops])
        out[f"stable_gain_set_top{k}"] = {
            "n_in_all_three_seeds": len(inter),
            "rows_in_all_three_seeds": sorted(inter),
            "n_in_union": len(union),
            "jaccard_intersection_over_union": float(len(inter) / max(len(union), 1)),
        }
    # headline against the reported 3-seed mean AUC 0.9463
    mean_auc = float(np.mean([out["per_seed"][t]["pooled_auc"] for t in out["per_seed"]]))
    out["pooled_mean_auc_across_seeds"] = mean_auc
    out["headline_gap_0.9104_to_0.9463"] = mean_auc - auc_b
    ks = [out["per_seed"][t]["k_positives_for_100pct_of_gap"] for t in out["per_seed"]]
    out["n_positive_images_that_account_for_the_whole_gap"] = {
        "per_seed": ks, "median": float(np.median([k for k in ks if k])),
        "interpretation": "the minimum number of positive images whose improved "
                          "ranking alone reproduces the entire AUC gap; the rest "
                          "of the positives net out to zero or negative",
    }
    # negative-side view
    out["negative_side"] = {}
    for t in list(out["per_seed"]):
        p = runs[t]["test"][:, li]
        neg = np.flatnonzero(y == 0)
        rb = stats.rankdata(pb)[neg]
        rt = stats.rankdata(p)[neg]
        out["negative_side"][t] = {
            "mean_prob_delta_negatives": float((p - pb)[neg].mean()),
            "spearman_negative_ranks": float(stats.spearmanr(rb, rt).statistic)}
    # patient concentration, counts only
    pat = np.array(patients)[pos]
    for t in list(out["per_seed"]):
        rows = out["per_seed"][t]["largest_gain_rows"]
        out["per_seed"][t]["n_distinct_patients_in_top10_gains"] = len(
            set(np.array(patients)[rows]))
    out["n_dr_positive_patients"] = len(set(pat))
    return out


def section8(y, runs, li, patients, doc):
    """Cross-links between the three sets the earlier sections produced, plus
    the label-efficiency curve in AUC/F1 terms for context."""
    out = {"_question": "How do the hard core, the zero-shot misses and the "
                        "concentrated-gain set relate, and what does the label "
                        "budget curve look like?"}
    s2 = doc.get("s2_which_cases")
    s4 = doc.get("s4_real_vs_synthesis")
    s7 = doc.get("s7_concentration")
    if s2 and s4 and s7:
        missed = set(s2["zero_shot_missed"]["rows"])
        core = set(s2["hard_core"]["all_models"]["never_detected_rows"])
        zs_core = set(s2["hard_core"]["zero_shot_eligible_only"]["never_detected_rows"])
        neither = set(s4["sets"]["neither"]["rows"])
        real_only = set(s4["sets"]["real_only"]["rows"])
        gain = set(s7.get("stable_gain_set_top25", {}).get("rows_in_all_three_seeds", []))
        out["set_sizes"] = {"zero_shot_missed": len(missed), "hard_core_all_models": len(core),
                            "hard_core_zero_shot_only": len(zs_core),
                            "fixed_by_neither": len(neither), "fixed_by_real_only": len(real_only),
                            "stable_top25_gain_set": len(gain)}
        out["overlaps"] = {
            "hard_core_subset_of_zero_shot_missed": core.issubset(missed),
            "hard_core_within_fixed_by_neither": len(core & neither),
            "stable_gain_set_in_zero_shot_missed": len(gain & missed),
            "stable_gain_set_in_hard_core": len(gain & core),
            "stable_gain_set_outside_zero_shot_missed": len(gain - missed),
            "real_only_fixes_in_stable_gain_set": len(real_only & gain),
        }
        out["interpretation_note"] = (
            "cases in the stable gain set but NOT in the zero-shot missed set are "
            "positives the zero-shot model already called correctly; the pooled "
            "model's AUC gain there is re-ranking headroom that never shows up in "
            "the false-negative count")
    # how far below do the never-detected cases actually rank?
    if s2:
        core = s2["hard_core"]["all_models"]["never_detected_rows"]
        pos = np.flatnonzero(y == 1)
        allm = sorted(runs)
        C = np.stack([per_positive_auc_contrib(y, runs[t]["test"][:, li]) for t in allm])
        idx = np.searchsorted(pos, core)
        out["hard_core_ranking_detail"] = {
            "n": len(core),
            "max_c_over_models_mean": float(C[:, idx].max(0).mean()),
            "max_c_over_models_min": float(C[:, idx].max(0).min()),
            "max_c_over_models_median": float(np.median(C[:, idx].max(0))),
            "mean_c_over_models_mean": float(C[:, idx].mean(0).mean()),
            "n_with_max_c_above_0.95": int((C[:, idx].max(0) > 0.95).sum()),
            "_reading": "c = fraction of the 573 negatives this positive outranks. "
                        "A hard-core case with max_c near 1 is ranked high by some "
                        "model but still under every threshold, i.e. a threshold "
                        "problem for that case; max_c near 0.5 means genuinely "
                        "unseen by every model.",
        }
    # label-efficiency curve
    curve = {}
    for b, ms in sorted(BUDGET_CURVE.items()):
        aucs, f1s = [], []
        for t in ms:
            if t not in runs or runs[t]["thr"] is None:
                continue
            p = runs[t]["test"][:, li]
            aucs.append(float(roc_auc_score(y, p)))
            f1s.append(f1_at(y, p, float(runs[t]["thr"][li]))[0])
        curve[str(b)] = {
            "n_seeds": len(aucs),
            "auc_per_seed": aucs, "auc_mean": float(np.mean(aucs)),
            "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
            "f1_per_seed": f1s, "f1_mean": float(np.mean(f1s)),
            "f1_sd": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else None,
        }
    out["label_efficiency_curve"] = curve
    synth = {}
    for t in SYNTH + ["balctl_s0", "balctl_s1"]:
        if t not in runs:
            continue
        p = runs[t]["test"][:, li]
        synth[t] = {"auc": float(roc_auc_score(y, p)),
                    "f1": f1_at(y, p, float(runs[t]["thr"][li]))[0],
                    "auc_minus_budget0_mean": float(roc_auc_score(y, p)) - curve["0"]["auc_mean"]}
    out["zero_budget_arms_vs_budget0_baseline"] = synth
    out["noise_floors"] = {"auc": AUC_NOISE, "f1": F1_NOISE}
    return out


def _gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    ap.add_argument("--labels-csv",
                    default=str(REPO / "data/finetune_mbrset_multilabel/test/labels.csv"))
    ap.add_argument("--out", default=str(REPO / "results/model_forensics.json"))
    ap.add_argument("--label-index", type=int, default=0,
                    help="0 = diabetic_retinopathy (default), 1 = macular_edema")
    ap.add_argument("--sections", default="all",
                    help="comma list from 1..7, or 'all'; section 8 (cross-links) runs whenever 2, 4 and 7 are all selected")
    ap.add_argument("--max-subset", type=int, default=None,
                    help="cap ensemble subset size (default: no cap)")
    ap.add_argument("--n-boot-ci", type=int, default=2000)
    args = ap.parse_args()

    li = args.label_index
    results_dir = Path(args.results_dir)
    y_full, files, patients = load_manifest(Path(args.labels_csv))
    y = y_full[:, li]

    runs, rejected = load_runs(results_dir, y_full)
    doc = {
        "_generated_by": "scripts/93_model_forensics.py",
        "label": LABEL_COLS[li],
        "n_test_images": int(len(y)),
        "n_positive": int(y.sum()),
        "n_patients": len(set(patients)),
        "n_positive_patients": len(set(np.array(patients)[y == 1])),
        "ordering_verification": {
            "labels_csv": os.path.relpath(args.labels_csv, REPO),
            "code_path": "BRSETMultiLabel reads labels.csv and uses .iloc[idx]; "
                         "test DataLoader shuffle=False; run_inference concatenates "
                         "batches in order => npz row i == csv data row i",
            "n_runs_accepted": len(runs),
            "n_runs_rejected": len(rejected),
            "rejected": rejected,
            "y_true_bitwise_identical_to_csv_for_all_accepted": True,
            "residual_caveat": "the value check alone cannot exclude a permutation "
                               "that only swaps rows with an identical (DR, ME) "
                               "label pair; the code path above is what excludes it",
            "also_identical": "data/finetune_joint/test/labels.csv (the path the "
                              "runs actually used) has byte-identical file/DR/ME "
                              "columns in the same order, differing only by an "
                              "extra `domain` column",
        },
        "caveats": [
            "Every arm, including the 'zero-shot' ones, tuned its decision "
            "threshold on the 725-image REAL mBRSET validation set. These arms "
            "are zero-shot in training data only, never in threshold selection.",
            "42 is a count of IMAGES. 159 DR-positive images come from 59 "
            "patients; image false negatives are not patient misses.",
            "The reported pooled headline 0.9463 / 0.8365 is the MEAN over the "
            "three joint seeds, not any single cached model; per-seed values are "
            "0.9377/0.8250, 0.9519/0.8449, 0.9492/0.8396.",
            "Noise floors used as the bar for calling anything an effect: "
            f"DR AUC {AUC_NOISE}, DR F1 {F1_NOISE}.",
            "No patient identifiers or filenames are written to this file; case "
            "references are row indices into the ordered 732-row test manifest.",
        ],
    }

    quality = None
    qf = results_dir / "mbrset_test_quality.csv"
    if qf.exists():
        qrows = {r["file"]: r for r in csv.DictReader(open(qf))}
        if all(f in qrows for f in files):
            quality = {k: np.array([float(qrows[f][k]) for f in files])
                       for k in ["sharpness", "brightness", "falloff"]}
            doc["quality_table"] = "results/mbrset_test_quality.csv joined by filename"

    doc["threshold_reproduction_check"] = verify_stored_thresholds(runs, li)

    want = args.sections
    sel = set(range(1, 8)) if want == "all" else {int(s) for s in want.split(",")}
    if 1 in sel:
        doc["s1_ranking_vs_calibration"] = section1(y, runs, li)
    if 2 in sel:
        doc["s2_which_cases"] = section2(y, runs, li, patients)
    if 3 in sel:
        doc["s3_score_distributions"] = section3(y, runs, li, patients)
    if 4 in sel:
        doc["s4_real_vs_synthesis"] = section4(y, runs, li, patients, y_full, quality)
    if 5 in sel:
        doc["s5_agreement"] = section5(y, runs, li)
    if 6 in sel:
        doc["s6_ensemble"] = section6(y, runs, li, patients,
                                      max_subset=args.max_subset,
                                      n_boot_ci=args.n_boot_ci)
    if 7 in sel:
        doc["s7_concentration"] = section7(y, runs, li, patients)

    if sel >= {2, 4, 7}:
        doc["s8_cross_links"] = section8(y, runs, li, patients, doc)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
    print(f"wrote {args.out}")
    print(json.dumps({k: v for k, v in doc.items() if k.startswith("_") or
                      not isinstance(v, dict)}, indent=2)[:1200])


if __name__ == "__main__":
    main()
