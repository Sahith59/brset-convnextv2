#!/usr/bin/env python
"""Data forensics for the BRSET -> mBRSET transfer gap.

Why this exists
---------------
Training on BRSET and testing on mBRSET gives DR AUC 0.9104. Adding 3,402 real
labelled mBRSET images raises it to 0.9463. Two obvious explanations have been
experimentally eliminated:

  - class balance: resampling BRSET to mBRSET's 23.4% positive rate closed only
    ~3% of the gap;
  - image appearance: a translator matching 13 appearance statistics to
    |z| <= 0.32 made downstream accuracy WORSE at every mixing ratio.

So the driver is upstream of pixels. This script measures the structural,
label-level and population-level differences between the two cohorts using the
ORIGINAL PhysioNet metadata, which carries far more than the binary labels the
training pipeline consumes:

  BRSET  labels_brset.csv  : DR_ICDR, DR_SDRG, quality, camera, patient_age,
                             diabetes, diabetes_time_y, insuline, comorbidities
  mBRSET labels_mbrset.csv : final_icdr, final_edema, final_quality,
                             final_artifacts, laterality, age, sex, dm_time,
                             insulin, 10 comorbidity fields, educational_level

Everything written to disk is an AGGREGATE. No patient images and no
per-patient rows are emitted, per the PhysioNet DUA.

Row-order contract
------------------
results/<task>/test_predictions.npz stores y_true/y_prob with no filenames.
scripts/30_train_strong_baseline.py builds the test loader with shuffle=False
over BRSETMultiLabel, which indexes pd.read_csv(split_dir/"labels.csv").iloc[idx].
Therefore npz row i == row i of data/finetune_joint/test/labels.csv. The script
asserts this by checking y_true against the CSV labels before using any
per-row join.

Usage
-----
  python scripts/92_data_forensics.py --out results/data_forensics.json
  python scripts/92_data_forensics.py --out results/data_forensics.json --image_stats

--image_stats decodes the 732 test images, so it must run on a compute node
(scripts/92_data_forensics.slurm, partition qCPU24), never on the login node.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/users/sthummala2/brset-convnextv2")
BRSET_CSV = Path(
    "/data/users4/nshaik3/Datasets/BRSET/physionet.org/files/"
    "brazilian-ophthalmological/1.0.1/labels_brset.csv"
)
MBRSET_CSV = Path(
    "/data/users4/nshaik3/Datasets/mBRSET/physionet.org/files/mbrset/1.0/labels_mbrset.csv"
)
BRSET_SPLITS = REPO / "data/finetune_multilabel"
MBRSET_SPLITS = REPO / "data/finetune_mbrset_multilabel"
JOINT_SPLITS = REPO / "data/finetune_joint"

ZEROSHOT = REPO / "results/lebase_b0_s0/test_predictions.npz"
POOLED = REPO / "results/brset_mbrset_joint/test_predictions.npz"

GRADES = [0, 1, 2, 3, 4]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def r(x, n=4):
    """Round, mapping non-finite to None so the result is JSON-clean."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(x) else round(x, n)


def numsum(s):
    s = pd.to_numeric(s, errors="coerce")
    return {
        "n": int(s.notna().sum()),
        "n_missing": int(s.isna().sum()),
        "mean": r(s.mean(), 2),
        "std": r(s.std(), 2),
        "median": r(s.median(), 2),
        "p25": r(s.quantile(0.25), 2),
        "p75": r(s.quantile(0.75), 2),
    }


def hist(series, grades=GRADES):
    """Counts and fractions over an ordinal grade column, NaN reported apart."""
    s = pd.to_numeric(series, errors="coerce")
    counts = {str(g): int((s == g).sum()) for g in grades}
    n = sum(counts.values())
    return {
        "counts": counts,
        "n_graded": n,
        "n_nan": int(s.isna().sum()),
        "frac": {k: r(v / n) if n else None for k, v in counts.items()},
    }


def tv_distance(p, q):
    """Total variation distance between two grade distributions (dicts)."""
    keys = sorted(set(p) | set(q))
    return 0.5 * sum(abs(float(p.get(k) or 0.0) - float(q.get(k) or 0.0)) for k in keys)


def auc(y, p, w=None):
    """ROC AUC. Rank based, so it accepts sample weights for reweighting."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, p, sample_weight=w))
    except ValueError:
        return None


def boot_auc_ci(y, p, w=None, n=2000, seed=0):
    """Percentile bootstrap CI for AUC, resampling rows."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        v = auc(y[idx], p[idx], None if w is None else np.asarray(w)[idx])
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    return r(np.percentile(vals, 2.5)), r(np.percentile(vals, 97.5))


def mcnemar(a_correct, b_correct):
    """Exact-ish McNemar on two paired binary correctness vectors."""
    from scipy import stats

    b = int(np.sum(a_correct & ~b_correct))  # a right, b wrong
    c = int(np.sum(~a_correct & b_correct))  # a wrong, b right
    if b + c == 0:
        return {"b_only": b, "c_only": c, "p": None}
    p = float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p": r(p, 5)}


# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------
def load_raw():
    # BRSET's header carries a UTF-8 BOM on image_id; utf-8-sig strips it.
    B = pd.read_csv(BRSET_CSV, encoding="utf-8-sig")
    M = pd.read_csv(MBRSET_CSV)
    B["file"] = B["image_id"].astype(str) + ".jpg"
    M["eye_id"] = M["patient"].astype(str) + "_" + M["laterality"].astype(str)
    return B, M


def load_split(d, name):
    return pd.read_csv(Path(d) / name / "labels.csv")


# ----------------------------------------------------------------------------
# 1. disease severity composition  (TOP HYPOTHESIS)
# ----------------------------------------------------------------------------
def severity(B, M, out):
    btr = load_split(BRSET_SPLITS, "train")
    mtr = load_split(MBRSET_SPLITS, "train")
    mte = load_split(MBRSET_SPLITS, "test")

    Bi = B.set_index("file")
    Mi = M.set_index("file")

    b_train = Bi.loc[Bi.index.intersection(btr["file"])]
    m_train = Mi.loc[Mi.index.intersection(mtr["file"])]
    m_test = Mi.loc[Mi.index.intersection(mte["file"])]

    res = {
        "_what": (
            "ICDR 0-4 severity histograms. BRSET grades live in DR_ICDR, mBRSET in "
            "final_icdr. The binary training label is (grade >= 1) in both cohorts."
        ),
        "brset_full": hist(B["DR_ICDR"]),
        "brset_train_split": hist(b_train["DR_ICDR"]),
        "mbrset_full": hist(M["final_icdr"]),
        "mbrset_train_split": hist(m_train["final_icdr"]),
        "mbrset_test_split": hist(m_test["final_icdr"]),
    }

    # Conditional on being positive: this is the comparison that matters, because
    # prevalence (the negative mass) has already been controlled experimentally.
    def cond_pos(s):
        s = pd.to_numeric(s, errors="coerce")
        return hist(s[s >= 1], grades=[1, 2, 3, 4])

    cp = {
        "brset_full": cond_pos(B["DR_ICDR"]),
        "brset_train_split": cond_pos(b_train["DR_ICDR"]),
        "mbrset_full": cond_pos(M["final_icdr"]),
        "mbrset_train_split": cond_pos(m_train["final_icdr"]),
        "mbrset_test_split": cond_pos(m_test["final_icdr"]),
    }
    res["conditional_on_positive"] = cp
    res["tv_distance_pos_brset_train_vs_mbrset_test"] = r(
        tv_distance(cp["brset_train_split"]["frac"], cp["mbrset_test_split"]["frac"])
    )
    res["tv_distance_pos_brset_train_vs_mbrset_train"] = r(
        tv_distance(cp["brset_train_split"]["frac"], cp["mbrset_train_split"]["frac"])
    )

    # Referable DR (ICDR >= 2) share of positives: the clinically actionable split.
    def ref_share(s):
        s = pd.to_numeric(s, errors="coerce")
        pos = s[s >= 1]
        return r((pos >= 2).mean()) if len(pos) else None

    res["referable_share_of_positives"] = {
        "brset_full": ref_share(B["DR_ICDR"]),
        "brset_train_split": ref_share(b_train["DR_ICDR"]),
        "mbrset_full": ref_share(M["final_icdr"]),
        "mbrset_test_split": ref_share(m_test["final_icdr"]),
    }
    # Proliferative (grade 4) share: the most visually unmistakable class.
    def prolif_share(s):
        s = pd.to_numeric(s, errors="coerce")
        pos = s[s >= 1]
        return r((pos == 4).mean()) if len(pos) else None

    res["proliferative_share_of_positives"] = {
        "brset_train_split": prolif_share(b_train["DR_ICDR"]),
        "mbrset_test_split": prolif_share(m_test["final_icdr"]),
    }

    # chi-square on the 4-bin positive histogram
    try:
        from scipy import stats

        obs = np.array(
            [
                [cp["brset_train_split"]["counts"][str(g)] for g in [1, 2, 3, 4]],
                [cp["mbrset_test_split"]["counts"][str(g)] for g in [1, 2, 3, 4]],
            ]
        )
        chi2, p, dof, _ = stats.chi2_contingency(obs)
        res["chi2_pos_grade_brset_train_vs_mbrset_test"] = {
            "chi2": r(chi2, 2),
            "dof": int(dof),
            "p": r(p, 8),
        }
    except Exception as e:  # pragma: no cover
        res["chi2_pos_grade_brset_train_vs_mbrset_test"] = {"error": str(e)}

    out["1_severity_composition"] = res
    return m_test


# ----------------------------------------------------------------------------
# 2. label definition / grader effects
# ----------------------------------------------------------------------------
def label_definition(B, M, out):
    res = {
        "_what": (
            "Is the binary target the same construct in both cohorts? BRSET ships an "
            "explicit diabetic_retinopathy column AND two severity scales (DR_ICDR, "
            "DR_SDRG). mBRSET ships only final_icdr, so the pipeline derives the "
            "binary label as (final_icdr >= 1). This section audits that derivation "
            "and the asymmetric preprocessing."
        )
    }

    # Is BRSET's own binary column == (DR_ICDR >= 1)?
    ct = pd.crosstab(B["diabetic_retinopathy"], B["DR_ICDR"] >= 1)
    agree = int(np.trace(ct.values)) if ct.shape == (2, 2) else None
    res["brset_binary_vs_icdr_ge1"] = {
        "crosstab": ct.to_dict(),
        "n": int(len(B)),
        "agreement": r(agree / len(B)) if agree is not None else None,
        "note": "derivation rule used for mBRSET is sound on BRSET's own data",
    }

    # Two grading scales inside BRSET disagree with each other.
    ct2 = pd.crosstab(B["DR_ICDR"], B["DR_SDRG"])
    both = B[["DR_ICDR", "DR_SDRG"]].dropna()
    exact = float((both["DR_ICDR"] == both["DR_SDRG"]).mean())
    posmask = (both["DR_ICDR"] >= 1) | (both["DR_SDRG"] >= 1)
    res["brset_icdr_vs_sdrg"] = {
        "crosstab": {str(k): v for k, v in ct2.to_dict().items()},
        "exact_grade_agreement_all": r(exact),
        "exact_grade_agreement_among_positives": r(
            float((both.loc[posmask, "DR_ICDR"] == both.loc[posmask, "DR_SDRG"]).mean())
        ),
        "binary_agreement": r(
            float(((both["DR_ICDR"] >= 1) == (both["DR_SDRG"] >= 1)).mean())
        ),
        "note": (
            "ICDR and SDRG are two protocols applied to the SAME BRSET images. They "
            "agree on presence/absence but not on grade, which bounds how reproducible "
            "an ordinal severity target is even within one cohort."
        ),
    }

    # Asymmetric quality filtering between the two pipelines.
    btr = load_split(BRSET_SPLITS, "train")
    Bi = B.set_index("file")
    b_train = Bi.loc[Bi.index.intersection(btr["file"])]
    res["quality_filter_asymmetry"] = {
        "_what": (
            "scripts/13_prepare_splits_mbrset_multilabel.py keeps only "
            'final_quality == "yes"; scripts/06_prepare_splits_multilabel.py applies NO '
            "quality filter to BRSET. So the source domain contains ungradable images "
            "and the target domain does not."
        ),
        "brset_quality_counts_full": B["quality"].value_counts(dropna=False).to_dict(),
        "brset_quality_counts_train_split": b_train["quality"]
        .value_counts(dropna=False)
        .to_dict(),
        "brset_train_frac_inadequate": r(
            float((b_train["quality"] == "Inadequate").mean())
        ),
        "mbrset_final_quality_counts_full": M["final_quality"]
        .value_counts(dropna=False)
        .to_dict(),
        "mbrset_frac_dropped_by_quality_filter": r(
            float((M["final_quality"] != "yes").mean())
        ),
        "mbrset_final_artifacts_counts_full": M["final_artifacts"]
        .value_counts(dropna=False)
        .to_dict(),
        "brset_artifacts_counts_full": B["artifacts"].value_counts(dropna=False).to_dict(),
    }

    # BRSET per-image quality sub-flags vs mBRSET's single collapsed flag.
    res["quality_vocabulary"] = {
        "brset_fields": ["focus", "Illuminaton", "image_field", "artifacts", "quality"],
        "mbrset_fields": ["final_quality", "final_artifacts"],
        "note": (
            "BRSET grades four separate image-quality axes plus an overall verdict; "
            "mBRSET ships one binary gradability flag and one artifact flag. The "
            "quality constructs are not interchangeable."
        ),
    }

    # Macular edema definition.
    res["macular_edema"] = {
        "brset_counts": B["macular_edema"].value_counts(dropna=False).to_dict(),
        "mbrset_final_edema_counts": M["final_edema"].value_counts(dropna=False).to_dict(),
        "brset_me_by_icdr": pd.crosstab(B["macular_edema"], B["DR_ICDR"]).to_dict(),
        "brset_me_prevalence": r(float(B["macular_edema"].mean())),
        "mbrset_me_prevalence": r(float((M["final_edema"] == "yes").mean())),
        "note": (
            "BRSET also ships a 3-level `macula` field; mBRSET does not. BRSET's "
            "macular_edema is not conditioned on diabetes, mBRSET's final_edema is "
            "graded in an all-diabetic cohort."
        ),
    }

    # Unlabelled mass.
    res["unlabelled"] = {
        "brset_icdr_nan": int(B["DR_ICDR"].isna().sum()),
        "mbrset_icdr_nan": int(M["final_icdr"].isna().sum()),
        "mbrset_icdr_nan_frac": r(float(M["final_icdr"].isna().mean())),
        "note": (
            "mBRSET leaves a non-trivial slice ungraded and those rows are dropped; "
            "BRSET grades every image. Dropping is not random -- see section 5 for "
            "how it interacts with the two-photos-per-eye design."
        ),
    }

    out["2_label_definition"] = res


# ----------------------------------------------------------------------------
# 3. patient population
# ----------------------------------------------------------------------------
def population(B, M, out):
    res = {
        "_what": (
            "BRSET is a hospital ophthalmology cohort (Sao Paulo), mBRSET a handheld "
            "community-screening cohort in an underserved population. The decisive "
            "number here is what fraction of each cohort even HAS diabetes."
        )
    }

    res["diabetes_status"] = {
        "brset_diabetes_counts": B["diabetes"].value_counts(dropna=False).to_dict(),
        "brset_frac_diabetic": r(float((B["diabetes"] == "yes").mean())),
        "mbrset_frac_diabetic": 1.0,
        "mbrset_note": (
            "mBRSET is a diabetic-screening dataset by construction: every row carries "
            "diabetes-specific fields (dm_time, insulin, oraltreatment_dm)."
        ),
        "brset_icdr_ge1_by_diabetes": pd.crosstab(
            B["diabetes"].fillna("MISSING"), B["DR_ICDR"] >= 1
        ).to_dict(),
    }
    # How many of BRSET's positives come from patients NOT recorded as diabetic?
    pos = B[B["DR_ICDR"] >= 1]
    res["diabetes_status"]["brset_positives_total"] = int(len(pos))
    res["diabetes_status"]["brset_positives_from_nondiabetic_record"] = int(
        (pos["diabetes"] != "yes").sum()
    )
    res["diabetes_status"]["brset_positives_from_nondiabetic_frac"] = r(
        float((pos["diabetes"] != "yes").mean())
    )
    # The negative class: this is the population mismatch that matters.
    neg = B[B["DR_ICDR"] == 0]
    res["negative_class_composition"] = {
        "_what": (
            "A BRSET-trained model's notion of 'negative' is mostly a NON-DIABETIC eye. "
            "Every mBRSET negative is a diabetic eye graded ICDR 0 -- a harder negative, "
            "and one the source domain barely contains."
        ),
        "brset_negatives_total": int(len(neg)),
        "brset_negatives_diabetic": int((neg["diabetes"] == "yes").sum()),
        "brset_negatives_diabetic_frac": r(float((neg["diabetes"] == "yes").mean())),
        "mbrset_negatives_diabetic_frac": 1.0,
    }

    # Apples-to-apples: restrict BRSET to its DIABETIC patients, which is the only
    # subpopulation comparable to mBRSET. If BRSET's diabetics still show a far more
    # advanced severity mix, the severity gap is referral bias (hospital vs community
    # screening), not merely "BRSET contains non-diabetics".
    Bd = B[B["diabetes"] == "yes"]
    bd = pd.to_numeric(Bd["DR_ICDR"], errors="coerce")
    mg = pd.to_numeric(M["final_icdr"], errors="coerce")
    res["diabetic_subset_comparison"] = {
        "_what": (
            "BRSET restricted to diabetes == 'yes' vs all of mBRSET. The comparable "
            "denominators for a screening-vs-hospital contrast."
        ),
        "brset_diabetic_n_images": int(len(Bd)),
        "brset_diabetic_dr_prevalence": r(float((bd >= 1).mean())),
        "mbrset_dr_prevalence": r(float((mg >= 1).mean())),
        "brset_diabetic_pos_grade_frac": hist(bd[bd >= 1], grades=[1, 2, 3, 4])["frac"],
        "mbrset_pos_grade_frac": hist(mg[mg >= 1], grades=[1, 2, 3, 4])["frac"],
        "brset_diabetic_prolif_share_of_pos": r(
            float((bd[bd >= 1] == 4).mean()) if (bd >= 1).sum() else None
        ),
        "mbrset_prolif_share_of_pos": r(
            float((mg[mg >= 1] == 4).mean()) if (mg >= 1).sum() else None
        ),
        "brset_diabetic_mild_share_of_pos": r(
            float((bd[bd >= 1] == 1).mean()) if (bd >= 1).sum() else None
        ),
        "mbrset_mild_share_of_pos": r(
            float((mg[mg >= 1] == 1).mean()) if (mg >= 1).sum() else None
        ),
        "interpretation_hook": (
            "A persisting severity shift inside the diabetic subset separates referral "
            "bias from cohort composition; a vanishing one would mean the shift is "
            "entirely explained by BRSET's non-diabetic mass."
        ),
    }

    res["age"] = {"brset": numsum(B["patient_age"]), "mbrset": numsum(M["age"])}
    res["diabetes_duration_years"] = {
        "brset_diabetes_time_y": numsum(B["diabetes_time_y"]),
        "mbrset_dm_time": numsum(M["dm_time"]),
        "note": (
            "BRSET records duration for only the diabetic subset, so the two summaries "
            "are not drawn from comparable denominators."
        ),
    }
    res["sex"] = {
        "brset_patient_sex_counts": B["patient_sex"].value_counts(dropna=False).to_dict(),
        "mbrset_sex_counts": M["sex"].value_counts(dropna=False).to_dict(),
        "note": (
            "Encodings differ (BRSET 1/2, mBRSET 0/1) and the codebooks are not in the "
            "CSVs, so the mapping to male/female is NOT established here. Reported as "
            "raw codes only."
        ),
    }
    res["insulin"] = {
        "brset_insuline_counts": B["insuline"].value_counts(dropna=False).to_dict(),
        "mbrset_insulin_counts": M["insulin"].value_counts(dropna=False).to_dict(),
        "brset_insulin_missing_frac": r(float(B["insuline"].isna().mean())),
    }
    res["brset_comorbidities_freetext_top"] = (
        B["comorbidities"].value_counts(dropna=False).head(12).to_dict()
    )
    res["mbrset_comorbidity_burden"] = {
        c: {
            "frac_positive": r(float(pd.to_numeric(M[c], errors="coerce").mean())),
            "n_missing": int(M[c].isna().sum()),
        }
        for c in [
            "systemic_hypertension",
            "oraltreatment_dm",
            "insulin",
            "alcohol_consumption",
            "smoking",
            "obesity",
            "vascular_disease",
            "acute_myocardial_infarction",
            "nephropathy",
            "neuropathy",
            "diabetic_foot",
        ]
    }
    res["mbrset_socioeconomic"] = {
        "insurance_counts": M["insurance"].value_counts(dropna=False).to_dict(),
        "educational_level_counts": M["educational_level"]
        .value_counts(dropna=False)
        .to_dict(),
        "note": (
            "BRSET ships no socioeconomic fields at all, so this axis cannot be matched "
            "even descriptively."
        ),
    }
    res["camera"] = {
        "brset_camera_counts": B["camera"].value_counts(dropna=False).to_dict(),
        "mbrset_camera": "handheld (single device family; no camera column in the CSV)",
    }
    out["3_population"] = res


# ----------------------------------------------------------------------------
# 4. which images does the model miss?
# ----------------------------------------------------------------------------
def misses(M, out, m_test_meta):
    res = {
        "_what": (
            "Join the 732 fixed test rows to their ICDR grade and ask where the "
            "zero-shot model's errors sit on the severity axis, and whether pooling "
            "fixes those same images."
        )
    }

    te = load_split(JOINT_SPLITS, "test")
    zs = np.load(ZEROSHOT)
    po = np.load(POOLED)

    # ---- enforce the row-order contract before any per-row join ----
    for name, d in [("zeroshot", zs), ("pooled", po)]:
        ok = np.array_equal(
            d["y_true"][:, 0], te["diabetic_retinopathy"].to_numpy(dtype="float32")
        ) and np.array_equal(
            d["y_true"][:, 1], te["macular_edema"].to_numpy(dtype="float32")
        )
        if not ok:
            raise RuntimeError(
                f"{name} npz y_true does not match labels.csv order; the per-row join "
                "would be invalid. Refusing to continue."
            )
    res["row_order_contract"] = {
        "verified": True,
        "n_rows": int(len(te)),
        "how": (
            "y_true in both npz files matches data/finetune_joint/test/labels.csv "
            "row-for-row on both label columns; the trainer builds the test loader with "
            "shuffle=False over .iloc, so npz row i is CSV row i."
        ),
    }

    Mi = M.set_index("file")
    meta = Mi.loc[te["file"].to_numpy()].reset_index()
    grade = pd.to_numeric(meta["final_icdr"], errors="coerce").to_numpy()

    y = zs["y_true"][:, 0].astype(int)
    p_zs = zs["y_prob"][:, 0]
    p_po = po["y_prob"][:, 0]
    thr_zs = float(zs["thresholds"][0])
    thr_po = float(po["thresholds"][0])

    pred_zs = p_zs >= thr_zs
    pred_po = p_po >= thr_po
    corr_zs = pred_zs == y.astype(bool)
    corr_po = pred_po == y.astype(bool)

    res["headline"] = {
        "n_test": int(len(y)),
        "n_positive": int(y.sum()),
        "zeroshot": {
            "auc": r(auc(y, p_zs)),
            "threshold": r(thr_zs),
            "missed_positives": int(((y == 1) & ~pred_zs).sum()),
            "false_positives": int(((y == 0) & pred_zs).sum()),
        },
        "pooled": {
            "auc": r(auc(y, p_po)),
            "threshold": r(thr_po),
            "missed_positives": int(((y == 1) & ~pred_po).sum()),
            "false_positives": int(((y == 0) & pred_po).sum()),
        },
        "auc_gap": r(auc(y, p_po) - auc(y, p_zs)),
    }

    # ---- per-grade behaviour ----
    per_grade = {}
    for g in GRADES:
        m = grade == g
        if m.sum() == 0:
            continue
        d = {
            "n": int(m.sum()),
            "is_positive_class": bool(g >= 1),
            "zeroshot_mean_prob": r(float(p_zs[m].mean())),
            "pooled_mean_prob": r(float(p_po[m].mean())),
        }
        if g >= 1:
            d["zeroshot_recall"] = r(float(pred_zs[m].mean()))
            d["pooled_recall"] = r(float(pred_po[m].mean()))
            d["zeroshot_missed"] = int((~pred_zs[m]).sum())
            d["pooled_missed"] = int((~pred_po[m]).sum())
        else:
            d["zeroshot_specificity"] = r(float((~pred_zs[m]).mean()))
            d["pooled_specificity"] = r(float((~pred_po[m]).mean()))
        per_grade[str(g)] = d
    res["per_grade"] = per_grade

    # Where do the misses live on the severity axis?
    miss_zs = (y == 1) & ~pred_zs
    miss_po = (y == 1) & ~pred_po
    res["miss_severity_profile"] = {
        "zeroshot_missed_grade_counts": {
            str(g): int(((grade == g) & miss_zs).sum()) for g in [1, 2, 3, 4]
        },
        "pooled_missed_grade_counts": {
            str(g): int(((grade == g) & miss_po).sum()) for g in [1, 2, 3, 4]
        },
        "positive_grade_counts": {
            str(g): int(((grade == g) & (y == 1)).sum()) for g in [1, 2, 3, 4]
        },
        "zeroshot_frac_of_misses_that_are_mild_g1": r(
            float(((grade == 1) & miss_zs).sum() / max(miss_zs.sum(), 1))
        ),
        "base_frac_of_positives_that_are_mild_g1": r(
            float(((grade == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
        ),
    }

    # Does pooling fix the SAME images the zero-shot model missed?
    res["paired_agreement"] = {
        "both_correct": int((corr_zs & corr_po).sum()),
        "both_wrong": int((~corr_zs & ~corr_po).sum()),
        "zeroshot_only_correct": int((corr_zs & ~corr_po).sum()),
        "pooled_only_correct": int((~corr_zs & corr_po).sum()),
        "mcnemar_all": mcnemar(corr_zs, corr_po),
    }
    pm = y == 1
    res["paired_agreement_positives_only"] = {
        "zeroshot_missed": int(miss_zs.sum()),
        "pooled_missed": int(miss_po.sum()),
        "missed_by_both": int((miss_zs & miss_po).sum()),
        "rescued_by_pooling": int((miss_zs & ~miss_po).sum()),
        "broken_by_pooling": int((~miss_zs & miss_po).sum()),
        "rescued_grade_counts": {
            str(g): int(((grade == g) & miss_zs & ~miss_po).sum()) for g in [1, 2, 3, 4]
        },
        "missed_by_both_grade_counts": {
            str(g): int(((grade == g) & miss_zs & miss_po).sum()) for g in [1, 2, 3, 4]
        },
        "mcnemar_positives": mcnemar(corr_zs[pm], corr_po[pm]),
    }

    # ---- THE decisive test: does severity composition account for the AUC gap? ----
    # Importance-reweight the test positives so their ICDR mix matches BRSET's
    # training positives, leaving negatives at weight 1. If the gap is a severity
    # story, the zero-shot AUC should move most of the 0.036 under this reweight.
    B, _ = load_raw()
    btr = load_split(BRSET_SPLITS, "train")
    Bi = B.set_index("file")
    b_train = Bi.loc[Bi.index.intersection(btr["file"])]
    bg = pd.to_numeric(b_train["DR_ICDR"], errors="coerce")
    bg = bg[bg >= 1]
    src_frac = {g: float((bg == g).mean()) for g in [1, 2, 3, 4]}
    tgt_frac = {
        g: float(((grade == g) & (y == 1)).sum() / max((y == 1).sum(), 1))
        for g in [1, 2, 3, 4]
    }
    w = np.ones(len(y), dtype=float)
    for g in [1, 2, 3, 4]:
        m = (grade == g) & (y == 1)
        if m.sum() and tgt_frac[g] > 0:
            w[m] = src_frac[g] / tgt_frac[g]
    # keep total positive weight equal to the native positive count
    posm = y == 1
    if w[posm].sum() > 0:
        w[posm] *= posm.sum() / w[posm].sum()

    a_nat_zs, a_nat_po = auc(y, p_zs), auc(y, p_po)
    a_rw_zs, a_rw_po = auc(y, p_zs, w), auc(y, p_po, w)
    lo, hi = boot_auc_ci(y, p_zs)
    res["severity_reweighted_auc"] = {
        "_what": (
            "Reweight test positives to BRSET-train's conditional ICDR mix (negatives "
            "unweighted, total positive weight preserved). This asks directly: if "
            "mBRSET's positives were as advanced as BRSET's, how much of the 0.036 AUC "
            "gap disappears?"
        ),
        "brset_train_pos_grade_frac": {str(k): r(v) for k, v in src_frac.items()},
        "mbrset_test_pos_grade_frac": {str(k): r(v) for k, v in tgt_frac.items()},
        "weights_applied_to_positives": {
            str(g): r(src_frac[g] / tgt_frac[g]) if tgt_frac[g] else None
            for g in [1, 2, 3, 4]
        },
        "zeroshot_auc_native": r(a_nat_zs),
        "zeroshot_auc_native_ci95": [lo, hi],
        "zeroshot_auc_reweighted": r(a_rw_zs),
        "zeroshot_auc_change": r(a_rw_zs - a_nat_zs),
        "pooled_auc_native": r(a_nat_po),
        "pooled_auc_reweighted": r(a_rw_po),
        "pooled_auc_change": r(a_rw_po - a_nat_po),
        "native_gap": r(a_nat_po - a_nat_zs),
        "reweighted_gap": r(a_rw_po - a_rw_zs),
        "frac_of_gap_removed_by_severity_matching": r(
            1.0 - (a_rw_po - a_rw_zs) / (a_nat_po - a_nat_zs)
        )
        if (a_nat_po - a_nat_zs)
        else None,
    }

    # Grade-stratified AUC: positives of one grade vs all negatives.
    strat = {}
    for g in [1, 2, 3, 4]:
        sel = (y == 0) | ((grade == g) & (y == 1))
        if ((grade == g) & (y == 1)).sum() < 5:
            continue
        strat[str(g)] = {
            "n_pos": int(((grade == g) & (y == 1)).sum()),
            "zeroshot_auc": r(auc(y[sel], p_zs[sel])),
            "pooled_auc": r(auc(y[sel], p_po[sel])),
            "gap": r(
                (auc(y[sel], p_po[sel]) or 0) - (auc(y[sel], p_zs[sel]) or 0)
            ),
        }
    res["grade_stratified_auc_vs_all_negatives"] = {
        "_what": (
            "AUC with only grade-g positives retained. Localises WHERE on the severity "
            "axis pooling buys its improvement."
        ),
        "by_grade": strat,
    }

    # Referable-only task (ICDR >= 2 positive, grade 1 dropped): if the gap
    # collapses here, mild disease is the whole story.
    sel = (grade == 0) | (grade >= 2)
    y_ref = (grade[sel] >= 2).astype(int)
    res["referable_only_task"] = {
        "_what": "Drop grade-1 positives entirely; score ICDR>=2 vs ICDR 0.",
        "n": int(sel.sum()),
        "n_pos": int(y_ref.sum()),
        "zeroshot_auc": r(auc(y_ref, p_zs[sel])),
        "pooled_auc": r(auc(y_ref, p_po[sel])),
        "gap": r((auc(y_ref, p_po[sel]) or 0) - (auc(y_ref, p_zs[sel]) or 0)),
    }

    out["4_misses"] = res
    return meta, grade, y, p_zs, p_po, pred_zs, pred_po


# ----------------------------------------------------------------------------
# 5. structural differences
# ----------------------------------------------------------------------------
def structure(B, M, out, meta, grade, y, p_zs, p_po):
    res = {}

    res["images_per_patient"] = {
        "_what": "Effective sample size and within-patient correlation.",
        "brset": {
            "n_images": int(len(B)),
            "n_patients": int(B["patient_id"].nunique()),
            "images_per_patient_mean": r(len(B) / B["patient_id"].nunique(), 3),
            "distribution": B.groupby("patient_id")
            .size()
            .value_counts()
            .sort_index()
            .to_dict(),
        },
        "mbrset": {
            "n_images": int(len(M)),
            "n_patients": int(M["patient"].nunique()),
            "images_per_patient_mean": r(len(M) / M["patient"].nunique(), 3),
            "distribution": M.groupby("patient").size().value_counts().sort_index().to_dict(),
        },
        "note": (
            "mBRSET is a fixed 4-images-per-patient protocol (2 per eye); BRSET is "
            "~2 (1 per eye). So the same image count buys half the patients, and the "
            "732-image test set is only ~183 patients."
        ),
    }

    res["laterality"] = {
        "brset_exam_eye_counts": B["exam_eye"].value_counts(dropna=False).to_dict(),
        "mbrset_laterality_counts": M["laterality"].value_counts(dropna=False).to_dict(),
        "brset_images_per_patient_eye": B.groupby(["patient_id", "exam_eye"])
        .size()
        .value_counts()
        .sort_index()
        .to_dict(),
        "mbrset_images_per_patient_eye": M.groupby(["patient", "laterality"])
        .size()
        .value_counts()
        .sort_index()
        .to_dict(),
    }

    # ---- repeat-photo grading consistency: a direct label-noise estimate ----
    gm = M.copy()
    gm["g"] = pd.to_numeric(gm["final_icdr"], errors="coerce")
    gm["bin"] = (gm["g"] >= 1).astype("float")
    eye = gm.dropna(subset=["g"]).groupby("eye_id")
    multi = eye.filter(lambda d: len(d) >= 2).groupby("eye_id")
    n_eyes_multi = multi.ngroups
    disc_grade = int((multi["g"].nunique() > 1).sum())
    disc_bin = int((multi["bin"].nunique() > 1).sum())
    res["intra_eye_grading_consistency"] = {
        "_what": (
            "mBRSET photographs each eye TWICE and grades each photo. Disagreement "
            "between two photos of the same eye is irreducible label noise, and it is "
            "measurable in mBRSET but NOT in BRSET (BRSET has ~1 photo per eye)."
        ),
        "mbrset_eyes_with_2plus_graded_photos": int(n_eyes_multi),
        "mbrset_eyes_discordant_on_grade": disc_grade,
        "mbrset_frac_eyes_discordant_on_grade": r(disc_grade / max(n_eyes_multi, 1)),
        "mbrset_eyes_discordant_on_binary_dr": disc_bin,
        "mbrset_frac_eyes_discordant_on_binary_dr": r(disc_bin / max(n_eyes_multi, 1)),
        "brset_measurable": False,
        "brset_reason": (
            "BRSET contributes about one photo per eye, so no repeat-photo agreement "
            "can be computed; its DR_ICDR vs DR_SDRG disagreement (section 2) is the "
            "nearest available analogue."
        ),
    }

    # Same measurement, restricted to the test split, plus the ceiling it implies.
    te = load_split(JOINT_SPLITS, "test")
    tmeta = M.set_index("file").loc[te["file"].to_numpy()].reset_index()
    tmeta["bin"] = (pd.to_numeric(tmeta["final_icdr"], errors="coerce") >= 1).astype(int)
    eg = tmeta.groupby("eye_id")["bin"]
    disc_eyes = eg.nunique() > 1
    disc_eye_ids = set(disc_eyes[disc_eyes].index)
    in_disc = tmeta["eye_id"].isin(disc_eye_ids).to_numpy()
    res["test_split_label_ceiling"] = {
        "_what": (
            "Test images whose sibling photo of the SAME eye carries the opposite "
            "binary DR label. Any model that is consistent at the eye level must be "
            "wrong on one of each such pair, so these images cap achievable accuracy."
        ),
        "n_test_images": int(len(tmeta)),
        "n_test_eyes": int(tmeta["eye_id"].nunique()),
        "n_test_patients": int(tmeta["patient"].nunique()),
        "n_eyes_discordant_binary": int(len(disc_eye_ids)),
        "n_images_in_discordant_eyes": int(in_disc.sum()),
        "frac_images_in_discordant_eyes": r(float(in_disc.mean())),
        "zeroshot_auc_all": r(auc(y, p_zs)),
        "zeroshot_auc_excluding_discordant_eyes": r(auc(y[~in_disc], p_zs[~in_disc])),
        "pooled_auc_all": r(auc(y, p_po)),
        "pooled_auc_excluding_discordant_eyes": r(auc(y[~in_disc], p_po[~in_disc])),
        "gap_all": r((auc(y, p_po) or 0) - (auc(y, p_zs) or 0)),
        "gap_excluding_discordant_eyes": r(
            (auc(y[~in_disc], p_po[~in_disc]) or 0)
            - (auc(y[~in_disc], p_zs[~in_disc]) or 0)
        ),
    }

    # ---- split integrity: patient-wise or image-wise? any leak? ----
    def patients_of(split):
        f = load_split(MBRSET_SPLITS, split)["file"].to_numpy()
        return set(M.set_index("file").loc[f]["patient"].unique())

    ptr, pva, pte = patients_of("train"), patients_of("val"), patients_of("test")
    jtr = load_split(JOINT_SPLITS, "train")
    jte = load_split(JOINT_SPLITS, "test")
    res["split_integrity"] = {
        "_what": "Is the split patient-wise, and is any patient or file shared?",
        "mbrset_patients": {
            "train": len(ptr),
            "val": len(pva),
            "test": len(pte),
            "train_test_overlap": len(ptr & pte),
            "train_val_overlap": len(ptr & pva),
            "val_test_overlap": len(pva & pte),
        },
        "mbrset_split_is_patient_wise": len(ptr & pte) == 0
        and len(ptr & pva) == 0
        and len(pva & pte) == 0,
        "joint_train_test_file_overlap": int(
            len(set(jtr["file"]) & set(jte["file"]))
        ),
        "joint_train_composition": {
            "n_total": int(len(jtr)),
            "by_domain": jtr["domain"].value_counts().to_dict()
            if "domain" in jtr.columns
            else None,
        },
        "brset_split_is_patient_wise_per_prep_script": True,
        "brset_prep_note": (
            "scripts/06_prepare_splits_multilabel.py groups by patient_id before "
            "splitting; scripts/13_* groups by patient. Verified here for mBRSET "
            "against the raw metadata."
        ),
    }

    # Positive-rate bookkeeping, for the record.
    res["prevalence"] = {
        "brset_train_dr_rate": r(
            float(load_split(BRSET_SPLITS, "train")["diabetic_retinopathy"].mean())
        ),
        "mbrset_train_dr_rate": r(
            float(load_split(MBRSET_SPLITS, "train")["diabetic_retinopathy"].mean())
        ),
        "mbrset_test_dr_rate": r(float(te["diabetic_retinopathy"].mean())),
        "note": "already controlled experimentally; closed only ~3% of the gap",
    }

    # Field of view / geometry is an image property; record what metadata says.
    res["field_of_view_metadata"] = {
        "brset_image_field_counts": B["image_field"].value_counts(dropna=False).to_dict(),
        "mbrset_equivalent_field_column": None,
        "note": (
            "BRSET grades whether the field is adequate (image_field); mBRSET ships no "
            "field-of-view column, so FOV cannot be compared from metadata. Measured "
            "from pixels only if --image_stats is passed."
        ),
    }

    out["5_structure"] = res
    return in_disc


# ----------------------------------------------------------------------------
# 7. seed-averaged version of the decisive tests
# ----------------------------------------------------------------------------
# The single-seed pair (lebase_b0_s0 vs brset_mbrset_joint) gives a 0.0273 AUC
# gap, but the figure quoted in the project notes, 0.9104 -> 0.9463, pairs the
# BEST zero-shot seed against the MEAN of three pooled seeds. Measured
# consistently, zero-shot is 0.9006 +- (3 seeds) and pooled is 0.9463 (3 seeds),
# so the real gap is 0.0456. Every decomposition below is therefore repeated
# across all 3x3 seed pairs; the project's own noise floor is 0.0199 DR F1, so a
# single-seed decomposition cannot be trusted on its own.
ZS_RUNS = ["lebase_b0_s0", "lebase_b0_s1", "lebase_b0_s2"]
PO_RUNS = ["brset_mbrset_joint", "joint_seed1", "joint_seed2"]


def seed_averaged(B, M, out):
    te = load_split(JOINT_SPLITS, "test")
    Mi = M.set_index("file")
    meta = Mi.loc[te["file"].to_numpy()].reset_index()
    grade = pd.to_numeric(meta["final_icdr"], errors="coerce").to_numpy()
    y = te["diabetic_retinopathy"].to_numpy(dtype=int)

    # discordant-eye mask (sibling photo of the same eye carries the other label)
    tm = meta.copy()
    tm["bin"] = (grade >= 1).astype(int)
    eg = tm.groupby("eye_id")["bin"].nunique() > 1
    disc = tm["eye_id"].isin(set(eg[eg].index)).to_numpy()

    # BRSET-train positive severity mix -> importance weights
    btr = load_split(BRSET_SPLITS, "train")
    Bi = B.set_index("file")
    bg = pd.to_numeric(
        Bi.loc[Bi.index.intersection(btr["file"])]["DR_ICDR"], errors="coerce"
    )
    bg = bg[bg >= 1]
    src = {g: float((bg == g).mean()) for g in [1, 2, 3, 4]}
    tgt = {
        g: float(((grade == g) & (y == 1)).sum() / max((y == 1).sum(), 1))
        for g in [1, 2, 3, 4]
    }
    w = np.ones(len(y))
    for g in [1, 2, 3, 4]:
        m = (grade == g) & (y == 1)
        if m.sum() and tgt[g] > 0:
            w[m] = src[g] / tgt[g]
    posm = y == 1
    w[posm] *= posm.sum() / w[posm].sum()

    def probs(run):
        d = np.load(REPO / "results" / run / "test_predictions.npz")
        if not np.array_equal(
            d["y_true"][:, 0], y.astype("float32")
        ):  # row-order contract
            raise RuntimeError(f"{run}: y_true does not match labels.csv order")
        return d["y_prob"][:, 0], float(d["thresholds"][0])

    def agg(vals):
        v = [x for x in vals if x is not None]
        return {
            "per_seed": [r(x) for x in v],
            "mean": r(np.mean(v)) if v else None,
            "std": r(np.std(v, ddof=1)) if len(v) > 1 else None,
        }

    ref_sel = (grade == 0) | (grade >= 2)
    y_ref = (grade[ref_sel] >= 2).astype(int)

    def measure(runs):
        nat, rew, refo, nodisc = [], [], [], []
        strat = {str(g): [] for g in [1, 2, 3, 4]}
        rec = {str(g): [] for g in [1, 2, 3, 4]}
        for run in runs:
            p, thr = probs(run)
            nat.append(auc(y, p))
            rew.append(auc(y, p, w))
            refo.append(auc(y_ref, p[ref_sel]))
            nodisc.append(auc(y[~disc], p[~disc]))
            for g in [1, 2, 3, 4]:
                sel = (y == 0) | ((grade == g) & (y == 1))
                strat[str(g)].append(auc(y[sel], p[sel]))
                m = grade == g
                rec[str(g)].append(float((p[m] >= thr).mean()) if m.sum() else None)
        return {
            "auc_native": agg(nat),
            "auc_severity_reweighted": agg(rew),
            "auc_referable_only": agg(refo),
            "auc_excluding_discordant_eyes": agg(nodisc),
            "auc_grade_stratified": {g: agg(v) for g, v in strat.items()},
            "recall_by_grade": {g: agg(v) for g, v in rec.items()},
        }

    zs, po = measure(ZS_RUNS), measure(PO_RUNS)

    def gap(key, sub=None):
        a = (zs[key][sub] if sub else zs[key])["mean"]
        b = (po[key][sub] if sub else po[key])["mean"]
        return r(b - a) if (a is not None and b is not None) else None

    g_nat = gap("auc_native")
    res = {
        "_what": (
            "Every decisive test repeated over 3 zero-shot seeds and 3 pooled seeds, "
            "because the project's noise floor (0.0199 DR F1, 9 patients) is the same "
            "order as the effects being decomposed."
        ),
        "zeroshot_runs": ZS_RUNS,
        "pooled_runs": PO_RUNS,
        "zeroshot": zs,
        "pooled": po,
        "quoted_figure_reconciliation": {
            "project_note_says": "0.9104 zero-shot -> 0.9463 pooled, gap 0.036",
            "0.9104_is": "lebase_b0_s0 alone, the BEST of three zero-shot seeds",
            "0.9463_is": "the MEAN of three pooled seeds",
            "seed_consistent_zeroshot_mean": zs["auc_native"]["mean"],
            "seed_consistent_pooled_mean": po["auc_native"]["mean"],
            "seed_consistent_gap": g_nat,
            "note": (
                "The 0.036 figure mixes a single best seed against a 3-seed mean. "
                "Measured consistently the gap is LARGER, not smaller."
            ),
        },
        "gaps": {
            "native": g_nat,
            "severity_reweighted": gap("auc_severity_reweighted"),
            "referable_only_grade1_dropped": gap("auc_referable_only"),
            "excluding_discordant_eyes": gap("auc_excluding_discordant_eyes"),
            "by_grade": {g: gap("auc_grade_stratified", g) for g in ["1", "2", "3", "4"]},
        },
        "frac_of_native_gap_removed": {
            "by_severity_reweighting": r(
                1 - gap("auc_severity_reweighted") / g_nat
            )
            if g_nat
            else None,
            "by_dropping_grade1_positives": r(
                1 - gap("auc_referable_only") / g_nat
            )
            if g_nat
            else None,
            "by_excluding_discordant_eye_images": r(
                1 - gap("auc_excluding_discordant_eyes") / g_nat
            )
            if g_nat
            else None,
        },
        "zeroshot_auc_lift_from_severity_matching_alone": r(
            zs["auc_severity_reweighted"]["mean"] - zs["auc_native"]["mean"]
        ),
    }
    out["7_seed_averaged"] = res


# ----------------------------------------------------------------------------
# 6. optional: image statistics on the test set (requires decoding -> Slurm)
# ----------------------------------------------------------------------------
def image_stats(out, meta, grade, y, p_zs, pred_zs, pred_po):
    from PIL import Image

    res = {
        "_what": (
            "Per-image appearance statistics for the 732 test images, used ONLY to ask "
            "whether the zero-shot misses are appearance outliers. Aggregates only."
        )
    }
    root = JOINT_SPLITS / "test"
    rows = []
    for f in meta["file"].to_numpy():
        try:
            im = Image.open(root / f).convert("RGB")
            w, h = im.size
            a = np.asarray(im.resize((256, 256)), dtype=np.float32) / 255.0
            g = a.mean(axis=2)
            # retina mask: drop the black surround so stats are not diluted
            mask = g > 0.08
            gm = g[mask] if mask.sum() > 100 else g.ravel()
            gy, gx = np.gradient(g)
            rows.append(
                dict(
                    file=f,
                    width=w,
                    height=h,
                    aspect=w / h,
                    brightness=float(gm.mean()),
                    contrast=float(gm.std()),
                    sharpness=float(np.sqrt(gx**2 + gy**2).mean()),
                    saturation=float(
                        (a.max(axis=2) - a.min(axis=2))[mask].mean()
                        if mask.sum() > 100
                        else 0.0
                    ),
                    retina_frac=float(mask.mean()),
                    r_mean=float(a[..., 0][mask].mean() if mask.sum() > 100 else 0.0),
                    g_mean=float(a[..., 1][mask].mean() if mask.sum() > 100 else 0.0),
                    b_mean=float(a[..., 2][mask].mean() if mask.sum() > 100 else 0.0),
                )
            )
        except Exception as e:
            rows.append(dict(file=f, error=str(e)))
    df = pd.DataFrame(rows)
    res["n_decoded"] = int(df["brightness"].notna().sum()) if "brightness" in df else 0
    res["n_failed"] = int(len(df) - res["n_decoded"])

    cols = [
        "brightness",
        "contrast",
        "sharpness",
        "saturation",
        "retina_frac",
        "aspect",
        "r_mean",
        "g_mean",
        "b_mean",
    ]
    # Are the zero-shot misses appearance outliers relative to the hits, among
    # positives only (so severity, not prevalence, is the contrast)?
    pm = y == 1
    miss = pm & ~pred_zs
    hit = pm & pred_zs
    comp = {}
    for c in cols:
        if c not in df:
            continue
        v = df[c].to_numpy(dtype=float)
        a, b = v[miss], v[hit]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            continue
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        from scipy import stats as st

        comp[c] = {
            "miss_mean": r(a.mean()),
            "hit_mean": r(b.mean()),
            "cohens_d": r((a.mean() - b.mean()) / sd) if sd > 0 else None,
            "mannwhitney_p": r(float(st.mannwhitneyu(a, b).pvalue), 5),
        }
    res["zeroshot_miss_vs_hit_among_positives"] = {
        "n_miss": int(miss.sum()),
        "n_hit": int(hit.sum()),
        "by_statistic": comp,
    }

    # Same contrast on severity, to show which axis actually separates.
    sev = {}
    gg = grade[pm]
    for c in cols:
        if c not in df:
            continue
        v = df[c].to_numpy(dtype=float)[pm]
        ok = np.isfinite(v) & np.isfinite(gg)
        if ok.sum() < 10:
            continue
        from scipy import stats as st

        rho, p = st.spearmanr(v[ok], gg[ok])
        sev[c] = {"spearman_rho_vs_icdr": r(rho), "p": r(float(p), 5)}
    res["appearance_vs_severity_among_positives"] = sev

    res["aggregate_summary"] = {
        c: numsum(df[c]) for c in cols if c in df
    }
    out["6_image_stats"] = res


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/data_forensics.json"))
    ap.add_argument(
        "--image_stats",
        action="store_true",
        help="decode the 732 test images (Slurm/qCPU24 only, never the login node)",
    )
    args = ap.parse_args()

    B, M = load_raw()
    out = {
        "_meta": {
            "script": "scripts/92_data_forensics.py",
            "question": (
                "What drives the BRSET->mBRSET DR AUC gap (0.9104 zero-shot vs 0.9463 "
                "pooled) now that class balance and image appearance are both ruled out?"
            ),
            "brset_csv": str(BRSET_CSV),
            "mbrset_csv": str(MBRSET_CSV),
            "brset_rows": int(len(B)),
            "mbrset_rows": int(len(M)),
            "dua_note": "aggregates only; no patient images or per-patient rows written",
        }
    }

    m_test = severity(B, M, out)
    label_definition(B, M, out)
    population(B, M, out)
    meta, grade, y, p_zs, p_po, pred_zs, pred_po = misses(M, out, m_test)
    structure(B, M, out, meta, grade, y, p_zs, p_po)
    seed_averaged(B, M, out)
    if args.image_stats:
        image_stats(out, meta, grade, y, p_zs, pred_zs, pred_po)

    # JSON keys must be strings; numpy/bool keys sneak in via value_counts dicts.
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return r(float(o))
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o

    out = clean(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)
    print(json.dumps(out, indent=2)[:200])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
