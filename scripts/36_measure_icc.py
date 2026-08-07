"""Measure the within-patient / within-eye intraclass correlation (ICC) of the
model's logits on mBRSET, and use it to PREDICT what multi-view aggregation can
achieve -- before implementing it.

Theory (hierarchical binormal / homogeneous-bag model):
    S_ij = m_y + b_i + e_ij,   Corr(S_ij, S_ik) = rho
    mean of K images has variance V*(rho + (1-rho)/K)
    =>  AUC_K = Phi( z_1 * sqrt( K / (1 + (K-1)*rho) ) ),  z_1 = Phi^-1(AUC_1)

so the achievable gain is a closed-form function of one measurable quantity.
ICC(1,1) estimated by one-way random-effects ANOVA (Shrout & Fleiss,
Psychological Bulletin 1979).

Break-even for AUC 0.909 -> 0.95:  rho <= 0.317 at K=2, rho <= 0.545 at K=4.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

PRED = "/home/users/sthummala2/brset-convnextv2/results/cross_dataset_BRSET_on_mBRSET/predictions.npz"
TEST_LABELS = "/home/users/sthummala2/brset-convnextv2/data/finetune_mbrset_multilabel/test/labels.csv"
RAW = "/data/users4/nshaik3/Datasets/mBRSET/physionet.org/files/mbrset/1.0/labels_mbrset.csv"
LABELS = ["diabetic_retinopathy", "macular_edema"]
EPS = 1e-6


def icc_oneway(groups):
    """ICC(1,1) from one-way random effects ANOVA. `groups` = list of arrays."""
    groups = [np.asarray(g, float) for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return np.nan, 0, np.nan
    k = np.mean([len(g) for g in groups])
    all_vals = np.concatenate(groups)
    grand = all_vals.mean()
    n = len(groups)
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (n - 1)
    dfw = sum(len(g) - 1 for g in groups)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / dfw
    icc = (msb - msw) / (msb + (k - 1) * msw)
    return float(np.clip(icc, 0.0, 1.0)), n, float(k)


def auc_after_aggregation(auc1, rho, K):
    z1 = norm.ppf(auc1)
    return float(norm.cdf(z1 * np.sqrt(K / (1.0 + (K - 1) * rho))))


def main():
    z = np.load(PRED)
    y, p = z["y_true"], z["y_prob"]
    test = pd.read_csv(TEST_LABELS)
    assert (y[:, 0] == test.diabetic_retinopathy.values).all(), "prediction/label order mismatch"

    raw = pd.read_csv(RAW)[["file", "patient", "laterality"]]
    df = test.merge(raw, on="file", how="left", validate="one_to_one")
    assert df.laterality.notna().all(), "missing laterality for some test images"
    df["eye"] = df.patient.astype(str) + "_" + df.laterality.astype(str)

    logit = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))

    print(f"mBRSET test: {len(df)} images, {df.patient.nunique()} patients, {df.eye.nunique()} eyes")
    print(f"images per patient: {df.groupby('patient').size().mean():.2f} | "
          f"per eye: {df.groupby('eye').size().mean():.2f}")
    print()

    from sklearn.metrics import roc_auc_score
    for i, lab in enumerate(LABELS):
        yt = df[lab].values.astype(int)
        s = logit[:, i]
        auc1 = roc_auc_score(yt, s)
        print("=" * 74)
        print(f"{lab}   image-level AUC = {auc1:.4f}   ({yt.sum()} positive of {len(yt)})")
        print("=" * 74)

        for unit, col, K in (("PATIENT", "patient", 4), ("EYE", "eye", 2)):
            print(f"  -- grouping by {unit} (K={K})")
            rhos = {}
            for cls, name in ((0, "negatives"), (1, "positives")):
                groups = [g[col].index for _, g in df[yt == cls].groupby(col)]
                arrs = [s[np.array(ix)] for ix in
                        [df[yt == cls].groupby(col).indices[k]
                         for k in df[yt == cls].groupby(col).indices]]
                arrs = [a for a in arrs if len(a) >= 2]
                icc, n, kbar = icc_oneway(arrs)
                rhos[cls] = icc
                print(f"       {name:10s} ICC = {icc:.3f}   ({n} groups, mean size {kbar:.2f})")
            rho = np.nanmean([rhos[0], rhos[1]])
            pred = auc_after_aggregation(auc1, rho, K)
            breakeven = (K / (norm.ppf(0.95) / norm.ppf(auc1)) ** 2 - 1) / (K - 1)
            print(f"       pooled rho = {rho:.3f}")
            print(f"       PREDICTED AUC after aggregating {K} images = {pred:.4f}  "
                  f"(gain {pred-auc1:+.4f})")
            print(f"       break-even rho to reach 0.95 = {breakeven:.3f}  ->  "
                  f"{'REACHABLE' if rho <= breakeven else 'NOT reachable'}")
            print()


if __name__ == "__main__":
    main()
