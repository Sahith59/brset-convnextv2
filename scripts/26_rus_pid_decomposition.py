"""
Redundant / Unique / Synergistic (RUS) decomposition between the BRSET-expert
and the mBRSET-expert, evaluated on mBRSET's test set - the piece Dr. Ye asked
for on the whiteboard (N_s = BRSET-trained model's view, N_t = mBRSET-trained
model's view, feeding a Mixture-of-Experts whose combination should be guided
by how much of what the two experts "know" is shared vs. exclusive vs. only
visible when combined).

This is the Williams & Beer (2010) Partial Information Decomposition (PID),
the standard formalism behind "redundancy / uniqueness / synergy" - normally
used to decompose what two *sensory channels* or *modalities* jointly tell you
about a target. Here the two "channels" are two *models* (one specialized on
BRSET, one on mBRSET), and the target is the true label, evaluated on
mBRSET's held-out test set (732 images) so both experts are scored on the
exact same data.

For two binary sources X1, X2 and a binary target Y:
    I(X1;Y), I(X2;Y)       - standard mutual information
    I(X1,X2;Y)             - joint MI, treating (X1,X2) as one 4-state variable
    R  (redundancy, Imin)  = sum_y p(y) * min_i I_spec(Xi;Y=y)
    U1 (unique to X1)      = I(X1;Y) - R
    U2 (unique to X2)      = I(X2;Y) - R
    S  (synergy)           = I(X1,X2;Y) - R - U1 - U2

Run on binary decisions at each expert's own tuned operating threshold (i.e.
"did the model flag this image or not"), since that's the interpretable,
decision-level question a Mixture-of-Experts gate actually has to answer.
"""
import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("/home/users/sthummala2/brset-convnextv2/results")
LABEL_COLS = ["diabetic_retinopathy", "macular_edema"]

BRSET_THRESHOLDS = {"diabetic_retinopathy": 0.61, "macular_edema": 0.39}


def mutual_info(x, y):
    """I(X;Y) in bits. x, y are integer-valued arrays of any (small) cardinality."""
    mi = 0.0
    for xv in np.unique(x):
        for yv in np.unique(y):
            pxy = np.mean((x == xv) & (y == yv))
            px = np.mean(x == xv)
            py = np.mean(y == yv)
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * np.log2(pxy / (px * py))
    return mi


def specific_info(x, y, y_val):
    """I_spec(X; Y=y_val) = sum_x p(x|y) * log2( p(y|x) / p(y) )."""
    py = np.mean(y == y_val)
    if py == 0:
        return 0.0
    info = 0.0
    for xv in (0, 1):
        px = np.mean(x == xv)
        if px == 0:
            continue
        pxy = np.mean((x == xv) & (y == y_val))
        p_x_given_y = pxy / py
        p_y_given_x = pxy / px
        if p_x_given_y > 0 and p_y_given_x > 0:
            info += p_x_given_y * np.log2(p_y_given_x / py)
    return info


def joint_mi(x1, x2, y):
    """I((X1,X2); Y) treating the pair as one 4-state variable."""
    z = x1 * 2 + x2
    return mutual_info(z, y)


def pid_rus(x1, x2, y):
    i1 = mutual_info(x1, y)
    i2 = mutual_info(x2, y)
    ijoint = joint_mi(x1, x2, y)

    r = 0.0
    for yv in (0, 1):
        py = np.mean(y == yv)
        if py == 0:
            continue
        spec1 = specific_info(x1, y, yv)
        spec2 = specific_info(x2, y, yv)
        r += py * min(spec1, spec2)

    u1 = i1 - r
    u2 = i2 - r
    s = ijoint - r - u1 - u2
    return {
        "I_X1_Y": i1, "I_X2_Y": i2, "I_joint_Y": ijoint,
        "redundancy": r, "unique_source_BRSET": u1, "unique_source_mBRSET": u2,
        "synergy": s,
    }


def main():
    a = np.load(RESULTS_DIR / "cross_dataset_BRSET_on_mBRSET/predictions.npz")
    b = np.load(RESULTS_DIR / "convnextv2_large_mBRSET_multilabel_512_regularized/test_predictions.npz")
    assert np.array_equal(a["y_true"], b["y_true"]), "test sets must be identical/aligned"

    y_true_all = a["y_true"]
    prob_brset_expert = a["y_prob"]      # BRSET-trained model, evaluated on mBRSET test
    prob_mbrset_expert = b["y_prob"]     # mBRSET-trained model, evaluated on mBRSET test
    mbrset_thresholds = {LABEL_COLS[i]: float(b["thresholds"][i]) for i in range(len(LABEL_COLS))}

    print(f"n = {len(y_true_all)} mBRSET test images, both experts scored on the same set\n")

    all_results = {}
    for i, col in enumerate(LABEL_COLS):
        y = y_true_all[:, i].astype(int)
        x1 = (prob_brset_expert[:, i] >= BRSET_THRESHOLDS[col]).astype(int)   # BRSET-expert decision
        x2 = (prob_mbrset_expert[:, i] >= mbrset_thresholds[col]).astype(int)  # mBRSET-expert decision

        rus = pid_rus(x1, x2, y)
        total = rus["I_joint_Y"]
        print(f"{col}  (n_positive={y.sum()}/{len(y)})")
        print(f"  I(BRSET-expert ; label)      = {rus['I_X1_Y']:.4f} bits")
        print(f"  I(mBRSET-expert ; label)     = {rus['I_X2_Y']:.4f} bits")
        print(f"  I(both experts jointly ; label) = {rus['I_joint_Y']:.4f} bits")
        print(f"  ---- PID breakdown of the joint {total:.4f} bits ----")
        if total > 0:
            print(f"  Redundant  : {rus['redundancy']:.4f} bits ({100*rus['redundancy']/total:.1f}%)")
            print(f"  Unique-BRSET : {rus['unique_source_BRSET']:.4f} bits ({100*rus['unique_source_BRSET']/total:.1f}%)")
            print(f"  Unique-mBRSET: {rus['unique_source_mBRSET']:.4f} bits ({100*rus['unique_source_mBRSET']/total:.1f}%)")
            print(f"  Synergy    : {rus['synergy']:.4f} bits ({100*rus['synergy']/total:.1f}%)")
        else:
            print("  (joint MI is ~0, decomposition percentages undefined)")
        print()

        all_results[col] = rus

    with open(RESULTS_DIR / "rus_pid_decomposition.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"wrote {RESULTS_DIR / 'rus_pid_decomposition.json'}")


if __name__ == "__main__":
    main()
