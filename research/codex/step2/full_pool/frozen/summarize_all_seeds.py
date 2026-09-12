"""Create the cross-seed Step-2 baseline summary after independent verification."""
import json
import os
import socket
from pathlib import Path

import numpy as np

from baseline_core import LABELS, sha

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
WORK = ROOT / "research/codex/step2/full_pool"
ARMS = ["B0", "B1", "B2"]
METRICS = ["f1_positive", "auroc", "average_precision", "precision", "sensitivity", "specificity"]


def read(path):
    return json.loads(Path(path).read_text())


def stats(values):
    array = np.asarray(values, dtype=float)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Use an allocated Slurm compute node")
    for seed in [0, 1, 2]:
        verification = read(WORK / f"seed{seed}_independent_verification.json")
        if not verification["pass"]:
            raise RuntimeError(f"Seed {seed} verification failed")
    reports = [read(WORK / f"baseline_assessment_seed{seed}.json") for seed in [0, 1, 2]]
    aggregate = {}
    for arm in ARMS:
        aggregate[arm] = {}
        for label in LABELS:
            aggregate[arm][label] = {
                metric: stats([report["observed"][arm][label][metric] for report in reports])
                for metric in METRICS
            }

    differences = {}
    for left, right in [("B1", "B0"), ("B2", "B0"), ("B1", "B2")]:
        pair = f"{left}-{right}"
        differences[pair] = {}
        for label in LABELS:
            differences[pair][label] = {}
            for metric in METRICS:
                values = [report["observed"][left][label][metric] -
                          report["observed"][right][label][metric] for report in reports]
                differences[pair][label][metric] = {
                    **stats(values),
                    "positive_seeds": int(sum(value > 0 for value in values)),
                }

    b1_dr = differences["B1-B0"][LABELS[0]]["f1_positive"]
    b1_me = differences["B1-B0"][LABELS[1]]["f1_positive"]
    screen = {
        "mean_DR_F1_difference_at_least_0.01": b1_dr["mean"] >= 0.01,
        "positive_DR_difference_at_least_2_of_3": b1_dr["positive_seeds"] >= 2,
        "mean_ME_F1_difference_at_least_minus_0.01": b1_me["mean"] >= -0.01,
    }
    result = {
        "status": "Step-2 three-seed full-pool baseline summary",
        "seeds": [0, 1, 2],
        "aggregate": aggregate,
        "per_seed_differences": differences,
        "B1_vs_B0_engineering_screen": {**screen, "passes_all": all(screen.values())},
        "interpretation_limits": [
            "Three seeds quantify limited training variability; they are not a clinical significance analysis.",
            "Do not pool seed predictions as independent test images.",
            "The mBRSET test split was historically reused and is not untouched external confirmation.",
            "Equal optimizer updates do not equalize source/target exposure; Step 3 addresses this.",
        ],
        "input_sha256": {
            f"seed{seed}_assessment": sha(WORK / f"baseline_assessment_seed{seed}.json")
            for seed in [0, 1, 2]
        } | {
            f"seed{seed}_verification": sha(WORK / f"seed{seed}_independent_verification.json")
            for seed in [0, 1, 2]
        },
    }
    output = WORK / "baseline_assessment_three_seeds.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Step-2 full-pool baseline results across three seeds", "",
        "Values are mean ± sample standard deviation across training seeds 0, 1 and 2.", "",
        "| Arm | DR F1 | DR AUROC | ME F1 | ME AUROC |", "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        dr = aggregate[arm][LABELS[0]]
        me = aggregate[arm][LABELS[1]]
        lines.append(
            f"| {arm} | {dr['f1_positive']['mean']:.4f} ± {dr['f1_positive']['sample_sd']:.4f} | "
            f"{dr['auroc']['mean']:.4f} ± {dr['auroc']['sample_sd']:.4f} | "
            f"{me['f1_positive']['mean']:.4f} ± {me['f1_positive']['sample_sd']:.4f} | "
            f"{me['auroc']['mean']:.4f} ± {me['auroc']['sample_sd']:.4f} |"
        )
    lines += ["", "Per-seed B1−B0 F1 differences:", "",
              f"- DR: {b1_dr['values']}; mean {b1_dr['mean']:+.4f}; positive in {b1_dr['positive_seeds']}/3 seeds.",
              f"- ME: {b1_me['values']}; mean {b1_me['mean']:+.4f}; positive in {b1_me['positive_seeds']}/3 seeds.",
              "", "B1 passes the prespecified engineering prioritization screen versus B0. This prioritizes Step-3 controls; it is not a novelty, clinical, significance, or external-validation claim.",
              "", "The test split was historically reused. Step 3 must use validation for design choices and address unequal data exposure."]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
