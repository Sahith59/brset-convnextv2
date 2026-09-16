"""Frozen pooled-feature GFP comparator; allocated CPU only."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "research/codex/step5"
BASE = ROOT / "research/codex/step2/full_pool/frozen"
PRIVATE = Path("/data/users3/sthummala2/brset-codex/step5/b1_features")
RUN = Path("/data/users3/sthummala2/brset-codex/step2-full-pool/B1_seed0")
sys.path.insert(0, str(BASE))
import baseline_core as core  # noqa: E402


GROUP_SIZES = [1, 4, 16, 64, 128]
MIN_RATIOS = [0.0, 0.25, 0.5, 0.75]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -60, 60)))


def score(truth, probability):
    return float((roc_auc_score(truth, probability) + average_precision_score(truth, probability)) / 2)


def group_contributions(features, weight, group_size):
    groups = features.shape[-1] // group_size
    shaped_features = features.reshape(*features.shape[:-1], groups, group_size)
    shaped_weight = weight.reshape(groups, group_size)
    return np.einsum("...gs,gs->...g", shaped_features, shaped_weight, optimize=True)


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Allocated node required")
    extraction = json.loads((WORK / "feature_extraction_summary.json").read_text())
    if extraction["status"] != "pass" or extraction["assessment_accessed"]:
        raise ValueError("Feature extraction gate failed")
    fit_x = np.load(PRIVATE / "fit_pooled.npy", mmap_mode="r")[:, 0]
    val_x = np.load(PRIVATE / "selection_pooled.npy", mmap_mode="r")
    rows = np.load(PRIVATE / "rows.npz", allow_pickle=False)
    head = np.load(PRIVATE / "head.npz", allow_pickle=False)
    fit_y, val_y = rows["fit_y"], rows["selection_y"]
    weight, bias = head["weight"], head["bias"]
    if fit_x.shape != (14774, 1536) or val_x.shape != (725, 4, 1536):
        raise ValueError(f"Unexpected pooled shapes: {fit_x.shape}, {val_x.shape}")

    saved = np.load(RUN / "selection_predictions.npz", allow_pickle=False)
    summary = json.loads((RUN / "selection_summary.json").read_text())
    unmasked_logits = np.einsum("nvc,kc->nvk", val_x, weight, optimize=True) + bias
    unmasked_probability = sigmoid(unmasked_logits).mean(1)
    reproduction_error = float(np.max(np.abs(unmasked_probability - saved["probabilities"])))
    if reproduction_error > 1e-2:
        raise ValueError(f"Cached pooled feature reproduction failed: {reproduction_error}")
    unmasked_thresholds = core.select_thresholds(val_y, unmasked_probability)
    baseline_metrics = core.metrics(val_y, unmasked_probability, unmasked_thresholds)

    started = time.monotonic()
    candidates = []
    private_rankings = {}
    for group_size in GROUP_SIZES:
        if fit_x.shape[1] % group_size:
            continue
        groups = fit_x.shape[1] // group_size
        fit_contrib = np.empty((len(fit_x), groups, 2), dtype=np.float32)
        val_contrib = np.empty((len(val_x), val_x.shape[1], groups, 2), dtype=np.float32)
        rankings = []
        prefix_scores = []
        for label in range(2):
            fit_contrib[:, :, label] = group_contributions(fit_x, weight[label], group_size)
            val_contrib[:, :, :, label] = group_contributions(val_x, weight[label], group_size)
            solo = np.asarray([score(fit_y[:, label], sigmoid(bias[label] + fit_contrib[:, group, label]))
                               for group in range(groups)])
            ranking = np.lexsort((np.arange(groups), -solo))
            rankings.append(ranking)
            cumulative = np.zeros(len(fit_x), dtype=np.float32)
            scores = np.empty(groups, dtype=np.float64)
            for prefix, group in enumerate(ranking):
                cumulative += fit_contrib[:, group, label]
                scores[prefix] = score(fit_y[:, label], sigmoid(bias[label] + cumulative))
            prefix_scores.append(scores)
        private_rankings[str(group_size)] = {
            core.LABELS[label]: rankings[label].tolist() for label in range(2)
        }

        for minimum_ratio in MIN_RATIOS:
            probabilities = np.empty((len(val_x), 2), dtype=np.float64)
            retained = []
            for label in range(2):
                minimum_groups = max(1, math.ceil(minimum_ratio * groups))
                eligible = prefix_scores[label][minimum_groups - 1:]
                best_value = eligible.max()
                tied = np.flatnonzero(np.isclose(eligible, best_value, atol=1e-12, rtol=0))
                prefix_count = minimum_groups + int(tied[-1])
                chosen = rankings[label][:prefix_count]
                retained.append(prefix_count * group_size)
                logits = bias[label] + val_contrib[:, :, chosen, label].sum(axis=2)
                probabilities[:, label] = sigmoid(logits).mean(1)
            aurocs = [roc_auc_score(val_y[:, label], probabilities[:, label]) for label in range(2)]
            candidates.append({
                "group_size": group_size,
                "minimum_keep_ratio": minimum_ratio,
                "retained_features": retained,
                "validation_mean_auroc": float(np.mean(aurocs)),
                "probabilities": probabilities,
            })

    candidates.sort(key=lambda item: (-item["validation_mean_auroc"],
                                      -sum(item["retained_features"]), item["group_size"],
                                      item["minimum_keep_ratio"]))
    selected = candidates[0]
    thresholds = core.select_thresholds(val_y, selected["probabilities"])
    selected_metrics = core.metrics(val_y, selected["probabilities"], thresholds)
    deltas = {label: {metric: selected_metrics[label][metric] - baseline_metrics[label][metric]
                      for metric in ["f1_positive", "auroc", "average_precision"]}
              for label in core.LABELS}
    f1 = [deltas[label]["f1_positive"] for label in core.LABELS]
    auc = [deltas[label]["auroc"] for label in core.LABELS]
    gate = ((f1[0] >= .01 and f1[1] >= -.01) or (f1[1] >= .01 and f1[0] >= -.01)) and min(auc) >= -.01

    private = PRIVATE / "gfp_private.npz"
    np.savez_compressed(private, validation_probabilities=selected["probabilities"],
                        validation_y=val_y, thresholds=np.asarray(thresholds),
                        rankings_json=np.asarray(json.dumps(private_rankings)))
    public_candidates = [{key: value for key, value in item.items() if key != "probabilities"}
                         for item in candidates]
    result = {
        "status": "pass",
        "job_id": os.environ["SLURM_JOB_ID"],
        "method": "Independent frozen-head GFP reproduction/comparator",
        "fit_images": len(fit_x),
        "selection_images": len(val_x),
        "assessment_accessed": False,
        "unmasked_reproduction_max_abs_error": reproduction_error,
        "unmasked_reproduction_tolerance": 0.01,
        "unmasked_thresholds": unmasked_thresholds,
        "baseline_metrics": baseline_metrics,
        "selected": {key: value for key, value in selected.items() if key != "probabilities"},
        "selected_thresholds": thresholds,
        "selected_metrics": selected_metrics,
        "delta_from_unmasked_b1": deltas,
        "engineering_screen_pass": bool(gate),
        "candidate_summaries": public_candidates,
        "seconds": time.monotonic() - started,
        "private_result_sha256": sha(private),
        "protocol_sha256": sha(WORK / "GFP_PROTOCOL.md"),
        "script_sha256": sha(Path(__file__)),
        "interpretation": "Validation-selected published comparator; not the proposed contribution and no test assessment.",
    }
    (WORK / "gfp_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
