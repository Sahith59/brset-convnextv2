"""
Compact 2-panel training curve (BRSET regularized + mBRSET regularized side
by side) for the executive report, to save space vs. two separate figures.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "/home/users/sthummala2/brset-convnextv2/results"
OUT_PATH = f"{RESULTS_DIR}/executive_training_curves.jpg"


def load_curve(log_path):
    epochs, aucs, f1s, scores = [], [], [], []
    with open(log_path) as f:
        for line in f:
            d = json.loads(line)
            epochs.append(d["epoch"])
            aucs.append(d["macro_auc"])
            f1s.append(d["macro_f1"])
            scores.append(d["score"])
    return epochs, aucs, f1s, scores


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    for ax, (title, log_path) in zip(axes, [
        ("BRSET (regularized)", f"{RESULTS_DIR}/convnextv2_large_BRSET_multilabel_512_regularized/log.txt"),
        ("mBRSET (regularized)", f"{RESULTS_DIR}/convnextv2_large_mBRSET_multilabel_512_regularized/log.txt"),
    ]):
        epochs, aucs, f1s, scores = load_curve(log_path)
        best_epoch = epochs[int(np.argmax(scores))]
        ax.plot(epochs, aucs, marker="o", markersize=2.5, label="Macro AUC", color="#2166ac", linewidth=1.3)
        ax.plot(epochs, f1s, marker="s", markersize=2.5, label="Macro F1", color="#b2182b", linewidth=1.3)
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Validation Score", fontsize=9)
        ax.set_title(f"{title} - best epoch {best_epoch}", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
