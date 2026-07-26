"""
Generate the two figures for the multi-label report: the training curve
(showing the epoch-9 peak and subsequent plateau) and per-label confusion
matrix heatmaps, using the actual logged/evaluated numbers (nothing
re-computed or estimated).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = "/home/users/sthummala2/brset-convnextv2/results/convnextv2_large_BRSET_multilabel_512/log.txt"
REPORT_JSON = "/home/users/sthummala2/brset-convnextv2/results/multilabel_classification_report.json"
CURVE_OUT = "/home/users/sthummala2/brset-convnextv2/results/convnextv2_large_BRSET_multilabel_512/training_curve.jpg"
CM_OUT = "/home/users/sthummala2/brset-convnextv2/results/convnextv2_large_BRSET_multilabel_512/confusion_matrices.jpg"


def plot_training_curve():
    epochs, aucs, f1s, scores = [], [], [], []
    with open(LOG_PATH) as f:
        for line in f:
            d = json.loads(line)
            epochs.append(d["epoch"])
            aucs.append(d["macro_auc"])
            f1s.append(d["macro_f1"])
            scores.append(d["score"])

    best_epoch = epochs[int(np.argmax(scores))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, aucs, marker="o", markersize=3, label="Macro AUC", color="#2166ac")
    ax.plot(epochs, f1s, marker="s", markersize=3, label="Macro F1", color="#b2182b")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    ax.text(best_epoch + 0.3, 0.72, f"best epoch = {best_epoch}", fontsize=9, color="gray")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Score")
    ax.set_title("Validation Macro AUC / F1 by Epoch (stopped at epoch 22)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CURVE_OUT, dpi=200)
    plt.close(fig)
    print(f"wrote {CURVE_OUT}")


def plot_confusion_matrices():
    with open(REPORT_JSON) as f:
        report = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, label in zip(axes, ["diabetic_retinopathy", "macular_edema"]):
        cm = np.array(report["per_label"][label]["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Negative", "Positive"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(label, fontsize=11)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.3f})", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=10)
    fig.tight_layout()
    fig.savefig(CM_OUT, dpi=200)
    plt.close(fig)
    print(f"wrote {CM_OUT}")


if __name__ == "__main__":
    plot_training_curve()
    plot_confusion_matrices()
