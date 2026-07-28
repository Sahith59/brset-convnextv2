"""
Generate the two figures for a multi-label report (training curve + per-label
confusion matrices), using the actual logged/evaluated numbers. Works for any
of the trained models by passing the result directory and metrics JSON path.

Usage: python 10_generate_figures.py <result_dir> <metrics_json_path> <best_epoch_label>
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(log_path, out_path, title_suffix):
    epochs, aucs, f1s, scores = [], [], [], []
    with open(log_path) as f:
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
    ax.text(best_epoch + 0.3, min(f1s) + 0.02, f"best epoch = {best_epoch}", fontsize=9, color="gray")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Score")
    ax.set_title(f"Validation Macro AUC / F1 by Epoch {title_suffix}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_confusion_matrices(metrics_json_path, out_path):
    with open(metrics_json_path) as f:
        report = json.load(f)
    per_label = report["per_label"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, label in zip(axes, ["diabetic_retinopathy", "macular_edema"]):
        cm = np.array(per_label[label]["confusion_matrix"])
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
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    result_dir = sys.argv[1]
    metrics_json = sys.argv[2]
    title_suffix = sys.argv[3] if len(sys.argv) > 3 else ""
    plot_training_curve(f"{result_dir}/log.txt", f"{result_dir}/training_curve.jpg", title_suffix)
    plot_confusion_matrices(metrics_json, f"{result_dir}/confusion_matrices.jpg")
