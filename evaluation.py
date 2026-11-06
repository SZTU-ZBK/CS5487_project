"""Shared metrics, predictions, report figures and cross-method summaries."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data_utils import RESULTS
from plot_utils import save_figure


def save_json(values, path):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(values, stream, indent=2, ensure_ascii=False)


def evaluate_predictions(labels, predictions, images, folder, method, trial,
                         train_seconds, predict_seconds, sample_indices=None,
                         probabilities=None):
    metrics = {
        "method": method, "trial": trial, "n_test": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "train_seconds": float(train_seconds), "predict_seconds": float(predict_seconds),
    }
    save_json(metrics, folder / "metrics.json")
    if sample_indices is None:
        sample_indices = np.arange(len(labels))
    table = pd.DataFrame({"sample_index": sample_indices, "true_label": labels,
                          "prediction": predictions, "correct": labels == predictions})
    if probabilities is not None:
        table["confidence"] = probabilities.max(axis=1)
        np.save(folder / "probabilities.npy", probabilities)
    table.to_csv(folder / "predictions.csv", index=False)
    report = classification_report(labels, predictions, labels=np.arange(10), output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(folder / "classification_report.csv")

    cm = confusion_matrix(labels, predictions, labels=np.arange(10))
    pd.DataFrame(cm, index=np.arange(10), columns=np.arange(10)).to_csv(folder / "confusion_matrix.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, normalized in zip(axes, [False, True]):
        values = cm / cm.sum(axis=1, keepdims=True) if normalized else cm
        image = ax.imshow(values, cmap="Blues", vmin=0)
        for row in range(10):
            for col in range(10):
                text = format(values[row, col], ".2f") if normalized else str(cm[row, col])
                color = "white" if values[row, col] > values.max() / 2 else "black"
                ax.text(col, row, text, ha="center", va="center", fontsize=7, color=color)
        ax.set(xticks=np.arange(10), yticks=np.arange(10), xlabel="Predicted digit", ylabel="True digit",
               title="Row-normalized" if normalized else "Counts")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(method + " | trial " + str(trial))
    fig.tight_layout()
    save_figure(fig, folder, "confusion_matrix")

    errors = np.where(labels != predictions)[0]
    fig, axes = plt.subplots(4, 5, figsize=(9, 7))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, index in zip(axes.ravel(), errors[:20]):
        ax.imshow(images[index], cmap="gray", vmin=0, vmax=1)
        ax.set_title("True " + str(labels[index]) + " / Pred " + str(predictions[index]), fontsize=9)
    fig.suptitle("First 20 errors in test order | " + method + " | trial " + str(trial))
    fig.tight_layout()
    save_figure(fig, folder, "misclassified_examples")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    per_class = cm.diagonal() / cm.sum(axis=1) * 100
    ax.bar(np.arange(10), per_class, color="steelblue")
    ax.set(xticks=np.arange(10), ylim=(0, 105), xlabel="Digit", ylabel="Recall (%)", title="Per-class recall")
    for digit, value in enumerate(per_class):
        ax.text(digit, value + 1, format(value, ".1f"), ha="center", fontsize=9)
    fig.tight_layout()
    save_figure(fig, folder, "per_class_recall")
    print(method, "trial", trial, "accuracy =", format(metrics["accuracy"], ".4f"))
    return metrics


def save_method_summary(rows, folder):
    table = pd.DataFrame(rows)
    table.to_csv(folder / "trial_results.csv", index=False)
    summary = table.groupby("method", sort=False).agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_train_seconds=("train_seconds", "mean"),
        mean_predict_seconds=("predict_seconds", "mean"),
    ).reset_index()
    summary.to_csv(folder / "summary.csv", index=False)
    return summary


def compare_methods():
    # Read saved results only: this function never selects or tunes a model.
    methods = ["1nn", "cnn_geometric", "cnn_enhanced", "scattering"]
    files = [RESULTS / name / "trial_results.csv" for name in methods
             if (RESULTS / name / "trial_results.csv").exists()]
    if not files:
        return pd.DataFrame()
    table = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    folder = RESULTS / "comparison"
    folder.mkdir(parents=True, exist_ok=True)
    summary = save_method_summary(table.to_dict("records"), folder)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = list(summary["method"])
    positions = np.arange(len(names))
    for offset, trial in [(-0.2, 1), (0.2, 2)]:
        subset = table[table["trial"] == trial].set_index("method")
        values = subset.loc[names, "accuracy"].to_numpy() * 100
        bars = ax.bar(positions + offset, values, width=0.38, label="Trial " + str(trial))
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.axhline(91.60, color="gray", linestyle="--", label="Provided 1-NN mean (91.60%)")
    ax.set(xticks=positions, xticklabels=names, ylabel="Test accuracy (%)", ylim=(0, 106), title="Prescribed two-trial comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, folder, "accuracy_comparison")
    # Pair CNN settings by trial so the ablation is visible without adding algorithms.
    pivot = table.pivot(index="trial", columns="method", values="accuracy")
    if "CNN geometric" in pivot.columns and "CNN enhanced" in pivot.columns:
        gain = (pivot["CNN enhanced"] - pivot["CNN geometric"]) * 100
        gain.rename("enhanced_gain_percentage_points").to_csv(folder / "cnn_ablation.csv")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(["Trial " + str(i) for i in gain.index], gain.values, color="teal")
        ax.bar_label(bars, fmt="%+.2f")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(ylabel="Accuracy gain (percentage points)", title="Effect of additional augmentation")
        ax.margins(y=0.25)
        fig.tight_layout()
        save_figure(fig, folder, "cnn_augmentation_ablation")
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for row, trial in enumerate([1, 2]):
            for key, label in [("cnn_geometric", "Geometric"), ("cnn_enhanced", "Enhanced")]:
                history = pd.read_csv(RESULTS / key / ("trial_" + str(trial)) / "history.csv")
                axes[row, 0].plot(history["epoch"], history["val_loss"], label=label)
                axes[row, 1].plot(history["epoch"], history["val_accuracy"] * 100, label=label)
            axes[row, 0].set(title="Trial " + str(trial), ylabel="Validation loss", xlabel="Epoch")
            axes[row, 1].set(title="Trial " + str(trial), ylabel="Validation accuracy (%)", xlabel="Epoch")
            for ax in axes[row]:
                ax.legend()
                ax.grid(alpha=0.2)
        fig.suptitle("Final two-setting CNN comparison (training-only validation)")
        fig.tight_layout()
        save_figure(fig, folder, "cnn_learning_curves")
    return summary
