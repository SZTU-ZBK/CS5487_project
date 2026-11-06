"""Small plotting functions; save every figure as PNG and PDF."""

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save_figure(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / (name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(folder / (name + ".pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_data_overview(data, folder):
    # Use only trial 1 training images for visual inspection of handwriting.
    train = data["train_indices"][0]
    images = data["images"][train]
    labels = data["labels"][train]
    fig, axes = plt.subplots(3, 10, figsize=(12, 4))
    for digit in range(10):
        selected = np.where(labels == digit)[0][:3]
        for row, index in enumerate(selected):
            axes[row, digit].imshow(images[index], cmap="gray", vmin=0, vmax=1)
            axes[row, digit].axis("off")
            if row == 0:
                axes[row, digit].set_title(str(digit))
    fig.suptitle("Training examples: correctly reconstructed 28 x 28 images")
    fig.tight_layout()
    save_figure(fig, folder, "training_examples")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    digits = np.arange(10)
    for offset, key, label in [(-0.18, "train_indices", "Train"), (0.18, "test_indices", "Test")]:
        counts = np.bincount(data["labels"][data[key][0]], minlength=10)
        axes[0].bar(digits + offset, counts, width=0.36, label=label)
    axes[0].set(xticks=digits, xlabel="Digit", ylabel="Images", title="Class balance (each trial)")
    axes[0].legend()
    axes[1].hist(images.ravel(), bins=32, color="steelblue")
    axes[1].set(xlabel="Pixel value after /255", ylabel="Pixels", title="Training pixel distribution")
    axes[1].set_yscale("log")
    fig.tight_layout()
    save_figure(fig, folder, "class_balance_and_pixels")

    split = np.zeros((2, 4000), dtype=int)
    for trial in range(2):
        split[trial, data["test_indices"][trial]] = 1
    fig, ax = plt.subplots(figsize=(11, 2.5))
    image = ax.imshow(split, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=0, vmax=1)
    ax.set(yticks=[0, 1], yticklabels=["Trial 1", "Trial 2"], xlabel="Original image index (zero-based)", title="Prescribed train/test splits")
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["Train", "Test"])
    fig.tight_layout()
    save_figure(fig, folder, "prescribed_splits")


def plot_history(history, folder):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in [("train", "Train"), ("val", "Validation")]:
        axes[0].plot(history["epoch"], history[key + "_loss"], label=label)
        axes[1].plot(history["epoch"], history[key + "_accuracy"] * 100, label=label)
    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    for ax in axes:
        ax.axvline(best_epoch, color="gray", linestyle="--", label="Selected epoch")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set(ylabel="Cross-entropy loss", title="Loss curves")
    axes[1].set(ylabel="Accuracy (%)", title="Accuracy curves")
    fig.tight_layout()
    save_figure(fig, folder, "learning_curves")


def plot_training_split(labels, fit, val, folder):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    digits = np.arange(10)
    ax.bar(digits - 0.2, np.bincount(labels[fit], minlength=10), width=0.4, label="Fit (1,600)")
    ax.bar(digits + 0.2, np.bincount(labels[val], minlength=10), width=0.4, label="Validation (400)")
    ax.set(xticks=digits, xlabel="Digit", ylabel="Images", title="Training-only stratified validation split")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, folder, "validation_split")
