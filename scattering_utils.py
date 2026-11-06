"""Fixed wavelet features, cached in batches; learned scaling stays in the SVM pipeline."""

import hashlib
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from kymatio import Scattering2D
import torch

from plot_utils import save_figure


def scattering_features(images, cache_path, device="cpu", batch_size=64, J=2):
    images = np.ascontiguousarray(images, dtype=np.float32)
    if images.ndim != 3 or images.shape[1:] != (28, 28) or len(images) == 0:
        raise ValueError("Expected nonempty images with shape [N, 28, 28]")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Images must be finite and scaled to [0, 1]")
    metadata_path = cache_path.with_suffix(".json")
    metadata = {"input_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
                "J": J, "L": 8, "max_order": 2, "shape": list(images.shape)}
    if cache_path.exists() and metadata_path.exists():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved == metadata:
            features = np.load(cache_path)
            print("Loaded fixed-feature cache:", features.shape)
            return features, 0.0
    # J=2, L=8 and max_order=2 produce 81 maps, each of size 7 x 7.
    # Select only the 2-D frontend. Importing kymatio.torch also imports its
    # unused 3-D frontend, which relies on a removed SciPy function.
    scattering = Scattering2D(J=J, shape=(28, 28), L=8, max_order=2, frontend="torch").to(device)
    parts = []
    start = time.perf_counter()
    with torch.no_grad():
        for start_index in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start_index:start_index + batch_size]).to(device)
            coefficients = scattering(batch)
            parts.append(coefficients.flatten(start_dim=1).cpu().numpy())
            if start_index % (batch_size * 10) == 0:
                print("Scattering:", min(start_index + batch_size, len(images)), "/", len(images))
    features = np.concatenate(parts).astype(np.float32)
    seconds = time.perf_counter() - start
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return features, seconds


def plot_scattering(images, labels, folder, device="cpu"):
    selected = [np.where(labels == digit)[0][0] for digit in [0, 3, 8]]
    scattering = Scattering2D(J=2, shape=(28, 28), L=8, max_order=2, frontend="torch").to(device)
    with torch.no_grad():
        maps = scattering(torch.from_numpy(images[selected]).to(device)).cpu().numpy()
    channels = [0, 1, 9, 17]
    fig, axes = plt.subplots(3, 5, figsize=(11, 6))
    for row, index in enumerate(selected):
        axes[row, 0].imshow(images[index], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("Digit " + str(labels[index]))
        for col, channel in enumerate(channels, start=1):
            axes[row, col].imshow(maps[row, channel], cmap="magma")
            axes[row, col].set_title("Coefficient " + str(channel))
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Fixed wavelet scattering: selected 7 x 7 coefficient maps")
    fig.tight_layout()
    save_figure(fig, folder, "scattering_feature_maps")
