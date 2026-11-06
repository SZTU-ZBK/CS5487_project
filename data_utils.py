"""Read the supplied data without extracting or changing the original archives."""

import io
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42


def load_data(source="mat"):
    if source == "mat":
        with zipfile.ZipFile(ROOT / "digits4000.zip") as archive:
            raw = loadmat(io.BytesIO(archive.read("digits4000.mat")))
        vectors = raw["digits_vec"].T
        labels = raw["digits_labels"].ravel()
        train_indices = raw["trainset"]
        test_indices = raw["testset"]
    elif source == "txt":
        with zipfile.ZipFile(ROOT / "digits4000_txt.zip") as archive:
            def read_text(name):
                content = archive.read("digits4000_txt/digits4000_" + name + ".txt")
                return np.loadtxt(io.BytesIO(content), dtype=np.int64)

            vectors = read_text("digits_vec")
            labels = read_text("digits_labels").ravel()
            train_indices = read_text("trainset").T
            test_indices = read_text("testset").T
    else:
        raise ValueError("source must be 'mat' or 'txt'")

    # MATLAB stores each image column by column. Transpose each reshaped image.
    images = vectors.reshape(-1, 28, 28).transpose(0, 2, 1)
    images = np.ascontiguousarray(images, dtype=np.float32) / 255.0
    data = {
        "images": images,
        "labels": labels.astype(np.int64),
        "train_indices": train_indices.astype(np.int64) - 1,
        "test_indices": test_indices.astype(np.int64) - 1,
    }
    validate_data(data)
    return data


def validate_data(data):
    assert data["images"].shape == (4000, 28, 28)
    assert data["labels"].shape == (4000,)
    assert np.isfinite(data["images"]).all()
    assert data["images"].min() >= 0 and data["images"].max() <= 1
    assert np.array_equal(np.bincount(data["labels"], minlength=10), [400] * 10)
    for trial in range(2):
        train = data["train_indices"][trial]
        test = data["test_indices"][trial]
        assert len(train) == len(test) == 2000
        assert len(np.unique(train)) == len(np.unique(test)) == 2000
        assert np.intersect1d(train, test).size == 0
        assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(4000))
        assert np.array_equal(np.bincount(data["labels"][train], minlength=10), [200] * 10)
        assert np.array_equal(np.bincount(data["labels"][test], minlength=10), [200] * 10)


def get_trial(data, trial):
    if trial not in (1, 2):
        raise ValueError("trial must be 1 or 2")
    train = data["train_indices"][trial - 1]
    test = data["test_indices"][trial - 1]
    return (
        data["images"][train], data["labels"][train],
        data["images"][test], data["labels"][test],
    )


def validation_indices(labels, seed=SEED):
    indices = np.arange(len(labels))
    return train_test_split(indices, test_size=0.2, stratify=labels, random_state=seed)


def output_folder(method, trial=None):
    folder = RESULTS / method
    if trial is not None:
        folder = folder / ("trial_" + str(trial))
    folder.mkdir(parents=True, exist_ok=True)
    return folder
