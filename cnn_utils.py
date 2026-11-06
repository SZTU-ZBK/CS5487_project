"""A small CNN, mild augmentation, and plain training loops."""

import copy
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from plot_utils import save_figure
from augmentation import geometric_augmentation, additional_augmentation


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DigitDataset(Dataset):
    def __init__(self, images, labels, augment=False, settings=None):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.settings = settings or {}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index].copy()
        if self.augment:
            image = geometric_augmentation(image, self.settings)
            image = additional_augmentation(image, self.settings)
        return torch.from_numpy(image.copy()).unsqueeze(0), int(self.labels[index])


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def make_loader(images, labels, config, training=False):
    dataset = DigitDataset(images, labels, augment=training, settings=config.get("augmentation", {}))
    generator = torch.Generator().manual_seed(config["seed"])
    # num_workers=0 keeps Windows notebooks straightforward and reproducible.
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=training,
                      num_workers=0, generator=generator)


def run_epoch(model, loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += len(labels)
    return total_loss / total_examples, total_correct / total_examples


def new_model_and_optimizer(config):
    set_seed(config["seed"])
    model = SimpleCNN().to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    return model, optimizer


def train_cnn(fit_images, fit_labels, val_images, val_labels, config, folder):
    model, optimizer = new_model_and_optimizer(config)
    train_loader = make_loader(fit_images, fit_labels, config, True)
    val_loader = make_loader(val_images, val_labels, config)
    loss_function = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    rows = []
    start = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, loss_function, config["device"], optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, loss_function, config["device"])
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     "train_accuracy": train_accuracy, "val_accuracy": val_accuracy,
                     "elapsed_seconds": time.perf_counter() - start})
        # Save the numeric log after every epoch, even if a later run is interrupted.
        pd.DataFrame(rows).to_csv(folder / "history.csv", index=False)
        print(f"Epoch {epoch:02d}: loss {train_loss:.4f}/{val_loss:.4f}, accuracy {train_accuracy:.3f}/{val_accuracy:.3f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            torch.save({"state_dict": best_state, "config": config, "epoch": best_epoch,
                        "augmentation": config.get("augmentation", {})}, folder / "validation_checkpoint.pt")
        else:
            wait += 1
        if wait >= config["patience"]:
            print("Early stopping. Selected epoch:", best_epoch)
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), best_epoch


def fit_full_training_set(images, labels, config, epochs, folder):
    # Reinitialize, then use ALL 2,000 training images for a validation-selected
    # number of epochs. Test images and labels never enter this function.
    model, optimizer = new_model_and_optimizer(config)
    loader = make_loader(images, labels, config, True)
    loss_function = nn.CrossEntropyLoss()
    rows = []
    for epoch in range(1, epochs + 1):
        loss, accuracy = run_epoch(model, loader, loss_function, config["device"], optimizer)
        rows.append({"epoch": epoch, "train_loss": loss, "train_accuracy": accuracy})
        pd.DataFrame(rows).to_csv(folder / "final_training_history.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epochs,
                "augmentation": config.get("augmentation", {})}, folder / "final_model.pt")
    history = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(history["epoch"], history["train_loss"])
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Final fit on 2,000 training images")
    axes[1].plot(history["epoch"], history["train_accuracy"] * 100)
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Augmented training accuracy")
    fig.tight_layout()
    save_figure(fig, folder, "final_training_curves")
    return model


def predict_cnn(model, images, config):
    images = np.ascontiguousarray(images, dtype=np.float32)
    if images.ndim != 3 or images.shape[1:] != (28, 28) or len(images) == 0:
        raise ValueError("Expected nonempty images with shape [N, 28, 28]")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Images must be finite and scaled to [0, 1]")
    labels = np.zeros(len(images), dtype=np.int64)
    loader = make_loader(images, labels, config)
    model.eval()
    batches = []
    with torch.no_grad():
        for batch, _ in loader:
            logits = model(batch.to(config["device"]))
            batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.concatenate(batches)
    return probabilities.argmax(axis=1), probabilities


def load_cnn_model(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    model = SimpleCNN().to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    config = saved["config"]
    config["device"] = device
    return model, config


def plot_augmentation(images, labels, folder, settings, seed=42, geometric_settings=None):
    set_seed(seed)
    if geometric_settings is None:
        geometric_settings = {"rotation": 10, "translation": 2, "scale_min": 0.85,
                              "scale_max": 1.15, "shear": 0}
    fig, axes = plt.subplots(10, 3, figsize=(7, 15))
    titles = ["Original", "Geometric", "Selected enhanced"]
    for digit in range(10):
        image = images[np.where(labels == digit)[0][0]]
        state = random.getstate()
        transformed = geometric_augmentation(image, geometric_settings)
        random.setstate(state)
        selected = geometric_augmentation(image, settings)
        examples = [image, transformed, additional_augmentation(selected, settings)]
        for col in range(3):
            axes[digit, col].imshow(examples[col], cmap="gray", vmin=0, vmax=1)
            axes[digit, col].axis("off")
            if digit == 0:
                axes[digit, col].set_title(titles[col])
    fig.suptitle("Training-only examples of the selected augmentation policy")
    fig.tight_layout()
    save_figure(fig, folder, "augmentation_examples")
