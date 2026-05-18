from pathlib import Path

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from loguru import logger
from tqdm import tqdm
import typer

from manos_hablando.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()

KEYPOINTS_FILE = PROCESSED_DATA_DIR / "keypoints.json"


def load_keypoints(path: Path = KEYPOINTS_FILE) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load keypoints from JSON file.

    Returns:
        X: np.ndarray of shape (n_samples, 63) — flattened keypoints
        y: np.ndarray of shape (n_samples,) — encoded letter labels
        encoder: fitted LabelEncoder to decode predictions back to letters
    """
    with open(path) as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} samples from {path}")

    X = []
    labels = []

    for entry in tqdm(dataset, desc="Loading keypoints"):
        keypoints = np.array(entry["keypoints"], dtype=np.float32)  # (21, 3)
        X.append(keypoints.flatten())                                 # (63,)
        labels.append(entry["letter"])

    X = np.array(X, dtype=np.float32)
    labels = np.array(labels)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    logger.info(f"Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")

    return X, y, encoder


def normalize_keypoints(X: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints per sample.

    Strategy: subtract wrist position (point 0) and scale by the
    max distance from wrist — makes the model invariant to hand
    position and size in the image.

    Args:
        X: np.ndarray of shape (n_samples, 63)

    Returns:
        X_norm: np.ndarray of shape (n_samples, 63)
    """
    X = X.reshape(-1, 21, 3)

    # Translate: subtract wrist (point 0) from all points
    wrist = X[:, 0:1, :]
    X = X - wrist

    # Scale: divide by max distance from wrist per sample
    distances = np.linalg.norm(X, axis=2)
    max_dist = distances.max(axis=1, keepdims=True)
    max_dist = np.where(max_dist == 0, 1, max_dist)
    X = X / max_dist[:, :, np.newaxis]

    return X.reshape(-1, 63).astype(np.float32)


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """
    Split dataset into train, validation and test sets.
    Stratified by class to maintain class balance across splits.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1"

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_size + test_size),
        stratify=y,
        random_state=random_state,
    )

    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=random_state,
    )

    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Val:   {len(X_val)} samples")
    logger.info(f"Test:  {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


class LSMDataset(Dataset):
    """PyTorch Dataset for LSM hand sign keypoints."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    """
    Full pipeline: load → normalize → split → return DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, encoder
    """
    X, y, encoder = load_keypoints()
    X = normalize_keypoints(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    train_loader = DataLoader(LSMDataset(X_train, y_train), batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(LSMDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(LSMDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, encoder


@app.command()
def main(
    input_path: Path = KEYPOINTS_FILE,
    batch_size: int = 32,
):
    """Load, normalize and split the LSM keypoints dataset."""
    logger.info("Loading dataset...")
    X, y, encoder = load_keypoints(input_path)

    logger.info("Normalizing keypoints...")
    X = normalize_keypoints(X)

    logger.info("Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # Sanity check
    train_loader, val_loader, test_loader, encoder = get_dataloaders(batch_size=batch_size)
    X_batch, y_batch = next(iter(train_loader))

    logger.info(f"Batch X shape: {X_batch.shape}")
    logger.info(f"Batch y shape: {y_batch.shape}")
    logger.info(f"Sample labels: {[encoder.classes_[i] for i in y_batch[:5].tolist()]}")
    logger.success("Dataset ready.")


if __name__ == "__main__":
    app()