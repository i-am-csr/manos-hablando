from pathlib import Path

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from loguru import logger
from tqdm import tqdm
import typer

from manos_hablando.config import PROCESSED_DATA_DIR
from manos_hablando.dataset import normalize_keypoints

app = typer.Typer()

VIDEO_KEYPOINTS_FILE = PROCESSED_DATA_DIR / "video_keypoints.json"


def load_video_keypoints(
    path: Path = VIDEO_KEYPOINTS_FILE,
) -> tuple[list[np.ndarray], np.ndarray, LabelEncoder]:
    """
    Load video keypoints from JSON file.

    Returns:
        sequences: list of np.ndarray, each of shape (num_frames, 63)
        y: np.ndarray of shape (n_samples,) — encoded letter labels
        encoder: fitted LabelEncoder
    """
    with open(path) as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} video samples from {path}")

    sequences = []
    labels = []

    for entry in tqdm(dataset, desc="Loading video keypoints"):
        # keypoints shape: (num_frames, 21, 3)
        kp = np.array(entry["keypoints"], dtype=np.float32)
        # Flatten each frame: (num_frames, 63)
        kp_flat = kp.reshape(kp.shape[0], -1)
        sequences.append(kp_flat)
        labels.append(entry["letter"])

    labels = np.array(labels)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    lengths = [s.shape[0] for s in sequences]
    logger.info(f"Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")
    logger.info(f"Samples: {len(sequences)}")
    logger.info(
        f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
        f"mean={np.mean(lengths):.1f}"
    )

    return sequences, y, encoder


def normalize_video_sequences(
    sequences: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Normalize keypoints per frame using the same wrist-centering strategy
    as the static pipeline.

    Args:
        sequences: list of arrays, each (num_frames, 63)

    Returns:
        list of normalized arrays, each (num_frames, 63)
    """
    normalized = []
    for seq in sequences:
        # normalize_keypoints expects (n_samples, 63), treats each row independently
        normalized.append(normalize_keypoints(seq))
    return normalized


def split_video_dataset(
    sequences: list[np.ndarray],
    y: np.ndarray,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """
    Split video dataset into train, validation and test sets.
    Stratified by class.

    Returns:
        seq_train, seq_val, seq_test, y_train, y_val, y_test
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    indices = np.arange(len(sequences))

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, y,
        test_size=(val_size + test_size),
        stratify=y,
        random_state=random_state,
    )

    val_ratio = val_size / (val_size + test_size)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=random_state,
    )

    seq_train = [sequences[i] for i in idx_train]
    seq_val = [sequences[i] for i in idx_val]
    seq_test = [sequences[i] for i in idx_test]

    logger.info(f"Train: {len(seq_train)} videos")
    logger.info(f"Val:   {len(seq_val)} videos")
    logger.info(f"Test:  {len(seq_test)} videos")

    return seq_train, seq_val, seq_test, y_train, y_val, y_test


class LSMVideoDataset(Dataset):
    """PyTorch Dataset for variable-length video keypoint sequences."""

    def __init__(self, sequences: list[np.ndarray], labels: np.ndarray):
        self.sequences = [
            torch.tensor(s, dtype=torch.float32) for s in sequences
        ]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def collate_video_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate variable-length sequences into a padded batch.

    Returns:
        X_padded: (batch, max_seq_len, 63) — zero-padded sequences
        y: (batch,) — labels
        lengths: (batch,) — original sequence lengths (for masking)
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in sequences])
    X_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    y = torch.stack(labels)
    return X_padded, y, lengths


def get_video_dataloaders(
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    """
    Full pipeline: load → normalize → split → return DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, encoder
    """
    sequences, y, encoder = load_video_keypoints()
    sequences = normalize_video_sequences(sequences)
    seq_train, seq_val, seq_test, y_train, y_val, y_test = split_video_dataset(
        sequences, y,
    )

    train_loader = DataLoader(
        LSMVideoDataset(seq_train, y_train),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_video_batch,
    )
    val_loader = DataLoader(
        LSMVideoDataset(seq_val, y_val),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_video_batch,
    )
    test_loader = DataLoader(
        LSMVideoDataset(seq_test, y_test),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_video_batch,
    )

    return train_loader, val_loader, test_loader, encoder


@app.command()
def main(
    input_path: Path = VIDEO_KEYPOINTS_FILE,
    batch_size: int = 16,
):
    """Load, normalize and split the LSM video keypoints dataset."""
    logger.info("Loading video dataset...")
    sequences, y, encoder = load_video_keypoints(input_path)

    logger.info("Normalizing keypoints...")
    sequences = normalize_video_sequences(sequences)

    logger.info("Splitting dataset...")
    seq_train, seq_val, seq_test, y_train, y_val, y_test = split_video_dataset(
        sequences, y,
    )

    # Sanity check
    train_loader, val_loader, test_loader, encoder = get_video_dataloaders(
        batch_size=batch_size,
    )
    X_batch, y_batch, lengths = next(iter(train_loader))

    logger.info(f"Batch X shape: {X_batch.shape}")
    logger.info(f"Batch y shape: {y_batch.shape}")
    logger.info(f"Batch lengths: {lengths.tolist()}")
    logger.info(
        f"Sample labels: "
        f"{[encoder.classes_[i] for i in y_batch[:5].tolist()]}"
    )
    logger.success("Video dataset ready.")


if __name__ == "__main__":
    app()
