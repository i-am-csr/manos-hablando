import json
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import typer

from manos_hablando.config import PROCESSED_DATA_DIR
from manos_hablando.dataset import normalize_keypoints
from manos_hablando.dataset_video import (
    LSMVideoDataset,
    collate_video_batch,
    normalize_video_sequences,
)

app = typer.Typer()

FULL_KEYPOINTS_FILE = PROCESSED_DATA_DIR / "full_keypoints.json"

# Static images are stored as 1-frame entries. At training/inference time we
# replicate them to this length so the model sees similar sequence sizes for
# static and dynamic letters (real videos of a held static pose run ~30 frames).
STATIC_REPLICATE_LENGTH = 30


def load_full_keypoints(
    path: Path = FULL_KEYPOINTS_FILE,
    replicate_length: int = STATIC_REPLICATE_LENGTH,
) -> tuple[list[np.ndarray], np.ndarray, LabelEncoder]:
    """
    Load the unified static + dynamic dataset.

    Single-frame entries (source="image") are replicated to `replicate_length`
    so the temporal distribution at training matches what a real video of a
    held static sign produces at inference. Multi-frame entries are kept as-is.

    Returns:
        sequences: list of np.ndarray, each (num_frames, 63)
        y: np.ndarray of shape (n_samples,) — encoded letter labels
        encoder: fitted LabelEncoder
    """
    with open(path) as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} entries from {path}")

    sequences: list[np.ndarray] = []
    labels: list[str] = []
    n_static = 0
    n_dynamic = 0

    for entry in tqdm(dataset, desc="Loading full keypoints"):
        kp = np.array(entry["keypoints"], dtype=np.float32)  # (num_frames, 21, 3)
        kp_flat = kp.reshape(kp.shape[0], -1)                 # (num_frames, 63)

        if entry.get("source") == "image" and kp_flat.shape[0] == 1:
            # Replicate the single frame so static signs have a real temporal
            # extent at training time. np.repeat preserves identity per frame.
            kp_flat = np.repeat(kp_flat, replicate_length, axis=0)
            n_static += 1
        else:
            n_dynamic += 1

        sequences.append(kp_flat)
        labels.append(entry["letter"])

    labels_arr = np.array(labels)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels_arr)

    lengths = [s.shape[0] for s in sequences]
    logger.info(f"Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")
    logger.info(f"Static (replicated): {n_static} | Dynamic: {n_dynamic}")
    logger.info(
        f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
        f"mean={np.mean(lengths):.1f}"
    )

    return sequences, y, encoder


def split_full_dataset(
    sequences: list[np.ndarray],
    y: np.ndarray,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """Stratified split over the unified dataset."""
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

    logger.info(f"Train: {len(seq_train)} samples")
    logger.info(f"Val:   {len(seq_val)} samples")
    logger.info(f"Test:  {len(seq_test)} samples")

    return seq_train, seq_val, seq_test, y_train, y_val, y_test


def get_full_dataloaders(
    batch_size: int = 16,
    num_workers: int = 0,
    replicate_length: int = STATIC_REPLICATE_LENGTH,
) -> tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    """Full pipeline: load → normalize → split → return DataLoaders + encoder."""
    sequences, y, encoder = load_full_keypoints(replicate_length=replicate_length)
    sequences = normalize_video_sequences(sequences)
    seq_train, seq_val, seq_test, y_train, y_val, y_test = split_full_dataset(
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


def prepare_inference_sequence(
    keypoints_per_frame: list,
    is_static_image: bool = False,
    replicate_length: int = STATIC_REPLICATE_LENGTH,
) -> torch.Tensor:
    """
    Helper for predict_full: turn raw per-frame keypoints into a normalized,
    batched tensor ready for LSMVideoTransformer.

    Args:
        keypoints_per_frame: list of (21, 3) lists from MediaPipe.
        is_static_image: if True, replicate the single frame to match training.

    Returns:
        Tensor of shape (1, seq_len, 63), float32.
    """
    kp = np.array(keypoints_per_frame, dtype=np.float32)  # (N, 21, 3)
    kp_flat = kp.reshape(kp.shape[0], -1)                  # (N, 63)
    if is_static_image and kp_flat.shape[0] == 1:
        kp_flat = np.repeat(kp_flat, replicate_length, axis=0)
    kp_norm = normalize_keypoints(kp_flat)                 # (N, 63)
    return torch.tensor(kp_norm, dtype=torch.float32).unsqueeze(0)


@app.command()
def main(
    input_path: Path = FULL_KEYPOINTS_FILE,
    batch_size: int = 16,
):
    """Sanity-check the full dataset pipeline."""
    logger.info("Loading full dataset...")
    sequences, y, encoder = load_full_keypoints(input_path)

    logger.info("Normalizing keypoints...")
    sequences = normalize_video_sequences(sequences)

    logger.info("Splitting dataset...")
    split_full_dataset(sequences, y)

    train_loader, val_loader, test_loader, encoder = get_full_dataloaders(
        batch_size=batch_size,
    )
    X_batch, y_batch, lengths = next(iter(train_loader))
    logger.info(f"Batch X shape: {X_batch.shape}")
    logger.info(f"Batch y shape: {y_batch.shape}")
    logger.info(f"Batch lengths: {lengths.tolist()}")
    logger.info(
        "Sample labels: "
        f"{[encoder.classes_[i] for i in y_batch[:5].tolist()]}"
    )
    logger.success("Full dataset ready.")


if __name__ == "__main__":
    app()