"""
Load the holistic words dataset (face + pose + hands features per frame)
and expose PyTorch DataLoaders for train / val / test.

The manifest lives at data/processed/holistic_keypoints.json and points at
per-video .npz files. We reuse the LSMVideoDataset + collate_video_batch
infrastructure from the dynamic-letters pipeline; the only difference is
the per-frame feature dimension (1692 vs 63).
"""

import json
from pathlib import Path

from loguru import logger
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import typer

from manos_hablando.config import PROCESSED_DATA_DIR
from manos_hablando.data.holistic_handler import (
    FACE_DIMS,
    FACE_KEEP_INDICES,
    FACE_LANDMARKS,
    FACE_OFFSET,
    FEATURE_DIM,
    HAND_DIMS,
    HAND_LANDMARKS,
    LEFT_HAND_OFFSET,
    LEFT_SHOULDER_IDX,
    POSE_DIMS,
    POSE_LANDMARKS,
    POSE_OFFSET,
    RIGHT_HAND_OFFSET,
    RIGHT_SHOULDER_IDX,
    TRIMMED_FACE_DIMS,
)
from manos_hablando.dataset_video import LSMVideoDataset, collate_video_batch

app = typer.Typer()

MANIFEST_FILE = PROCESSED_DATA_DIR / "holistic_keypoints.json"
HOLISTIC_DIR = PROCESSED_DATA_DIR / "holistic"


def load_holistic_keypoints(
    manifest_path: Path = MANIFEST_FILE,
) -> tuple[list[np.ndarray], np.ndarray, LabelEncoder]:
    """
    Load all per-video feature tensors and labels from the manifest.

    Returns:
        sequences: list of np.ndarray, each (num_frames, FEATURE_DIM)
        y: np.ndarray of shape (n_samples,) — encoded word labels
        encoder: fitted LabelEncoder
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    logger.info(f"Loaded manifest with {len(manifest)} entries from {manifest_path}")

    sequences: list[np.ndarray] = []
    labels: list[str] = []

    base_dir = manifest_path.parent
    for entry in tqdm(manifest, desc="Loading holistic features"):
        npz_path = base_dir / entry["npz"]
        with np.load(npz_path) as npz:
            features = npz["features"].astype(np.float32)
        if features.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"{npz_path}: feature dim {features.shape[1]} != expected {FEATURE_DIM}. "
                "Re-extract with the current holistic_handler."
            )
        sequences.append(features)
        labels.append(entry["word"])

    labels_arr = np.array(labels)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels_arr)

    lengths = [s.shape[0] for s in sequences]
    logger.info(f"Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")
    logger.info(f"Samples: {len(sequences)}  |  feature_dim: {FEATURE_DIM}")
    logger.info(
        f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
        f"mean={np.mean(lengths):.1f}"
    )

    return sequences, y, encoder


def normalize_holistic_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Shoulder-centered, scale-invariant normalization for one holistic sequence.

    For every frame:
      - anchor = midpoint of pose LEFT_SHOULDER (11) and RIGHT_SHOULDER (12)
      - scale  = euclidean distance between those two shoulders
      - subtract anchor from all (x, y) coords (pose, face, hands)
      - divide all (x, y, z) coords by scale
      - leave pose visibility values untouched
      - hands that were absent (all-zero block) stay zero
      - frames where shoulders are missing (scale ≈ 0) are zeroed entirely

    This is the standard normalization used in holistic sign-language papers:
    it removes the effect of where the signer stands in the frame and how
    far they are from the camera, so the model sees shape rather than
    absolute pixel positions.
    """
    if seq.size == 0:
        return seq.copy()

    N = seq.shape[0]
    seq = seq.astype(np.float32)

    pose = seq[:, POSE_OFFSET : POSE_OFFSET + POSE_DIMS].reshape(N, POSE_LANDMARKS, 4).copy()
    face = seq[:, FACE_OFFSET : FACE_OFFSET + FACE_DIMS].reshape(N, FACE_LANDMARKS, 3).copy()
    lh = seq[:, LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + HAND_DIMS].reshape(N, HAND_LANDMARKS, 3).copy()
    rh = seq[:, RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + HAND_DIMS].reshape(N, HAND_LANDMARKS, 3).copy()

    ls_xy = pose[:, LEFT_SHOULDER_IDX, :2]
    rs_xy = pose[:, RIGHT_SHOULDER_IDX, :2]
    mid_xy = (ls_xy + rs_xy) / 2.0
    scale = np.linalg.norm(ls_xy - rs_xy, axis=1)
    valid = scale > 1e-4

    lh_present = (lh != 0).any(axis=(1, 2))
    rh_present = (rh != 0).any(axis=(1, 2))

    safe_scale = np.where(valid, scale, 1.0).astype(np.float32)
    scale_inv = (1.0 / safe_scale).reshape(N, 1, 1)
    mid_b = mid_xy.reshape(N, 1, 2).astype(np.float32)

    pose[:, :, :2] = (pose[:, :, :2] - mid_b) * scale_inv
    pose[:, :, 2:3] = pose[:, :, 2:3] * scale_inv

    face[:, :, :2] = (face[:, :, :2] - mid_b) * scale_inv
    face[:, :, 2:3] = face[:, :, 2:3] * scale_inv

    if lh_present.any():
        m = lh_present
        lh[m, :, :2] = (lh[m, :, :2] - mid_b[m]) * scale_inv[m]
        lh[m, :, 2:3] = lh[m, :, 2:3] * scale_inv[m]
    if rh_present.any():
        m = rh_present
        rh[m, :, :2] = (rh[m, :, :2] - mid_b[m]) * scale_inv[m]
        rh[m, :, 2:3] = rh[m, :, 2:3] * scale_inv[m]

    out = np.zeros_like(seq)
    out[:, POSE_OFFSET : POSE_OFFSET + POSE_DIMS] = pose.reshape(N, POSE_DIMS)
    out[:, FACE_OFFSET : FACE_OFFSET + FACE_DIMS] = face.reshape(N, FACE_DIMS)
    out[:, LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + HAND_DIMS] = lh.reshape(N, HAND_DIMS)
    out[:, RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + HAND_DIMS] = rh.reshape(N, HAND_DIMS)

    if not valid.all():
        out[~valid] = 0

    return out


def normalize_holistic_sequences(sequences: list[np.ndarray]) -> list[np.ndarray]:
    """Per-sequence wrapper for normalize_holistic_sequence."""
    return [normalize_holistic_sequence(s) for s in sequences]


def trim_holistic_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Drop the unused face landmarks and re-pack into the trimmed layout
    `[pose | trimmed_face | left_hand | right_hand]` of shape
    `(N, TRIMMED_FEATURE_DIM)`. Pose, hand, and selected face slots are
    preserved verbatim; the full mesh just shrinks via `FACE_KEEP_INDICES`.
    Apply *after* `normalize_holistic_sequence` since normalization uses
    the original 478-point face layout.
    """
    if seq.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} dims, got {seq.shape[1]}")
    n = seq.shape[0]
    pose = seq[:, POSE_OFFSET : POSE_OFFSET + POSE_DIMS]
    face_full = seq[:, FACE_OFFSET : FACE_OFFSET + FACE_DIMS].reshape(n, FACE_LANDMARKS, 3)
    face_kept = face_full[:, FACE_KEEP_INDICES, :].reshape(n, TRIMMED_FACE_DIMS)
    lh = seq[:, LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + HAND_DIMS]
    rh = seq[:, RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + HAND_DIMS]
    return np.concatenate([pose, face_kept, lh, rh], axis=1).astype(np.float32)


def trim_holistic_sequences(sequences: list[np.ndarray]) -> list[np.ndarray]:
    """Per-sequence wrapper for trim_holistic_sequence."""
    return [trim_holistic_sequence(s) for s in sequences]


def split_holistic_dataset(
    sequences: list[np.ndarray],
    y: np.ndarray,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """Stratified split. Returns seq_train, seq_val, seq_test, y_train, y_val, y_test."""
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


def get_holistic_dataloaders(
    batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    """Full pipeline: load → normalize → trim face → split → return DataLoaders."""
    sequences, y, encoder = load_holistic_keypoints()
    sequences = normalize_holistic_sequences(sequences)
    sequences = trim_holistic_sequences(sequences)
    seq_train, seq_val, seq_test, y_train, y_val, y_test = split_holistic_dataset(
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


def prepare_inference_sequence(features: np.ndarray) -> torch.Tensor:
    """
    Helper for predict_holistic: run the same load-time transforms used
    at training (shoulder-centered normalize → face-mesh trim), then
    return as a batched float32 tensor of shape (1, N, TRIMMED_FEATURE_DIM).
    """
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected shape (N, {FEATURE_DIM}), got {features.shape}"
        )
    features = normalize_holistic_sequence(features)
    features = trim_holistic_sequence(features)
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


@app.command()
def main(
    manifest_path: Path = MANIFEST_FILE,
    batch_size: int = 8,
) -> None:
    """Sanity-check the holistic dataset pipeline."""
    logger.info("Loading holistic dataset...")
    sequences, y, encoder = load_holistic_keypoints(manifest_path)

    logger.info("Splitting dataset...")
    split_holistic_dataset(sequences, y)

    train_loader, val_loader, test_loader, encoder = get_holistic_dataloaders(
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
    logger.success("Holistic dataset ready.")


if __name__ == "__main__":
    app()
