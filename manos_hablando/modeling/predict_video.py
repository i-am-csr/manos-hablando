from pathlib import Path

import torch
import numpy as np
from loguru import logger
import typer

from manos_hablando.config import MODELS_DIR
from manos_hablando.dataset import normalize_keypoints
from manos_hablando.dataset_video import load_video_keypoints
from manos_hablando.data.mediapipe_handler import extract_keypoints_video
from manos_hablando.modeling.train_video import LSMVideoTransformer

app = typer.Typer()

MODEL_PATH = MODELS_DIR / "lsm_video_transformer.pt"


def load_model(
    model_path: Path,
    num_classes: int,
    device: torch.device,
) -> LSMVideoTransformer:
    """Load trained LSMVideoTransformer from disk."""
    model = LSMVideoTransformer(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model


def predict_video(
    video_path: Path,
    model: LSMVideoTransformer,
    encoder,
    device: torch.device,
) -> tuple[str, float] | None:
    """
    Predict the LSM letter from a video of a dynamic sign.

    Args:
        video_path: Path to the video file.
        model: Trained LSMVideoTransformer.
        encoder: Fitted LabelEncoder from training.
        device: torch device.

    Returns:
        (letter, confidence) or None if no hand detected.
    """
    result = extract_keypoints_video(video_path)
    if result is None:
        logger.warning(f"No hand detected in {video_path}")
        return None

    # (num_frames, 21, 3) → (num_frames, 63)
    kp = np.array(result["keypoints"], dtype=np.float32)
    kp_flat = kp.reshape(kp.shape[0], -1)

    # Normalize per frame
    kp_norm = normalize_keypoints(kp_flat)

    # Add batch dim: (1, num_frames, 63)
    X_tensor = torch.tensor(kp_norm, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=-1)
        confidence, idx = probs.max(dim=-1)

    letter = encoder.classes_[idx.item()]
    confidence = confidence.item()

    return letter, confidence


def predict_video_top_k(
    video_path: Path,
    model: LSMVideoTransformer,
    encoder,
    device: torch.device,
    k: int = 3,
) -> list[tuple[str, float]] | None:
    """Return top-k predictions with confidence scores for a video."""
    result = extract_keypoints_video(video_path)
    if result is None:
        return None

    kp = np.array(result["keypoints"], dtype=np.float32)
    kp_flat = kp.reshape(kp.shape[0], -1)
    kp_norm = normalize_keypoints(kp_flat)

    X_tensor = torch.tensor(kp_norm, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        top_k = torch.topk(probs, k)

    return [
        (encoder.classes_[idx.item()], score.item())
        for idx, score in zip(top_k.indices, top_k.values)
    ]


@app.command()
def main(
    video_path: Path = typer.Argument(..., help="Path to the hand sign video"),
    model_path: Path = MODEL_PATH,
    top_k: int = typer.Option(3, help="Show top K predictions"),
):
    """Predict LSM letter from a hand sign video."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder from video keypoints dataset
    _, y, encoder = load_video_keypoints()
    num_classes = len(encoder.classes_)

    # Load model
    model = load_model(model_path, num_classes, device)

    # Predict
    result = predict_video_top_k(video_path, model, encoder, device, k=top_k)

    if result is None:
        logger.error("No hand detected in video.")
        raise typer.Exit(1)

    logger.success(f"Predictions for {video_path.name}:")
    for rank, (letter, confidence) in enumerate(result, 1):
        bar = "█" * int(confidence * 20)
        logger.info(f"  #{rank} {letter:4s} {confidence*100:5.1f}% {bar}")


if __name__ == "__main__":
    app()
