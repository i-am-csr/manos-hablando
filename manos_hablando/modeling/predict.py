from pathlib import Path

import torch
import torch.nn as nn
import math
import numpy as np
from loguru import logger
import typer

from manos_hablando.config import MODELS_DIR
from manos_hablando.dataset import load_keypoints, normalize_keypoints
from manos_hablando.data.mediapipe_handler import extract_keypoints
from manos_hablando.modeling.train import LSMTransformer

app = typer.Typer()

MODEL_PATH = MODELS_DIR / "lsm_transformer.pt"


def load_model(
    model_path: Path,
    num_classes: int,
    device: torch.device,
) -> LSMTransformer:
    """Load trained LSMTransformer from disk."""
    model = LSMTransformer(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model


def predict_image(
    image_path: Path,
    model: LSMTransformer,
    encoder,
    device: torch.device,
) -> tuple[str, float] | None:
    """
    Predict the LSM letter from a single image.

    Args:
        image_path: Path to the image file.
        model: Trained LSMTransformer.
        encoder: Fitted LabelEncoder from training.
        device: torch device.

    Returns:
        (letter, confidence) or None if no hand detected.
    """
    # Extract keypoints
    result = extract_keypoints(image_path)
    if result is None:
        logger.warning(f"No hand detected in {image_path}")
        return None

    # Normalize
    X = np.array(result["keypoints"], dtype=np.float32).flatten()    # (63,)
    X = normalize_keypoints(X.reshape(1, -1))              # (1, 63)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Inference
    with torch.no_grad():
        logits = model(X_tensor)                           # (1, num_classes)
        probs = torch.softmax(logits, dim=-1)
        confidence, idx = probs.max(dim=-1)

    letter = encoder.classes_[idx.item()]
    confidence = confidence.item()

    return letter, confidence


def predict_top_k(
    image_path: Path,
    model: LSMTransformer,
    encoder,
    device: torch.device,
    k: int = 3,
) -> list[tuple[str, float]] | None:
    """
    Return top-k predictions with confidence scores.

    Useful for ambiguous signs.
    """
    result = extract_keypoints(image_path)
    if result is None:
        return None

    X = np.array(result["keypoints"], dtype=np.float32).flatten()
    X = normalize_keypoints(X.reshape(1, -1))
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

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
    image_path: Path = typer.Argument(..., help="Path to the hand sign image"),
    model_path: Path = MODEL_PATH,
    top_k: int = typer.Option(3, help="Show top K predictions"),
):
    """Predict LSM letter from a hand sign image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder from keypoints dataset
    _, y, encoder = load_keypoints()
    num_classes = len(encoder.classes_)

    # Load model
    model = load_model(model_path, num_classes, device)

    # Predict
    result = predict_top_k(image_path, model, encoder, device, k=top_k)

    if result is None:
        logger.error("No hand detected in image.")
        raise typer.Exit(1)

    logger.success(f"Predictions for {image_path.name}:")
    for rank, (letter, confidence) in enumerate(result, 1):
        bar = "█" * int(confidence * 20)
        logger.info(f"  #{rank} {letter:4s} {confidence*100:5.1f}% {bar}")


if __name__ == "__main__":
    app()