"""
Classify a single word video with the holistic transformer.

Each clip is assumed to contain one sign — no segmentation is performed
(unlike predict_full, which splits multi-sign fingerspelling). The whole
video's per-frame holistic features feed the model once and the top-K
predicted words are printed.
"""

from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table
import torch
import typer

from manos_hablando.config import MODELS_DIR
from manos_hablando.data.holistic_handler import TRIMMED_FEATURE_DIM, extract_holistic_video
from manos_hablando.dataset_holistic import (
    load_holistic_keypoints,
    prepare_inference_sequence,
)
from manos_hablando.modeling.train_video import LSMVideoTransformer

app = typer.Typer()
console = Console()

MODEL_PATH = MODELS_DIR / "lsm_holistic_transformer.pt"


def load_model(
    model_path: Path,
    num_classes: int,
    device: torch.device,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
) -> LSMVideoTransformer:
    """Build the architecture and load trained weights."""
    model = LSMVideoTransformer(
        input_dim=TRIMMED_FEATURE_DIM,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model


@app.command()
def main(
    video_path: Path = typer.Argument(
        ...,
        help="Path to a video containing a single signed word.",
    ),
    model_path: Path = MODEL_PATH,
    top_k: int = typer.Option(5, help="Number of candidate words to show."),
    d_model: int = typer.Option(128, help="Must match training."),
    nhead: int = typer.Option(4, help="Must match training."),
    num_layers: int = typer.Option(2, help="Must match training."),
):
    """
    Predict the LSM word in a single video using the holistic transformer.

    Example:
        uv run python -m manos_hablando.modeling.predict_holistic clip.mp4 2>/dev/null
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, encoder = load_holistic_keypoints()
    num_classes = len(encoder.classes_)
    model = load_model(
        model_path, num_classes, device,
        d_model=d_model, nhead=nhead, num_layers=num_layers,
    )

    logger.info(f"Extracting holistic features from {video_path.name}...")
    result = extract_holistic_video(video_path)
    if result is None:
        logger.error("No pose or hands detected anywhere in the video.")
        raise typer.Exit(1)

    features = result["features"]
    mc = result["modality_counts"]
    logger.info(
        f"Frames: {result['total_frames']} "
        f"(useful: {result['detected_frames']}) "
        f"@ {result['fps']:.1f} fps  |  "
        f"face={mc['face']} pose={mc['pose']} hands={mc['hands']}"
    )

    X = prepare_inference_sequence(features).to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=-1)[0]
        k = min(top_k, probs.shape[0])
        topk = torch.topk(probs, k)

    table = Table(title=f"Top-{k} predictions for {video_path.name}")
    table.add_column("Rank", style="cyan")
    table.add_column("Word", style="green")
    table.add_column("Confidence", style="yellow")
    for rank, (idx, score) in enumerate(zip(topk.indices, topk.values), 1):
        table.add_row(
            str(rank),
            encoder.classes_[idx.item()],
            f"{score.item() * 100:.1f}%",
        )
    console.print(table)

    best_idx = topk.indices[0].item()
    best_score = topk.values[0].item()
    logger.success(
        f"Predicted word: {encoder.classes_[best_idx]} "
        f"({best_score * 100:.1f}% confidence)"
    )


if __name__ == "__main__":
    app()
