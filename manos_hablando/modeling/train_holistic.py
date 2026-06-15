"""
Train the LSMVideoTransformer on the holistic (face + pose + hands)
words dataset.

The architecture is unchanged from train_video.py — only `input_dim` grows
from 63 (one hand × 21 × 3) to 1692 (pose + face + 2 hands). Batch size
defaults are lower than train_full because each sample is ~27× larger.
"""

from pathlib import Path

from loguru import logger
import mlflow
import torch
import torch.nn as nn
import typer

from manos_hablando.config import MODELS_DIR
from manos_hablando.data.holistic_handler import TRIMMED_FEATURE_DIM
from manos_hablando.dataset_holistic import get_holistic_dataloaders
from manos_hablando.modeling.train_video import (
    LSMVideoTransformer,
    _make_padding_mask,
    evaluate,
    train_epoch,
)

app = typer.Typer()


@app.command()
def main(
    model_path: Path = MODELS_DIR / "lsm_holistic_transformer.pt",
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.3,
):
    """Train the LSM Holistic Transformer on the words dataset."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading holistic data...")
    train_loader, val_loader, test_loader, encoder = get_holistic_dataloaders(
        batch_size=batch_size,
    )
    num_classes = len(encoder.classes_)
    logger.info(f"Classes: {num_classes} — {list(encoder.classes_)}")

    model = LSMVideoTransformer(
        input_dim=TRIMMED_FEATURE_DIM,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )

    mlflow.set_experiment("lsm-holistic-transformer")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "input_dim": TRIMMED_FEATURE_DIM,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout,
            "num_classes": num_classes,
        })

        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device,
            )
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device,
            )
            scheduler.step(val_loss)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_path)
                logger.info(f"Best model saved → {model_path}")

        model.load_state_dict(torch.load(model_path))
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device,
        )

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

        mlflow.pytorch.log_model(model, "model")

        logger.success(
            f"Training complete. "
            f"Best val_acc={best_val_acc:.4f} | test_acc={test_acc:.4f}"
        )


__all__ = ["LSMVideoTransformer", "_make_padding_mask", "evaluate", "train_epoch", "main"]


if __name__ == "__main__":
    app()
