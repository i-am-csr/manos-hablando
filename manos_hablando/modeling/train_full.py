from pathlib import Path

import mlflow
from loguru import logger
import torch
import torch.nn as nn
import typer

from manos_hablando.config import MODELS_DIR
from manos_hablando.dataset_full import get_full_dataloaders
from manos_hablando.modeling.train_video import (
    LSMVideoTransformer,
    _make_padding_mask,
    evaluate,
    train_epoch,
)

app = typer.Typer()


@app.command()
def main(
    model_path: Path = MODELS_DIR / "lsm_full_transformer.pt",
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 3,
    dropout: float = 0.1,
):
    """Train the unified LSM Video Transformer on all 30 letters."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading full data...")
    train_loader, val_loader, test_loader, encoder = get_full_dataloaders(
        batch_size=batch_size,
    )
    num_classes = len(encoder.classes_)
    logger.info(f"Classes: {num_classes} — {list(encoder.classes_)}")

    model = LSMVideoTransformer(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )

    mlflow.set_experiment("lsm-full-transformer")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
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


# Re-exported so module-level imports stay consistent with train_video.py
__all__ = ["LSMVideoTransformer", "_make_padding_mask", "evaluate", "train_epoch", "main"]


if __name__ == "__main__":
    app()
