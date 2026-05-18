from pathlib import Path

import math
import torch
import torch.nn as nn
import mlflow
from loguru import logger
from tqdm import tqdm
import typer

from manos_hablando.config import MODELS_DIR
from manos_hablando.dataset import get_dataloaders

app = typer.Typer()


# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Injects position information into the sequence."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LSMTransformer(nn.Module):
    """
    Transformer model for LSM hand sign classification.

    Input:  (batch, 63) — flattened 21 keypoints x 3 coords
    Output: (batch, num_classes) — letter logits
    """

    def __init__(
        self,
        input_dim: int = 63,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        num_classes: int = 26,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 63)
        x = x.unsqueeze(1)          # (batch, 1, 63) — add sequence dim
        x = self.input_proj(x)      # (batch, 1, d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)           # (batch, d_model)
        return self.classifier(x)   # (batch, num_classes)


# ─────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0

    for X_batch, y_batch in tqdm(loader, desc="Training", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == y_batch).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a dataloader. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item()
        correct += (logits.argmax(dim=-1) == y_batch).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

@app.command()
def main(
    model_path: Path = MODELS_DIR / "lsm_transformer.pt",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 3,
    dropout: float = 0.1,
):
    """Train the LSM Transformer model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, encoder = get_dataloaders(batch_size=batch_size)
    num_classes = len(encoder.classes_)
    logger.info(f"Classes: {num_classes}")

    # Model
    model = LSMTransformer(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # MLflow
    mlflow.set_experiment("lsm-transformer")

    with mlflow.start_run():
        # Log hyperparameters
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
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            # Log metrics
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

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_path)
                logger.info(f"Best model saved → {model_path}")

        # Final evaluation on test set
        model.load_state_dict(torch.load(model_path))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

        mlflow.pytorch.log_model(model, "model")

        logger.success(f"Training complete. Best val_acc={best_val_acc:.4f} | test_acc={test_acc:.4f}")


if __name__ == "__main__":
    app()