from pathlib import Path

import math
import torch
import torch.nn as nn
import mlflow
from loguru import logger
from tqdm import tqdm
import typer

from manos_hablando.config import MODELS_DIR
from manos_hablando.dataset_video import get_video_dataloaders

app = typer.Typer()


# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Injects position information into the sequence."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
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


class LSMVideoTransformer(nn.Module):
    """
    Transformer model for LSM dynamic hand sign classification.

    Unlike the static LSMTransformer (which adds a dummy sequence dim),
    this model receives a real temporal sequence — each frame is a token.

    Input:  (batch, seq_len, 63) — sequence of flattened keypoints per frame
    Output: (batch, num_classes) — letter logits
    """

    def __init__(
        self,
        input_dim: int = 63,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        num_classes: int = 6,
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
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 63)
            src_key_padding_mask: (batch, seq_len) — True for padded positions
        """
        x = self.input_proj(x)      # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.transformer(
            x, src_key_padding_mask=src_key_padding_mask,
        )
        # Mean-pool over non-padded positions
        if src_key_padding_mask is not None:
            mask = ~src_key_padding_mask  # True for real positions
            mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.classifier(x)   # (batch, num_classes)


# ─────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────

def _make_padding_mask(
    lengths: torch.Tensor, max_len: int,
) -> torch.Tensor:
    """Create a padding mask: True for padded positions."""
    batch_size = lengths.size(0)
    # (batch, max_len) — True where position >= length (i.e. padding)
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return arange >= lengths.unsqueeze(1)


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
    total_samples = 0

    for X_batch, y_batch, lengths in tqdm(loader, desc="Training", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        lengths = lengths.to(device)

        mask = _make_padding_mask(lengths, X_batch.size(1))

        optimizer.zero_grad()
        logits = model(X_batch, src_key_padding_mask=mask)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == y_batch).sum().item()
        total_samples += y_batch.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total_samples
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
    total_samples = 0

    for X_batch, y_batch, lengths in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        lengths = lengths.to(device)

        mask = _make_padding_mask(lengths, X_batch.size(1))

        logits = model(X_batch, src_key_padding_mask=mask)
        total_loss += criterion(logits, y_batch).item()
        correct += (logits.argmax(dim=-1) == y_batch).sum().item()
        total_samples += y_batch.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total_samples
    return avg_loss, accuracy


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

@app.command()
def main(
    model_path: Path = MODELS_DIR / "lsm_video_transformer.pt",
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 3,
    dropout: float = 0.1,
):
    """Train the LSM Video Transformer model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    logger.info("Loading video data...")
    train_loader, val_loader, test_loader, encoder = get_video_dataloaders(
        batch_size=batch_size,
    )
    num_classes = len(encoder.classes_)
    logger.info(f"Classes: {num_classes} — {list(encoder.classes_)}")

    # Model
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

    # MLflow
    mlflow.set_experiment("lsm-video-transformer")

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

        # Final evaluation on test set
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


if __name__ == "__main__":
    app()
