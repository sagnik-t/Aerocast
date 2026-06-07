"""
PyTorch Lightning MLP — primary AQI forecast model.

Architecture: fully-connected feed-forward network with configurable
hidden layers, BatchNorm, Dropout, and ReLU activations.

W&B metrics logged per step (train_loss, val_loss, val_rmse).
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class AQIForecastMLP(L.LightningModule):
    """
    Feed-forward MLP for AQI regression.

    Args:
        input_dim:   Number of input features (must match X.shape[1]).
        hidden_dims: Sequence of hidden layer widths.
        dropout:     Dropout probability applied after each hidden layer.
        lr:          Learning rate for Adam optimiser.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 64, 32),
        dropout: float = 0.2,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

    # ── forward ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, input_dim) → (B,)
        return self.net(x).squeeze(-1)

    # ── steps ───────────────────────────────────────────────────────────

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        rmse = torch.sqrt(loss)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        rmse = torch.sqrt(loss)
        self.log("test_loss", loss)
        self.log("test_rmse", rmse)

    # ── optimiser ───────────────────────────────────────────────────────

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------


def make_tensor_dataset(X, y) -> TensorDataset:
    """Convert pandas X/y (or numpy arrays) to a TensorDataset."""
    import numpy as np

    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)

    X_tensor = torch.tensor(X_arr, dtype=torch.float32)
    y_tensor = torch.tensor(y_arr, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


def make_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) from pandas/numpy arrays."""
    train_ds = make_tensor_dataset(X_train, y_train)
    val_ds = make_tensor_dataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader
