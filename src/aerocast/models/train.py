"""
Training entrypoint for AeroCast Phase 3.

Trains both the MLP (primary) and LightGBM (challenger) models,
logs metrics and artifacts to W&B, compares validation RMSE, and
promotes the better model as ``champion`` in the registry.

Usage
-----
Direct (synthetic data for testing)::

    python -m aerocast.models.train --synthetic

From a DVC-tracked parquet::

    python -m aerocast.models.train --data-path data/processed/features.parquet

Importable (for Airflow DAGs)::

    from aerocast.models.train import run_training
    results = run_training(data_path="data/processed/features.parquet")
"""

from __future__ import annotations

import argparse
import logging
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_parquet(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a preprocessed feature parquet and return (X, y)."""
    from aerocast.data.preprocess import make_dataset

    df = pd.read_parquet(path)
    # If already split into X columns + 'target', use directly;
    # otherwise run make_dataset to produce (X, y).
    if "target" in df.columns:
        y = df.pop("target")
        X = df
    else:
        X, y = make_dataset(df)
    return X, y


def _make_synthetic(
    n_samples: int = 2000, n_features: int = 20, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate a synthetic regression dataset that mimics the AeroCast
    feature schema (numeric features, no missing values).

    Used for offline testing when live data is unavailable.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n_samples, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Simple linear target + noise
    coefs = rng.standard_normal(n_features)
    y = pd.Series(X.values @ coefs + rng.standard_normal(n_samples) * 5, name="aqi")
    return X, y


def _time_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple:
    """Chronological train / val / test split (no shuffling)."""
    n = len(X)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train : n_train + n_val], y.iloc[n_train : n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val :], y.iloc[n_train + n_val :]

    logger.info(
        "Split: train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test)
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------


def _train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    wandb_run,
    **lgbm_kwargs,
) -> tuple[object, dict[str, float]]:
    """Fit LightGBM pipeline, log metrics, return (pipeline, val_metrics)."""
    from aerocast.models.lgbm import build_lgbm_pipeline, evaluate

    pipeline = build_lgbm_pipeline(**lgbm_kwargs)
    pipeline.fit(X_train, y_train)

    val_metrics = evaluate(pipeline, X_val, y_val)
    train_metrics = evaluate(pipeline, X_train, y_train)

    wandb_run.log(
        {
            "lgbm/train_rmse": train_metrics["rmse"],
            "lgbm/val_rmse": val_metrics["rmse"],
            "lgbm/val_mae": val_metrics["mae"],
        }
    )
    logger.info("LightGBM val RMSE: %.4f", val_metrics["rmse"])
    return pipeline, val_metrics


def _train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    wandb_run,
    max_epochs: int = 50,
    batch_size: int = 256,
    hidden_dims: tuple[int, ...] = (128, 64, 32),
    dropout: float = 0.2,
    lr: float = 1e-3,
) -> tuple[object, dict[str, float]]:
    """Train MLP with PyTorch Lightning, log metrics, return (model, val_metrics)."""
    import torch
    from lightning import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger

    from aerocast.models.mlp import AQIForecastMLP, make_dataloaders

    # Fill NaNs with column median (MLP cannot handle NaN)
    X_train_filled = X_train.fillna(X_train.median())
    X_val_filled = X_val.fillna(X_train.median())

    train_loader, val_loader = make_dataloaders(
        X_train_filled, y_train, X_val_filled, y_val, batch_size=batch_size
    )

    model = AQIForecastMLP(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
    )

    wandb_logger = WandbLogger(experiment=wandb_run)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
    ]

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader)

    # Extract best val_rmse from callback metrics
    _inf = torch.tensor(float("inf"))
    val_rmse = float(trainer.callback_metrics.get("val_rmse", _inf))
    val_loss = float(trainer.callback_metrics.get("val_loss", _inf))

    wandb_run.log({"mlp/val_rmse": val_rmse, "mlp/val_loss": val_loss})
    logger.info("MLP val RMSE: %.4f", val_rmse)

    return model, trainer, {"rmse": val_rmse, "mse": val_loss}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def run_training(
    data_path: Optional[str | Path] = None,
    synthetic: bool = False,
    mlp_epochs: int = 50,
    mlp_hidden_dims: tuple[int, ...] = (128, 64, 32),
    mlp_dropout: float = 0.2,
    mlp_lr: float = 1e-3,
    batch_size: int = 256,
    wandb_tags: list[str] | None = None,
) -> dict[str, float]:
    """
    Train both models, log to W&B, promote champion.

    Args:
        data_path:       Path to a preprocessed parquet. Mutually exclusive
                         with ``synthetic``.
        synthetic:       If True, generate synthetic data for offline testing.
        mlp_epochs:      Max training epochs for MLP.
        mlp_hidden_dims: Hidden layer widths for MLP.
        mlp_dropout:     Dropout probability for MLP.
        mlp_lr:          Learning rate for MLP Adam.
        batch_size:      Batch size for MLP DataLoader.
        wandb_tags:      Optional tags forwarded to wandb.init.

    Returns:
        Dict with keys ``mlp_val_rmse``, ``lgbm_val_rmse``, ``champion``.
    """
    import wandb

    from aerocast.config import settings
    from aerocast.models.registry import log_model_artifact, promote_champion

    # ── Load data ────────────────────────────────────────────────────────
    if synthetic:
        logger.info("Using synthetic data.")
        X, y = _make_synthetic()
    elif data_path:
        logger.info("Loading data from %s", data_path)
        X, y = _load_parquet(data_path)
    else:
        raise ValueError("Provide either --data-path or --synthetic.")

    X_train, y_train, X_val, y_val, X_test, y_test = _time_split(X, y)

    # ── W&B run ──────────────────────────────────────────────────────────
    run = wandb.init(
        project=settings.wandb_project,
        entity=settings.wandb_entity or None,
        tags=wandb_tags or ["phase-3"],
        config={
            "mlp_epochs": mlp_epochs,
            "mlp_hidden_dims": list(mlp_hidden_dims),
            "mlp_dropout": mlp_dropout,
            "mlp_lr": mlp_lr,
            "batch_size": batch_size,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_features": X_train.shape[1],
            "synthetic": synthetic,
        },
    )

    try:
        # ── Train LightGBM ───────────────────────────────────────────────
        lgbm_pipeline, lgbm_val_metrics = _train_lgbm(
            X_train, y_train, X_val, y_val, run
        )

        # ── Train MLP ────────────────────────────────────────────────────
        mlp_model, mlp_trainer, mlp_val_metrics = _train_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            run,
            max_epochs=mlp_epochs,
            batch_size=batch_size,
            hidden_dims=mlp_hidden_dims,
            dropout=mlp_dropout,
            lr=mlp_lr,
        )

        mlp_rmse = mlp_val_metrics["rmse"]
        lgbm_rmse = lgbm_val_metrics["rmse"]

        # ── Save artifacts ───────────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save LightGBM
            lgbm_path = Path(tmpdir) / "lgbm_pipeline.pkl"
            with open(lgbm_path, "wb") as f:
                pickle.dump(lgbm_pipeline, f)
            lgbm_artifact = log_model_artifact(run, lgbm_path, "lgbm", lgbm_val_metrics)

            # Save MLP (best checkpoint is already on disk via ModelCheckpoint)
            best_ckpt = mlp_trainer.checkpoint_callback.best_model_path
            mlp_artifact = log_model_artifact(run, best_ckpt, "mlp", mlp_val_metrics)

        # ── Champion promotion ────────────────────────────────────────────
        if mlp_rmse <= lgbm_rmse:
            champion_name = "mlp"
            promote_champion(
                run, champion_artifact=mlp_artifact, challenger_artifact=lgbm_artifact
            )
        else:
            champion_name = "lgbm"
            promote_champion(
                run, champion_artifact=lgbm_artifact, challenger_artifact=mlp_artifact
            )

        run.summary["champion"] = champion_name
        run.summary["mlp_val_rmse"] = mlp_rmse
        run.summary["lgbm_val_rmse"] = lgbm_rmse

        logger.info(
            "Champion: %s (MLP RMSE=%.4f, LightGBM RMSE=%.4f)",
            champion_name,
            mlp_rmse,
            lgbm_rmse,
        )

    finally:
        run.finish()

    return {
        "mlp_val_rmse": mlp_rmse,
        "lgbm_val_rmse": lgbm_rmse,
        "champion": champion_name,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AeroCast models.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-path", type=str, help="Path to preprocessed parquet.")
    group.add_argument(
        "--synthetic", action="store_true", help="Use synthetic data (testing)."
    )
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()
    results = run_training(
        data_path=args.data_path,
        synthetic=args.synthetic,
        mlp_epochs=args.mlp_epochs,
        mlp_lr=args.mlp_lr,
        mlp_dropout=args.mlp_dropout,
        batch_size=args.batch_size,
    )
    print("\n── Training complete ──────────────────────")
    for k, v in results.items():
        print(f"  {k}: {v}")
