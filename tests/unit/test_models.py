"""
Unit tests for Phase 3 model code.

No live W&B calls — all W&B usage is patched out.
Tests cover:
  - AQIForecastMLP: construction, forward pass, training_step, val_step
  - make_tensor_dataset / make_dataloaders
  - LightGBM pipeline: build, fit, predict, evaluate
  - train._make_synthetic / _time_split
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from aerocast.models.lgbm import build_lgbm_pipeline, evaluate
from aerocast.models.mlp import AQIForecastMLP, make_dataloaders, make_tensor_dataset
from aerocast.models.train import _make_synthetic, _time_split

# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def small_Xy() -> tuple[pd.DataFrame, pd.Series]:
    """Small deterministic (X, y) for fast tests."""
    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(10)]
    X = pd.DataFrame(rng.standard_normal((100, 10)), columns=cols)
    y = pd.Series(rng.standard_normal(100), name="aqi")
    return X, y


@pytest.fixture()
def mlp_model() -> AQIForecastMLP:
    return AQIForecastMLP(input_dim=10, hidden_dims=(32, 16), dropout=0.0, lr=1e-3)


# ── AQIForecastMLP ────────────────────────────────────────────────────────────


class TestAQIForecastMLP:
    def test_forward_shape(self, mlp_model):
        x = torch.randn(8, 10)
        out = mlp_model(x)
        assert out.shape == (8,), f"Expected (8,), got {out.shape}"

    def test_forward_is_float(self, mlp_model):
        x = torch.randn(4, 10)
        out = mlp_model(x)
        assert out.dtype == torch.float32

    def test_training_step_returns_scalar(self, mlp_model):
        batch = (torch.randn(16, 10), torch.randn(16))
        loss = mlp_model.training_step(batch, 0)
        assert loss.ndim == 0  # scalar tensor
        assert loss.item() >= 0

    def test_validation_step_does_not_raise(self, mlp_model):
        batch = (torch.randn(8, 10), torch.randn(8))
        # Should not raise; logs are handled internally
        mlp_model.validation_step(batch, 0)

    def test_configure_optimizers_returns_dict(self, mlp_model):
        result = mlp_model.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_hyperparams_saved(self):
        model = AQIForecastMLP(input_dim=5, hidden_dims=[64], dropout=0.1, lr=5e-4)
        assert model.hparams.input_dim == 5
        assert model.hparams.lr == pytest.approx(5e-4)

    def test_single_sample_no_batchnorm_error(self):
        """BatchNorm1d fails at training time with batch_size=1; eval mode is fine."""
        model = AQIForecastMLP(input_dim=6, hidden_dims=(16,), dropout=0.0)
        model.eval()
        out = model(torch.randn(1, 6))
        assert out.shape == (1,)

    def test_different_hidden_dims(self):
        for dims in [(64,), (128, 64), (256, 128, 64, 32)]:
            model = AQIForecastMLP(input_dim=10, hidden_dims=dims)
            out = model(torch.randn(4, 10))
            assert out.shape == (4,)


# ── make_tensor_dataset / make_dataloaders ─────────────────────────────────


class TestDataLoaderHelpers:
    def test_tensor_dataset_shapes(self, small_Xy):
        X, y = small_Xy
        ds = make_tensor_dataset(X, y)
        x_item, y_item = ds[0]
        assert x_item.shape == (10,)
        assert y_item.shape == ()

    def test_tensor_dataset_dtype(self, small_Xy):
        X, y = small_Xy
        ds = make_tensor_dataset(X, y)
        x_item, y_item = ds[0]
        assert x_item.dtype == torch.float32
        assert y_item.dtype == torch.float32

    def test_make_dataloaders_lengths(self, small_Xy):
        X, y = small_Xy
        X_tr, y_tr = X.iloc[:70], y.iloc[:70]
        X_val, y_val = X.iloc[70:], y.iloc[70:]
        tr_loader, val_loader = make_dataloaders(
            X_tr, y_tr, X_val, y_val, batch_size=16
        )

        total_tr = sum(b[0].shape[0] for b in tr_loader)
        total_val = sum(b[0].shape[0] for b in val_loader)
        assert total_tr == 70
        assert total_val == 30

    def test_accepts_numpy_arrays(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 5))
        y = rng.standard_normal(50)
        ds = make_tensor_dataset(X, y)
        assert len(ds) == 50


# ── LightGBM pipeline ────────────────────────────────────────────────────────


class TestLGBMPipeline:
    def test_build_returns_pipeline(self):
        from sklearn.pipeline import Pipeline

        pipe = build_lgbm_pipeline()
        assert isinstance(pipe, Pipeline)
        assert "model" in pipe.named_steps

    def test_fit_predict(self, small_Xy):
        X, y = small_Xy
        pipe = build_lgbm_pipeline(n_estimators=10)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (100,)

    def test_handles_nan_in_X(self):
        """Imputer step should handle NaN without error."""
        rng = np.random.default_rng(2)
        X = pd.DataFrame(rng.standard_normal((80, 5)), columns=list("abcde"))
        X.iloc[::10, 0] = np.nan  # introduce NaN
        y = pd.Series(rng.standard_normal(80))
        pipe = build_lgbm_pipeline(n_estimators=10)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert np.isfinite(preds).all()

    def test_evaluate_returns_expected_keys(self, small_Xy):
        X, y = small_Xy
        pipe = build_lgbm_pipeline(n_estimators=10)
        pipe.fit(X, y)
        metrics = evaluate(pipe, X, y)
        assert set(metrics.keys()) == {"rmse", "mae", "mse"}

    def test_evaluate_rmse_nonnegative(self, small_Xy):
        X, y = small_Xy
        pipe = build_lgbm_pipeline(n_estimators=10)
        pipe.fit(X, y)
        metrics = evaluate(pipe, X, y)
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    def test_kwargs_override_defaults(self):
        pipe = build_lgbm_pipeline(n_estimators=42)
        assert pipe.named_steps["model"].n_estimators == 42


# ── train helpers ─────────────────────────────────────────────────────────────


class TestTrainHelpers:
    def test_make_synthetic_shapes(self):
        X, y = _make_synthetic(n_samples=200, n_features=15)
        assert X.shape == (200, 15)
        assert y.shape == (200,)

    def test_make_synthetic_no_nan(self):
        X, y = _make_synthetic()
        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_time_split_sizes(self):
        X, y = _make_synthetic(n_samples=1000)
        X_tr, y_tr, X_val, y_val, X_te, y_te = _time_split(X, y)
        assert len(X_tr) + len(X_val) + len(X_te) == 1000
        assert len(X_tr) > len(X_val)  # train is largest

    def test_time_split_no_overlap(self):
        """Indices must be disjoint and in chronological order."""
        X, y = _make_synthetic(n_samples=500)
        X_tr, _, X_val, _, X_te, _ = _time_split(X, y)
        assert X_tr.index[-1] < X_val.index[0]
        assert X_val.index[-1] < X_te.index[0]

    def test_time_split_small_dataset(self):
        """Should not crash on small datasets."""
        X, y = _make_synthetic(n_samples=10, n_features=3)
        parts = _time_split(X, y)
        total = sum(len(p) for p in parts[::2])  # X pieces
        assert total == 10
