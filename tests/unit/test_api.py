"""
Unit tests for the AeroCast FastAPI application.

The W&B model loader is patched so tests run offline without downloading
any artifacts.  Both MLP and LightGBM inference paths are exercised via
lightweight fakes.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fake models
# ---------------------------------------------------------------------------


class _FakeLGBM:
    """Minimal sklearn-pipeline lookalike that returns a constant prediction."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([42.0] * len(X))


class _FakeMLP:
    """Minimal Lightning-module lookalike."""

    def eval(self):
        return self

    def __call__(self, tensor):
        import torch

        return torch.tensor([[37.5]])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def lgbm_client():
    """TestClient with a mocked LightGBM champion loaded."""
    with (
        patch("aerocast.serving.loader._model", _FakeLGBM()),
        patch("aerocast.serving.loader._model_kind", "lgbm"),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client


@pytest.fixture()
def mlp_client():
    """TestClient with a mocked MLP champion loaded."""
    with (
        patch("aerocast.serving.loader._model", _FakeMLP()),
        patch("aerocast.serving.loader._model_kind", "mlp"),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client


@pytest.fixture()
def unloaded_client():
    """TestClient with no model loaded (simulates cold startup)."""
    with (
        patch("aerocast.serving.loader._model", None),
        patch("aerocast.serving.loader._model_kind", None),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, lgbm_client):
        r = lgbm_client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_health_always_up_regardless_of_model(self, unloaded_client):
        r = unloaded_client.get("/health")
        assert r.status_code == 200


class TestReadyEndpoint:
    def test_ready_200_when_model_loaded(self, lgbm_client):
        r = lgbm_client.get("/ready")
        assert r.status_code == 200
        assert r.json() == {"status": "ready"}

    def test_ready_503_when_model_not_loaded(self, unloaded_client):
        r = unloaded_client.get("/ready")
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------

_FEATURES = {
    "pm25": 12.3,
    "pm10": 20.1,
    "o3": 31.5,
    "aqi_lag_1h": 48.0,
    "hour": 14,
    "day_of_week": 2,
}


class TestPredictLGBM:
    def test_predict_returns_200(self, lgbm_client):
        r = lgbm_client.post("/predict", json={"features": _FEATURES})
        assert r.status_code == 200

    def test_predict_response_schema(self, lgbm_client):
        r = lgbm_client.post("/predict", json={"features": _FEATURES})
        body = r.json()
        assert "aqi" in body
        assert "model_kind" in body

    def test_predict_model_kind_lgbm(self, lgbm_client):
        r = lgbm_client.post("/predict", json={"features": _FEATURES})
        assert r.json()["model_kind"] == "lgbm"

    def test_predict_aqi_value(self, lgbm_client):
        r = lgbm_client.post("/predict", json={"features": _FEATURES})
        assert r.json()["aqi"] == pytest.approx(42.0)


class TestPredictMLP:
    def test_predict_returns_200(self, mlp_client):
        r = mlp_client.post("/predict", json={"features": _FEATURES})
        assert r.status_code == 200

    def test_predict_model_kind_mlp(self, mlp_client):
        r = mlp_client.post("/predict", json={"features": _FEATURES})
        assert r.json()["model_kind"] == "mlp"

    def test_predict_aqi_value(self, mlp_client):
        r = mlp_client.post("/predict", json={"features": _FEATURES})
        assert r.json()["aqi"] == pytest.approx(37.5)


class TestPredictValidation:
    def test_missing_features_key_returns_422(self, lgbm_client):
        r = lgbm_client.post("/predict", json={})
        assert r.status_code == 422

    def test_empty_features_dict_still_works(self, lgbm_client):
        """LightGBM pipeline handles empty input via SimpleImputer."""
        r = lgbm_client.post("/predict", json={"features": {}})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, lgbm_client):
        r = lgbm_client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_content_type(self, lgbm_client):
        r = lgbm_client.get("/metrics")
        assert "text/plain" in r.headers["content-type"]

    def test_metrics_contains_prediction_counter(self, lgbm_client):
        lgbm_client.post("/predict", json={"features": _FEATURES})
        r = lgbm_client.get("/metrics")
        assert "aerocast_predictions_total" in r.text
