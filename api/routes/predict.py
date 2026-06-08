"""
Prediction endpoint.

POST /predict  — accepts a feature dict, returns AQI forecast + model name.

The request accepts any subset of feature columns; missing values are handled
by the LightGBM pipeline's built-in SimpleImputer and by zero-filling for the
MLP (matching the training-time fillna strategy).

Prometheus metrics exported:
  aerocast_predictions_total        — Counter(model_kind)
  aerocast_prediction_latency_seconds — Histogram
"""

from __future__ import annotations

import time

import pandas as pd
from fastapi import APIRouter, HTTPException
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

router = APIRouter(tags=["predict"])

# ── Prometheus metrics ─────────────────────────────────────────────────────

PREDICTIONS_TOTAL = Counter(
    "aerocast_predictions_total",
    "Total number of AQI predictions served",
    ["model_kind"],
)

PREDICTION_LATENCY = Histogram(
    "aerocast_prediction_latency_seconds",
    "End-to-end prediction latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


# ── Request / Response schemas ─────────────────────────────────────────────


class PredictRequest(BaseModel):
    """
    Feature vector for a single AQI forecast.

    Pass whichever columns your model was trained on.  Any column the model
    expects that is missing will be imputed (LightGBM) or zero-filled (MLP).
    """

    features: dict[str, float] = Field(
        ...,
        json_schema_extra={
            "example": {
                "pm25": 12.3,
                "pm10": 20.1,
                "o3": 31.5,
                "no2": 14.0,
                "aqi_lag_1h": 48.0,
                "hour": 14,
                "day_of_week": 2,
            }
        },
    )


class PredictResponse(BaseModel):
    aqi: float = Field(..., description="Predicted AQI value (1-hour horizon)")
    model_kind: str = Field(..., description="Model that produced the prediction")


# ── Route ──────────────────────────────────────────────────────────────────


@router.post("/predict", response_model=PredictResponse, summary="AQI forecast")
def predict(request: PredictRequest) -> PredictResponse:
    """
    Return a 1-hour-ahead AQI forecast given current sensor readings.

    The champion model (MLP or LightGBM) is selected automatically based on
    the W&B registry alias.
    """
    from aerocast.serving.loader import get_model

    model, model_kind = get_model()

    t0 = time.perf_counter()
    try:
        aqi = _run_inference(model, model_kind, request.features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - t0)

    PREDICTIONS_TOTAL.labels(model_kind=model_kind).inc()
    return PredictResponse(aqi=round(aqi, 4), model_kind=model_kind)


# ── Inference helpers ──────────────────────────────────────────────────────


def _run_inference(model: object, model_kind: str, features: dict[str, float]) -> float:
    """Dispatch to the correct inference path based on *model_kind*."""
    if model_kind == "lgbm":
        return _infer_lgbm(model, features)
    return _infer_mlp(model, features)


def _infer_lgbm(pipeline: object, features: dict[str, float]) -> float:
    """LightGBM sklearn Pipeline — handles NaN via its own SimpleImputer."""
    X = pd.DataFrame([features])
    pred = pipeline.predict(X)  # type: ignore[attr-defined]
    return float(pred[0])


def _infer_mlp(model: object, features: dict[str, float]) -> float:
    """PyTorch Lightning MLP — zero-fills missing features before inference."""
    import torch

    X = pd.DataFrame([features]).fillna(0.0)
    tensor = torch.tensor(X.values, dtype=torch.float32)
    model.eval()  # type: ignore[attr-defined]
    with torch.no_grad():
        pred = model(tensor)  # type: ignore[operator]
    return float(pred.squeeze().item())
