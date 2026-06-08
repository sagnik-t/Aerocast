"""
Health and readiness endpoints.

GET /health  — liveness probe  (always 200 if the process is up)
GET /ready   — readiness probe (200 only when the champion model is loaded)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health() -> HealthResponse:
    """Returns 200 as long as the API process is running."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
def ready() -> HealthResponse:
    """
    Returns 200 when the champion model is loaded and ready to serve.
    Returns 503 while the model is still loading on startup.
    """
    from aerocast.serving.loader import is_loaded

    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    return HealthResponse(status="ready")
