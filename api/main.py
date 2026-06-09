"""
AeroCast FastAPI application.

Endpoints
---------
GET  /health    liveness probe
GET  /ready     readiness probe (model loaded?)
POST /predict   AQI forecast
GET  /metrics   Prometheus metrics (scraped by Prometheus server)

Startup
-------
The champion model is downloaded from W&B on startup and cached
in-process.  Use the WANDB_* env vars to point at the right project.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.routes.health import router as health_router
from api.routes.predict import router as predict_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model once on startup; nothing to clean up."""
    logger.info("AeroCast API starting — loading champion model…")
    try:
        from aerocast.serving.loader import get_model

        _, kind = get_model()
        logger.info("Startup complete — champion kind=%s", kind)
    except Exception as exc:
        # Allow the API to start even if W&B is unreachable;
        # /ready will return 503 until the model is loaded.
        logger.warning("Model load failed on startup: %s", exc)
    yield


app = FastAPI(
    title="AeroCast AQI Forecasting API",
    description=(
        "Production serving layer for the AeroCast ML pipeline. "
        "Returns 1-hour-ahead Air Quality Index forecasts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(health_router)
app.include_router(predict_router)


@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    include_in_schema=False,
    summary="Prometheus metrics scrape endpoint",
)
def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics for scraping."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
