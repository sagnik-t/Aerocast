"""
Thread-safe champion model loader.

Keeps a single in-process copy of the champion model so FastAPI worker
threads share one loaded model rather than re-downloading on every request.

Usage
-----
    from aerocast.serving.loader import get_model, reload_model

    model, kind = get_model()      # fast — loads once, cached thereafter
    model, kind = reload_model()   # force re-download from W&B registry
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Literal

logger = logging.getLogger(__name__)

_ModelKind = Literal["mlp", "lgbm"]

_lock = threading.Lock()
_model: Any = None
_model_kind: _ModelKind | None = None


def _download() -> tuple[Any, _ModelKind]:
    """Download and deserialise the current W&B champion artifact."""
    from aerocast.config import settings
    from aerocast.models.registry import load_champion

    return load_champion(
        entity=settings.wandb_entity,
        project=settings.wandb_project,
    )


def get_model() -> tuple[Any, _ModelKind]:
    """
    Return (model, model_kind), loading from W&B on the first call.

    Subsequent calls return the cached object; use :func:`reload_model`
    to force a fresh download after a champion promotion.
    """
    global _model, _model_kind

    if _model is None:
        with _lock:
            if _model is None:  # double-checked locking
                logger.info("Loading champion model from W&B registry…")
                _model, _model_kind = _download()
                logger.info("Champion loaded (kind=%s).", _model_kind)

    return _model, _model_kind


def reload_model() -> tuple[Any, _ModelKind]:
    """
    Force a fresh download from W&B and replace the cached model.

    Called by the ``redeploy_api`` Airflow task after champion promotion.
    """
    global _model, _model_kind

    logger.info("Reloading champion model from W&B registry…")
    with _lock:
        _model, _model_kind = _download()
        logger.info("Champion reloaded (kind=%s).", _model_kind)

    return _model, _model_kind


def is_loaded() -> bool:
    """Return True if a model has been loaded into the cache."""
    return _model is not None
