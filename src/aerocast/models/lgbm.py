"""
LightGBM challenger model — sklearn Pipeline.

Pipeline: SimpleImputer → StandardScaler → LGBMRegressor

Using a Pipeline keeps preprocessing and the model together so the
same object can be serialised (joblib/pickle) and loaded for serving
without separate imputer/scaler state files.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_LGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def build_lgbm_pipeline(**lgbm_kwargs: Any) -> Pipeline:
    """
    Build a sklearn Pipeline: impute → scale → LightGBM.

    Any keyword argument is forwarded to LGBMRegressor, overriding defaults.

    Example::

        pipe = build_lgbm_pipeline(n_estimators=200, learning_rate=0.1)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
    """
    params = {**_LGBM_DEFAULTS, **lgbm_kwargs}
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LGBMRegressor(**params)),
        ]
    )


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def evaluate(pipeline: Pipeline, X, y) -> dict[str, float]:
    """Return val/test metrics dict for a fitted pipeline."""
    preds = pipeline.predict(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    mse = mean_squared_error(y_arr, preds)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_arr - preds)))
    return {"rmse": rmse, "mae": mae, "mse": float(mse)}
