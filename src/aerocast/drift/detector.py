"""
Drift detector — stub for Phase 5.

Phase 5 replaces the body of ``detect_drift`` with an Evidently AI
implementation. The interface (signature + return shape) is frozen here
so the Airflow DAGs can import it now without breaking later.

Return schema
-------------
{
    "drift_detected": bool,
    "drift_score":    float,   # dataset-level p-value or distance metric
    "report_path":   str,      # absolute path to HTML report, or "" if none
    "details":       dict,     # per-column drift flags  {col: bool, ...}
}
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def detect_drift(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    threshold: float = 0.05,
    target_col: str = "aqi",
) -> dict:
    """
    Detect data drift between *current_df* and *reference_df*.

    Args:
        current_df:   Latest processed feature DataFrame.
        reference_df: Baseline / reference DataFrame (from Phase 2 EDA).
        threshold:    Significance level; drift flagged when score < threshold.
        target_col:   Primary column to focus drift analysis on.

    Returns:
        Result dict — see module docstring for schema.

    Note:
        Stub implementation — always returns no drift.
        Phase 5 replaces this with Evidently-based column + target drift.
    """
    logger.warning(
        "detect_drift: stub — always returns no drift. "
        "Phase 5 will implement Evidently AI detection."
    )
    return {
        "drift_detected": False,
        "drift_score": 0.0,
        "report_path": "",
        "details": {},
    }
