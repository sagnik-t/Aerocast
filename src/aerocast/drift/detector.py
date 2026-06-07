"""
Drift detector — Evidently AI implementation.

Uses ``DataDriftPreset`` to run per-column statistical tests and produce a
dataset-level drift verdict.  An HTML report is written to the configured
``drift_report_dir`` on every call so results are auditable over time.

Return schema
-------------
{
    "drift_detected": bool,
    "drift_score":    float,   # share of columns that drifted (0.0–1.0)
    "report_path":   str,      # absolute path to HTML report, or "" on error
    "details":       dict,     # per-column drift flags  {col: bool, ...}
}
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from aerocast.config import settings

logger = logging.getLogger(__name__)


def _save_report(report: Report, report_dir: str) -> str:
    """Persist *report* as HTML and return the absolute path."""
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"drift_{timestamp}.html"
    report.save_html(str(path))
    return str(path.resolve())


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
        threshold:    Per-column stattest significance level; a column is
                      flagged as drifted when its test score falls below
                      this value.
        target_col:   Primary column to focus drift analysis on (informational
                      only — all shared numeric columns are tested).

    Returns:
        Result dict — see module docstring for schema.
    """
    # ── Guard: need overlapping numeric columns ────────────────────────
    shared_cols = list(
        set(current_df.select_dtypes("number").columns)
        & set(reference_df.select_dtypes("number").columns)
    )
    if not shared_cols:
        logger.warning(
            "detect_drift: no shared numeric columns — skipping drift check."
        )
        return {
            "drift_detected": False,
            "drift_score": 0.0,
            "report_path": "",
            "details": {},
        }

    cur = current_df[shared_cols].copy()
    ref = reference_df[shared_cols].copy()

    # ── Run Evidently report ───────────────────────────────────────────
    report = Report(
        metrics=[
            DataDriftPreset(
                stattest_threshold=threshold,
                drift_share=settings.drift_dataset_threshold,
            )
        ]
    )
    report.run(reference_data=ref, current_data=cur)

    result_dict = report.as_dict()

    # metrics[0] → DatasetDriftMetric
    dataset_result = result_dict["metrics"][0]["result"]
    drift_detected: bool = bool(dataset_result.get("dataset_drift", False))
    drift_score: float = float(dataset_result.get("share_of_drifted_columns", 0.0))

    # metrics[1] → DataDriftTable (per-column)
    column_results = result_dict["metrics"][1]["result"].get("drift_by_columns", {})
    details: dict[str, bool] = {
        col: bool(info.get("drift_detected", False))
        for col, info in column_results.items()
    }

    # ── Persist HTML report ────────────────────────────────────────────
    report_path = ""
    try:
        report_path = _save_report(report, settings.drift_report_dir)
        logger.info("Drift report saved → %s", report_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save drift report: %s", exc)

    logger.info(
        "Drift check — drift_detected=%s  score=%.3f  drifted=%d/%d cols",
        drift_detected,
        drift_score,
        sum(details.values()),
        len(details),
    )

    return {
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "report_path": report_path,
        "details": details,
    }
