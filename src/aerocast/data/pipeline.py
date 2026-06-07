"""
Data pipeline orchestrator: ingest → preprocess → validate → save → DVC push.

This module is called by:
  - scripts/ingest.py  (manual / ad-hoc runs)
  - airflow/dags/ingestion_dag.py  (scheduled runs)
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from aerocast.config import Settings
from aerocast.config import settings as default_settings
from aerocast.data.client import OpenAQClient
from aerocast.data.preprocess import (
    clean_raw,
    compute_aqi_column,
    engineer_features,
    pivot_to_wide,
)
from aerocast.data.validate import validate_processed, validate_raw

logger = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_ingestion(
    cfg: Settings | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    dvc_push: bool = True,
    fail_on_validation_error: bool = True,
) -> dict[str, Path]:
    """
    Full ingestion pipeline.

    1. Fetch raw measurements from OpenAQ.
    2. Validate raw data with Great Expectations.
    3. Clean + preprocess to wide hourly format + AQI.
    4. Validate processed data.
    5. Persist raw and processed CSV files.
    6. (Optional) DVC add + push.

    Returns paths to the saved files.
    """
    cfg = cfg or default_settings

    if not cfg.location_id_list:
        raise ValueError(
            "No location IDs configured. Set OPENAQ_LOCATION_IDS in your .env file "
            "(comma-separated, e.g. OPENAQ_LOCATION_IDS=1234,5678)."
        )

    # ── 1. Fetch ─────────────────────────────────────────────────────────
    logger.info("Starting ingestion for locations: %s", cfg.location_id_list)
    client = OpenAQClient(
        api_key=cfg.openaq_api_key,
        timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
    )
    raw_df = client.fetch_measurements(
        location_ids=cfg.location_id_list,
        parameters=cfg.openaq_parameters,
        date_from=date_from,
        date_to=date_to,
    )

    if raw_df.empty:
        logger.warning("No data returned from OpenAQ. Aborting pipeline.")
        return {}

    # ── 2. Validate raw ──────────────────────────────────────────────────
    raw_result = validate_raw(raw_df)
    if fail_on_validation_error:
        raw_result.raise_on_failure()

    # ── 3. Preprocess ────────────────────────────────────────────────────
    clean_df = clean_raw(raw_df)
    wide_df = pivot_to_wide(clean_df)
    wide_df = compute_aqi_column(wide_df)
    processed_df = engineer_features(wide_df)

    # ── 4. Validate processed ────────────────────────────────────────────
    processed_result = validate_processed(wide_df)  # validate before feature cols
    if fail_on_validation_error:
        processed_result.raise_on_failure()

    # ── 5. Persist ───────────────────────────────────────────────────────
    ts = _timestamp()
    raw_dir = Path(cfg.data_raw_dir)
    processed_dir = Path(cfg.data_processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"measurements_{ts}.csv"
    processed_path = processed_dir / f"features_{ts}.csv"

    raw_df.to_csv(raw_path, index=False)
    processed_df.to_csv(processed_path, index=False)
    logger.info("Saved raw → %s  processed → %s", raw_path, processed_path)

    # ── 6. DVC ───────────────────────────────────────────────────────────
    if dvc_push:
        _dvc_add_and_push([raw_path, processed_path])

    return {"raw": raw_path, "processed": processed_path}


def _dvc_add_and_push(paths: list[Path]) -> None:
    """Run `dvc add` on each path then `dvc push`."""
    for path in paths:
        _run(["dvc", "add", str(path)])
    _run(["dvc", "push"])
    logger.info("DVC push complete.")


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Command %s failed:\n%s", cmd, result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    logger.debug("$ %s\n%s", " ".join(cmd), result.stdout)
