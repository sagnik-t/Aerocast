#!/usr/bin/env python
"""
End-to-end drift simulation script.

Loads (or generates) a reference dataset, applies a configurable distribution
shift to produce a "current" dataset, then calls ``detect_drift()`` and prints
a summary.  Use this to verify the full Evidently pipeline works before the
Airflow DAG runs it in production.

Usage
-----
    # Synthetic data (no real data required):
    python scripts/simulate_drift.py --synthetic

    # Use real reference.csv from the processed data directory:
    python scripts/simulate_drift.py

    # Control the magnitude of the shift (default 3 sigma):
    python scripts/simulate_drift.py --synthetic --shift 5.0

    # No shift — sanity-check that no-drift path passes:
    python scripts/simulate_drift.py --synthetic --shift 0.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_RNG = np.random.default_rng(0)
_N = 200  # rows for synthetic data


def _make_synthetic_reference() -> pd.DataFrame:
    """Return a clean reference DataFrame with plausible AQI feature ranges."""
    return pd.DataFrame(
        {
            "aqi": _RNG.normal(50, 10, _N),
            "pm25": _RNG.normal(12, 3, _N).clip(0),
            "pm10": _RNG.normal(20, 5, _N).clip(0),
            "o3": _RNG.normal(30, 6, _N).clip(0),
            "no2": _RNG.normal(15, 4, _N).clip(0),
            "so2": _RNG.normal(5, 2, _N).clip(0),
            "co": _RNG.normal(0.5, 0.1, _N).clip(0),
        }
    )


def _apply_shift(df: pd.DataFrame, shift_sigma: float) -> pd.DataFrame:
    """Shift all numeric columns by *shift_sigma* standard deviations."""
    shifted = df.copy()
    for col in shifted.select_dtypes("number").columns:
        sigma = shifted[col].std()
        shifted[col] = shifted[col] + shift_sigma * sigma
    return shifted


def _load_reference() -> pd.DataFrame:
    """Load reference.csv from the configured processed data directory."""
    from aerocast.config import settings

    ref_path = Path(settings.data_processed_dir) / "reference.csv"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"reference.csv not found at {ref_path}. "
            "Run the ingestion pipeline first, or use --synthetic."
        )
    return pd.read_csv(ref_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate drift and run AeroCast drift detector."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of real reference.csv",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=3.0,
        help="Shift in standard deviations (default: 3.0; use 0 for no drift)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Per-column stattest significance level (default: 0.05)",
    )
    args = parser.parse_args()

    # ── Load or generate reference ─────────────────────────────────────
    if args.synthetic:
        logger.info("Generating synthetic reference data (%d rows).", _N)
        reference_df = _make_synthetic_reference()
    else:
        logger.info("Loading reference.csv from processed data directory.")
        try:
            reference_df = _load_reference()
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1

    # ── Build current dataset ──────────────────────────────────────────
    if args.shift == 0.0:
        logger.info("shift=0 — current data drawn from same distribution (no drift).")
        rng = np.random.default_rng(99)
        current_df = reference_df.copy()
        for col in current_df.select_dtypes("number").columns:
            noise = rng.normal(0, current_df[col].std() * 0.05, len(current_df))
            current_df[col] = current_df[col] + noise
    else:
        logger.info("Applying %.1f-sigma shift to all numeric columns.", args.shift)
        current_df = _apply_shift(reference_df, args.shift)

    logger.info(
        "Reference shape: %s  |  Current shape: %s",
        reference_df.shape,
        current_df.shape,
    )

    # ── Run drift detection ────────────────────────────────────────────
    from aerocast.drift.detector import detect_drift

    result = detect_drift(
        current_df=current_df,
        reference_df=reference_df,
        threshold=args.threshold,
    )

    # ── Print summary ──────────────────────────────────────────────────
    print()
    print("=" * 56)
    print("  AeroCast Drift Simulation — Results")
    print("=" * 56)
    print(f"  Drift detected : {result['drift_detected']}")
    print(f"  Drift score    : {result['drift_score']:.4f}  (share of drifted cols)")
    print(f"  Report         : {result['report_path'] or '(none saved)'}")
    print()
    if result["details"]:
        print("  Per-column breakdown:")
        for col, drifted in sorted(result["details"].items()):
            flag = "DRIFT" if drifted else "  ok "
            print(f"    [{flag}]  {col}")
    print("=" * 56)
    print()

    return 0 if not result["drift_detected"] else 2  # 2 = drift found (not an error)


if __name__ == "__main__":
    sys.exit(main())
