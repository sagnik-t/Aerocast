#!/usr/bin/env python
"""
CLI entry point for a manual ingestion run.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --date-from 2024-01-01 --date-to 2024-01-07
    python scripts/ingest.py --no-dvc-push
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run AeroCast data ingestion pipeline."
    )
    parser.add_argument(
        "--date-from",
        type=_parse_dt,
        default=None,
        help="Start date in ISO format, e.g. 2024-01-01 (default: no lower bound)",
    )
    parser.add_argument(
        "--date-to",
        type=_parse_dt,
        default=None,
        help="End date in ISO format (default: now)",
    )
    parser.add_argument(
        "--no-dvc-push",
        action="store_true",
        help="Skip DVC add + push (useful for local dev)",
    )
    parser.add_argument(
        "--no-fail-on-validation-error",
        action="store_true",
        help="Log validation errors but continue pipeline",
    )
    args = parser.parse_args()

    # Import here so the CLI is fast to parse even if deps aren't installed yet
    from aerocast.data.pipeline import run_ingestion

    paths = run_ingestion(
        date_from=args.date_from,
        date_to=args.date_to,
        dvc_push=not args.no_dvc_push,
        fail_on_validation_error=not args.no_fail_on_validation_error,
    )

    if paths:
        print(f"✓ Raw data      → {paths['raw']}")
        print(f"✓ Processed data → {paths['processed']}")
        return 0
    else:
        print("✗ Ingestion returned no data.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
