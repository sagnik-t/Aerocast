"""
Unit tests for aerocast.drift.detector.detect_drift.

Evidently is called with real (small) DataFrames so no mocking is needed.
Two data fixtures cover both the no-drift and drift-detected paths.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N = 150  # enough rows for Evidently's statistical tests


@pytest.fixture()
def reference_df() -> pd.DataFrame:
    """Stable reference data: Normal(50, 5) for all feature columns."""
    return pd.DataFrame(
        {
            "aqi": _RNG.normal(50, 5, _N),
            "pm25": _RNG.normal(12, 2, _N),
            "pm10": _RNG.normal(20, 3, _N),
            "o3": _RNG.normal(30, 4, _N),
        }
    )


@pytest.fixture()
def current_no_drift(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Current data drawn from the same distribution — no drift expected."""
    return pd.DataFrame(
        {
            "aqi": _RNG.normal(50, 5, _N),
            "pm25": _RNG.normal(12, 2, _N),
            "pm10": _RNG.normal(20, 3, _N),
            "o3": _RNG.normal(30, 4, _N),
        }
    )


@pytest.fixture()
def current_with_drift() -> pd.DataFrame:
    """Current data with heavily shifted distributions — drift expected."""
    return pd.DataFrame(
        {
            "aqi": _RNG.normal(150, 5, _N),  # mean shifted 100 units
            "pm25": _RNG.normal(80, 2, _N),  # mean shifted 68 units
            "pm10": _RNG.normal(120, 3, _N),  # mean shifted 100 units
            "o3": _RNG.normal(200, 4, _N),  # mean shifted 170 units
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESULT_KEYS = {"drift_detected", "drift_score", "report_path", "details"}


def _call_detect(current: pd.DataFrame, reference: pd.DataFrame, **kwargs) -> dict:
    """Import and call detect_drift, redirecting reports to a temp dir."""
    import aerocast.config as cfg_module
    from aerocast.drift.detector import detect_drift

    with tempfile.TemporaryDirectory() as tmp:
        original = cfg_module.settings.drift_report_dir
        cfg_module.settings.drift_report_dir = tmp
        try:
            result = detect_drift(current, reference, **kwargs)
        finally:
            cfg_module.settings.drift_report_dir = original
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReturnSchema:
    """detect_drift always returns the documented four-key schema."""

    def test_no_drift_schema(self, current_no_drift, reference_df):
        result = _call_detect(current_no_drift, reference_df)
        assert set(result.keys()) == _RESULT_KEYS

    def test_drift_schema(self, current_with_drift, reference_df):
        result = _call_detect(current_with_drift, reference_df)
        assert set(result.keys()) == _RESULT_KEYS

    def test_drift_score_is_float(self, current_no_drift, reference_df):
        result = _call_detect(current_no_drift, reference_df)
        assert isinstance(result["drift_score"], float)

    def test_details_is_dict(self, current_no_drift, reference_df):
        result = _call_detect(current_no_drift, reference_df)
        assert isinstance(result["details"], dict)

    def test_details_values_are_bool(self, current_with_drift, reference_df):
        result = _call_detect(current_with_drift, reference_df)
        assert all(isinstance(v, bool) for v in result["details"].values())


class TestNoDriftPath:
    """Same-distribution data should not trigger drift."""

    def test_drift_not_detected(self, current_no_drift, reference_df):
        result = _call_detect(current_no_drift, reference_df)
        assert result["drift_detected"] is False

    def test_drift_score_low(self, current_no_drift, reference_df):
        result = _call_detect(current_no_drift, reference_df)
        # With same-distribution data the share of drifted columns should be low
        assert result["drift_score"] < 0.6


class TestDriftDetectedPath:
    """Heavily shifted data must trigger drift."""

    def test_drift_detected(self, current_with_drift, reference_df):
        result = _call_detect(current_with_drift, reference_df)
        assert result["drift_detected"] is True

    def test_drift_score_high(self, current_with_drift, reference_df):
        result = _call_detect(current_with_drift, reference_df)
        assert result["drift_score"] > 0.5

    def test_details_populated(self, current_with_drift, reference_df):
        result = _call_detect(current_with_drift, reference_df)
        assert len(result["details"]) > 0


class TestReportGeneration:
    """An HTML report file must be written on each successful call."""

    def test_report_written(self, current_no_drift, reference_df):
        import aerocast.config as cfg_module
        from aerocast.drift.detector import detect_drift

        with tempfile.TemporaryDirectory() as tmp:
            original = cfg_module.settings.drift_report_dir
            cfg_module.settings.drift_report_dir = tmp
            try:
                result = detect_drift(current_no_drift, reference_df)
            finally:
                cfg_module.settings.drift_report_dir = original

            # Assertions inside the block — tempdir still exists here
            assert result["report_path"] != ""
            assert Path(result["report_path"]).exists()
            assert result["report_path"].endswith(".html")


class TestEdgeCases:
    """detect_drift handles degenerate inputs gracefully."""

    def test_no_shared_columns_returns_no_drift(self):
        """DataFrames with no overlapping columns → safe fallback."""
        from aerocast.drift.detector import detect_drift

        cur = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ref = pd.DataFrame({"y": [4.0, 5.0, 6.0]})
        result = detect_drift(cur, ref)
        assert result["drift_detected"] is False
        assert result["drift_score"] == 0.0
        assert result["report_path"] == ""
        assert result["details"] == {}
