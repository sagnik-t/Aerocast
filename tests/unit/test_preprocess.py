"""Unit tests for preprocessing and feature-engineering functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerocast.data.preprocess import (
    _AQI_BREAKPOINTS,
    _sub_index,
    clean_raw,
    compute_aqi_column,
    engineer_features,
    make_dataset,
    pivot_to_wide,
)

# ── fixtures ─────────────────────────────────────────────────────────────────


def _raw_df() -> pd.DataFrame:
    """Minimal long-form raw DataFrame (2 locations, 2 parameters, 3 hours)."""
    rows = []
    for loc in [1, 2]:
        for hour in range(3):
            for param, value in [("pm25", 10.0 + hour), ("pm10", 30.0 + hour)]:
                rows.append(
                    {
                        "location_id": loc,
                        "parameter": param,
                        "datetime": f"2024-01-01T0{hour}:00:00Z",
                        "value": value,
                        "unit": "µg/m³",
                        "latitude": 40.0,
                        "longitude": -74.0,
                    }
                )
    return pd.DataFrame(rows)


def _wide_df() -> pd.DataFrame:
    """Pre-pivoted wide DataFrame with an aqi column."""
    rng = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "datetime": list(rng) * 2,
            "location_id": [1] * 30 + [2] * 30,
            "pm25": np.random.uniform(5, 80, 60),
            "pm10": np.random.uniform(10, 120, 60),
        }
    )
    return df


# ── _sub_index ────────────────────────────────────────────────────────────────


class TestSubIndex:
    def test_returns_50_at_upper_good_boundary_pm25(self):
        bps = _AQI_BREAKPOINTS["pm25"]
        result = _sub_index(12.0, bps)
        assert result == pytest.approx(50.0)

    def test_returns_0_at_zero(self):
        bps = _AQI_BREAKPOINTS["pm25"]
        result = _sub_index(0.0, bps)
        assert result == pytest.approx(0.0)

    def test_returns_none_out_of_range(self):
        bps = _AQI_BREAKPOINTS["pm25"]
        assert _sub_index(9999, bps) is None

    def test_midpoint_interpolation(self):
        # pm25 [0,12] → [0,50]; midpoint 6 → 25
        bps = _AQI_BREAKPOINTS["pm25"]
        result = _sub_index(6.0, bps)
        assert result == pytest.approx(25.0, rel=1e-3)


# ── clean_raw ─────────────────────────────────────────────────────────────────


class TestCleanRaw:
    def test_removes_negative_values(self):
        df = _raw_df()
        df.loc[0, "value"] = -5.0
        cleaned = clean_raw(df)
        assert (cleaned["value"] >= 0).all()

    def test_removes_duplicate_rows(self):
        df = _raw_df()
        df = pd.concat([df, df.iloc[:1]], ignore_index=True)  # duplicate first row
        cleaned = clean_raw(df)
        assert len(cleaned) == len(_raw_df())

    def test_datetime_becomes_utc(self):
        cleaned = clean_raw(_raw_df())
        assert str(cleaned["datetime"].dt.tz) == "UTC"

    def test_empty_input_returns_empty(self):
        assert clean_raw(pd.DataFrame()).empty


# ── pivot_to_wide ─────────────────────────────────────────────────────────────


class TestPivotToWide:
    def test_output_has_parameter_columns(self):
        df = clean_raw(_raw_df())
        wide = pivot_to_wide(df)
        assert "pm25" in wide.columns
        assert "pm10" in wide.columns

    def test_one_row_per_location_per_hour(self):
        df = clean_raw(_raw_df())
        wide = pivot_to_wide(df)
        assert wide.duplicated(subset=["datetime", "location_id"]).sum() == 0

    def test_correct_number_of_rows(self):
        # 2 locations × 3 hours = 6 rows
        df = clean_raw(_raw_df())
        wide = pivot_to_wide(df)
        assert len(wide) == 6


# ── compute_aqi_column ────────────────────────────────────────────────────────


class TestComputeAqiColumn:
    def test_aqi_column_added(self):
        df = _wide_df()
        result = compute_aqi_column(df)
        assert "aqi" in result.columns

    def test_aqi_in_valid_range(self):
        df = _wide_df()
        result = compute_aqi_column(df)
        valid = result["aqi"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 500).all()

    def test_aqi_nan_when_no_pollutant_columns(self):
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=3, freq="1h"),
                "location_id": [1, 1, 1],
            }
        )
        result = compute_aqi_column(df)
        assert result["aqi"].isna().all()


# ── engineer_features ─────────────────────────────────────────────────────────


class TestEngineerFeatures:
    def test_time_features_present(self):
        df = compute_aqi_column(_wide_df())
        result = engineer_features(df)
        for col in ["hour", "day_of_week", "month", "is_weekend"]:
            assert col in result.columns

    def test_lag_features_present(self):
        df = compute_aqi_column(_wide_df())
        result = engineer_features(df)
        assert "aqi_lag_1h" in result.columns
        assert "aqi_lag_24h" in result.columns

    def test_rolling_features_present(self):
        df = compute_aqi_column(_wide_df())
        result = engineer_features(df)
        assert "aqi_roll_mean_6h" in result.columns
        assert "aqi_roll_mean_24h" in result.columns

    def test_no_cross_location_lag_leakage(self):
        """First row of each location group should have NaN lag_1h."""
        df = compute_aqi_column(_wide_df())
        result = engineer_features(df)
        # Use nth(0) — groupby().first() skips NaN values
        first_rows = (
            result.sort_values(["location_id", "datetime"])
            .groupby("location_id", as_index=False)
            .nth(0)
        )
        assert first_rows["aqi_lag_1h"].isna().all()


# ── make_dataset ──────────────────────────────────────────────────────────────


class TestMakeDataset:
    def _featured_df(self) -> pd.DataFrame:
        df = compute_aqi_column(_wide_df())
        return engineer_features(df)

    def test_returns_x_and_y(self):
        X, y = make_dataset(self._featured_df())
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_shapes_match(self):
        X, y = make_dataset(self._featured_df())
        assert len(X) == len(y)

    def test_no_datetime_in_X(self):
        X, _ = make_dataset(self._featured_df())
        assert "datetime" not in X.columns

    def test_no_aqi_in_X(self):
        X, _ = make_dataset(self._featured_df())
        assert "aqi" not in X.columns

    def test_y_named_aqi(self):
        _, y = make_dataset(self._featured_df())
        assert y.name == "aqi"
