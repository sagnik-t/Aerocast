"""
Preprocessing and feature-engineering for raw OpenAQ measurements.

Pipeline:
    raw DataFrame  →  clean_raw()
                   →  pivot_to_wide()
                   →  compute_aqi_column()
                   →  engineer_features()
                   →  make_dataset()  →  (X, y)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# US EPA AQI breakpoints: (C_low, C_high, I_low, I_high)
# Reference: https://www.airnow.gov/sites/default/files/2020-05/
#            aqi-technical-assistance-document-sept2018.pdf
# ---------------------------------------------------------------------------
_AQI_BREAKPOINTS: dict[str, list[tuple[float, float, int, int]]] = {
    "pm25": [  # µg/m³ (24-hour average; we use instantaneous as proxy)
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ],
    "pm10": [  # µg/m³
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    "o3": [  # ppb (8-hour average)
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300),
    ],
    "no2": [  # ppb
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ],
    "so2": [  # ppb
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 400),
        (805, 1004, 401, 500),
    ],
    "co": [  # ppm
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500),
    ],
}


def _sub_index(
    value: float, breakpoints: list[tuple[float, float, int, int]]
) -> Optional[float]:
    """Piecewise-linear interpolation for a single pollutant sub-index."""
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= value <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (value - c_lo) + i_lo
    return None  # out of range


# ---------------------------------------------------------------------------
# Step 1: clean raw long-form DataFrame
# ---------------------------------------------------------------------------


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise types, drop duplicates, and remove obviously invalid readings.

    Expects columns: location_id, parameter, datetime, value, unit, latitude, longitude
    """
    if df.empty:
        return df

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["location_id"] = df["location_id"].astype(int)

    before = len(df)
    df = df.dropna(subset=["datetime", "value"])
    df = df[df["value"] >= 0]  # sensor errors produce negative readings
    df = df.drop_duplicates(subset=["location_id", "parameter", "datetime"])
    df = df.sort_values(["location_id", "parameter", "datetime"]).reset_index(drop=True)

    logger.info(
        "clean_raw: %d → %d rows (dropped %d)", before, len(df), before - len(df)
    )
    return df


# ---------------------------------------------------------------------------
# Step 2: pivot long → wide and resample to 1-hour grid
# ---------------------------------------------------------------------------


def pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-form measurements to wide format (one column per parameter)
    resampled to a 1-hour UTC grid per location.

    Output columns: datetime, location_id, [pm25, pm10, o3, no2, so2, co] (any subset)
    """
    if df.empty:
        return df

    df = df.copy()
    df = df.set_index("datetime")

    wide = (
        df.groupby(["location_id", "parameter"])["value"]
        .resample("1h")
        .mean()
        .reset_index()
        .pivot_table(
            index=["datetime", "location_id"],
            columns="parameter",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide.columns.name = None
    wide = wide.sort_values(["location_id", "datetime"]).reset_index(drop=True)
    logger.info(
        "pivot_to_wide: %d hourly rows, %d locations",
        len(wide),
        wide["location_id"].nunique(),
    )
    return wide


# ---------------------------------------------------------------------------
# Step 3: compute AQI (max sub-index across available pollutants)
# ---------------------------------------------------------------------------


def compute_aqi_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an `aqi` column = max sub-index across available pollutant columns.
    At least one of the AQI_BREAKPOINTS keys must be present as a column.
    """
    df = df.copy()
    sub_index_cols: list[str] = []

    for param, bps in _AQI_BREAKPOINTS.items():
        if param not in df.columns:
            continue
        col_name = f"_si_{param}"
        df[col_name] = df[param].apply(
            lambda v, b=bps: _sub_index(v, b) if pd.notna(v) else np.nan
        )
        sub_index_cols.append(col_name)

    if not sub_index_cols:
        logger.warning("No AQI pollutant columns found; aqi will be NaN.")
        df["aqi"] = np.nan
    else:
        df["aqi"] = df[sub_index_cols].max(axis=1)

    df = df.drop(columns=sub_index_cols, errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Step 4: feature engineering
# ---------------------------------------------------------------------------


def engineer_features(
    df: pd.DataFrame,
    target_col: str = "aqi",
    lag_hours: tuple[int, ...] = (1, 2, 3, 6, 12, 24),
    rolling_windows: tuple[int, ...] = (6, 12, 24),
) -> pd.DataFrame:
    """
    Add calendar and lagged/rolling features for time-series modelling.

    Assumes the DataFrame is on a regular 1-hour grid per location_id.
    All lag/rolling ops are applied within each location group so no
    cross-location leakage occurs.
    """
    df = df.copy().sort_values(["location_id", "datetime"])

    # Calendar features
    dt = df["datetime"].dt
    df["hour"] = dt.hour
    df["day_of_week"] = dt.dayofweek
    df["month"] = dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)

    # Lag features
    for lag in lag_hours:
        df[f"aqi_lag_{lag}h"] = df.groupby("location_id")[target_col].shift(lag)

    # Rolling mean features (shift 1 first to avoid leakage)
    for window in rolling_windows:
        df[f"aqi_roll_mean_{window}h"] = df.groupby("location_id")[
            target_col
        ].transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 5: build (X, y)
# ---------------------------------------------------------------------------


def make_dataset(
    df: pd.DataFrame,
    horizon: int = 1,
    drop_nulls: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) for supervised learning.

    y = AQI at t + horizon hours
    X = all feature columns (no datetime, no aqi, no target)
    """
    df = df.copy()
    df["target"] = df.groupby("location_id")["aqi"].shift(-horizon)

    if drop_nulls:
        # drop rows where either target or any lag feature is NaN
        df = df.dropna(subset=["target", "aqi_lag_1h"])

    non_feature_cols = {"datetime", "aqi", "target"}
    # Keep location_id as a categorical feature
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    X = df[feature_cols].reset_index(drop=True)
    y = df["target"].reset_index(drop=True).rename("aqi")

    logger.info("make_dataset: X=%s  y=%s  horizon=%dh", X.shape, y.shape, horizon)
    return X, y


# ---------------------------------------------------------------------------
# Convenience: run full preprocessing chain
# ---------------------------------------------------------------------------


def preprocess(
    raw_df: pd.DataFrame, horizon: int = 1
) -> tuple[pd.DataFrame, pd.Series]:
    """
    End-to-end preprocessing:  raw long-form → (X, y)
    """
    df = clean_raw(raw_df)
    df = pivot_to_wide(df)
    df = compute_aqi_column(df)
    df = engineer_features(df)
    return make_dataset(df, horizon=horizon)
