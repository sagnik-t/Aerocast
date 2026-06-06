"""
OpenAQ v3 REST API client.

Fetches hourly air-quality measurements for a list of location IDs and
pollutant parameters, handling pagination and basic retry/back-off.

v3 migration notes
------------------
* Base URL is now https://api.openaq.org/v3
* Measurements are per-sensor, not per-location+parameter.
  fetch_measurements() now:
    1. Calls /v3/locations/{id}/sensors to map parameters → sensor IDs.
    2. Pages through /v3/sensors/{sensor_id}/measurements for each sensor.
* fetch_locations() uses `iso` instead of `country`, and `has_geo` is gone
  (all v3 locations carry coordinates).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Standard AQI pollutants tracked by OpenAQ
AQI_PARAMETERS: list[str] = ["pm25", "pm10", "o3", "no2", "so2", "co"]

_RAW_COLUMNS = [
    "location_id",
    "parameter",
    "datetime",
    "value",
    "unit",
    "latitude",
    "longitude",
]


class OpenAQClient:
    """Thin wrapper around the OpenAQ v3 REST API."""

    BASE_URL = "https://api.openaq.org/v3"

    def __init__(
        self,
        api_key: str = "",
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        if api_key:
            self.session.headers["X-API-Key"] = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    # ── private ─────────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict) -> dict:
        """GET with exponential back-off on 429 / transient errors."""
        url = f"{self.BASE_URL}/{endpoint}"
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    wait = 2**attempt
                    logger.warning(
                        "Rate-limited. Waiting %ds (attempt %d)…", wait, attempt + 1
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    raise
                wait = 2**attempt
                logger.warning("Request failed (%s). Retrying in %ds…", exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"All {self.max_retries} attempts failed for {url}")

    def _fetch_sensors(self, location_id: int, parameters: list[str]) -> list[dict]:
        """
        Return sensors for *location_id* whose parameter name is in *parameters*.

        Each element: {"sensor_id": int, "parameter": str, "unit": str}
        """
        data = self._get(f"locations/{location_id}/sensors", {})
        sensors = []
        for s in data.get("results", []):
            param_info = s.get("parameter") or {}
            param_name = (param_info.get("name") or "").lower()
            if param_name in parameters:
                sensors.append(
                    {
                        "sensor_id": s["id"],
                        "parameter": param_name,
                        "unit": param_info.get("units", ""),
                    }
                )
        return sensors

    # ── public ──────────────────────────────────────────────────────────

    def fetch_measurements(
        self,
        location_ids: list[int],
        parameters: list[str] = AQI_PARAMETERS,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit_per_page: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch measurements for all (location, parameter) combos.

        Returns a long-form DataFrame with columns:
            location_id, parameter, datetime, value, unit, latitude, longitude

        v3 note: resolves sensors per location first, then pages through
        /v3/sensors/{sensor_id}/measurements.
        """
        records: list[dict] = []

        for loc_id in location_ids:
            sensors = self._fetch_sensors(loc_id, parameters)
            if not sensors:
                logger.warning(
                    "No matching sensors for location=%d params=%s", loc_id, parameters
                )
                continue

            for sensor in sensors:
                sensor_id = sensor["sensor_id"]
                param = sensor["parameter"]
                unit = sensor["unit"]
                logger.info(
                    "Fetching location=%d sensor=%d parameter=%s",
                    loc_id,
                    sensor_id,
                    param,
                )
                page = 1
                while True:
                    params: dict = {
                        "limit": limit_per_page,
                        "page": page,
                    }
                    if date_from:
                        params["datetime_from"] = date_from.strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        )
                    if date_to:
                        params["datetime_to"] = date_to.strftime("%Y-%m-%dT%H:%M:%SZ")

                    data = self._get(f"sensors/{sensor_id}/measurements", params)
                    results = data.get("results", [])

                    if not results:
                        break

                    for r in results:
                        dt_info = r.get("period", {}).get("datetimeTo") or r.get(
                            "datetime", {}
                        )
                        dt_utc = (
                            dt_info.get("utc") if isinstance(dt_info, dict) else dt_info
                        )
                        coords = r.get("coordinates") or {}
                        records.append(
                            {
                                "location_id": loc_id,
                                "parameter": param,
                                "datetime": dt_utc,
                                "value": r.get("value"),
                                "unit": unit,
                                "latitude": coords.get("latitude"),
                                "longitude": coords.get("longitude"),
                            }
                        )

                    if len(results) < limit_per_page:
                        break  # last page

                    page += 1
                    time.sleep(0.25)  # polite inter-page delay

        if not records:
            logger.warning(
                "No measurements returned for locations=%s params=%s",
                location_ids,
                parameters,
            )
            return pd.DataFrame(columns=_RAW_COLUMNS)

        logger.info("Fetched %d raw measurement records.", len(records))
        return pd.DataFrame(records)

    def fetch_locations(
        self,
        country: str = "US",
        has_geo: bool = True,  # kept for backward compat; ignored in v3
        limit: int = 100,
    ) -> pd.DataFrame:
        """Convenience helper to discover location IDs for a country.

        *country* is the ISO 3166-1 alpha-2 code (e.g. "US").
        *has_geo* is accepted but ignored — all v3 locations have coordinates.
        """
        params: dict = {"iso": country, "limit": limit}
        data = self._get("locations", params)
        return pd.DataFrame(data.get("results", []))
