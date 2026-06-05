"""
OpenAQ v2 REST API client.

Fetches hourly air-quality measurements for a list of location IDs and
pollutant parameters, handling pagination and basic retry/back-off.
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
    """Thin wrapper around the OpenAQ v2 REST API."""

    BASE_URL = "https://api.openaq.org/v2"

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
        """
        records: list[dict] = []

        for loc_id in location_ids:
            for param in parameters:
                logger.info("Fetching location=%d parameter=%s", loc_id, param)
                page = 1
                while True:
                    params: dict = {
                        "location_id": loc_id,
                        "parameter": param,
                        "limit": limit_per_page,
                        "page": page,
                    }
                    if date_from:
                        params["date_from"] = date_from.strftime("%Y-%m-%dT%H:%M:%SZ")
                    if date_to:
                        params["date_to"] = date_to.strftime("%Y-%m-%dT%H:%M:%SZ")

                    data = self._get("measurements", params)
                    results = data.get("results", [])

                    if not results:
                        break

                    for r in results:
                        coords = r.get("coordinates") or {}
                        records.append(
                            {
                                "location_id": loc_id,
                                "parameter": param,
                                "datetime": r["date"]["utc"],
                                "value": r["value"],
                                "unit": r.get("unit", ""),
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
        has_geo: bool = True,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Convenience helper to discover location IDs for a country."""
        params: dict = {"country": country, "limit": limit}
        if has_geo:
            params["has_geo"] = "true"
        data = self._get("locations", params)
        return pd.DataFrame(data.get("results", []))
