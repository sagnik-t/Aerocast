"""Unit tests for OpenAQClient — all HTTP calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from aerocast.data.client import _RAW_COLUMNS, OpenAQClient

# ── helpers ─────────────────────────────────────────────────────────────────


def _make_response(json_body: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    if status_code >= 400:
        http_err = requests.HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_err
    else:
        resp.raise_for_status.return_value = None
    return resp


# v3 sensors response for location 1, parameter pm25
_SENSORS_RESULT = {
    "results": [
        {
            "id": 42,
            "parameter": {
                "id": 2,
                "name": "pm25",
                "displayName": "PM2.5",
                "units": "µg/m³",
            },
        }
    ]
}

# v3 measurements response
_SAMPLE_MEASUREMENTS = {
    "results": [
        {
            "datetime": {"utc": "2024-01-01T00:00:00Z"},
            "value": 12.5,
            "coordinates": {"latitude": 40.71, "longitude": -74.01},
        }
    ]
}

_EMPTY_RESULT = {"results": []}


# ── tests ────────────────────────────────────────────────────────────────────


def test_fetch_measurements_returns_dataframe():
    client = OpenAQClient()
    with patch.object(client.session, "get") as mock_get:
        # 1) sensors for location 1
        # 2) first measurements page (1 result)
        # 3) second measurements page (empty → stop)
        mock_get.side_effect = [
            _make_response(_SENSORS_RESULT),
            _make_response(_SAMPLE_MEASUREMENTS),
            _make_response(_EMPTY_RESULT),
        ]
        df = client.fetch_measurements(location_ids=[1], parameters=["pm25"])

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == _RAW_COLUMNS
    assert df.iloc[0]["location_id"] == 1
    assert df.iloc[0]["parameter"] == "pm25"
    assert df.iloc[0]["value"] == 12.5


def test_fetch_measurements_empty_sensors_returns_empty_df():
    """If a location has no matching sensors, result should be empty."""
    client = OpenAQClient()
    with patch.object(client.session, "get") as mock_get:
        mock_get.return_value = _make_response(_EMPTY_RESULT)
        df = client.fetch_measurements(location_ids=[1], parameters=["pm25"])

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == _RAW_COLUMNS


def test_fetch_measurements_api_key_set_in_header():
    client = OpenAQClient(api_key="test-key-123")
    assert client.session.headers["X-API-Key"] == "test-key-123"


def test_fetch_measurements_pagination():
    """If first measurements page is full, a second page request is made."""
    page1 = {
        "results": [
            {
                "datetime": {"utc": f"2024-01-01T0{i}:00:00Z"},
                "value": float(i),
                "coordinates": None,
            }
            for i in range(3)
        ]
    }
    page2 = {"results": []}

    client = OpenAQClient()
    with patch.object(client.session, "get") as mock_get:
        # sensors → page1 → page2
        mock_get.side_effect = [
            _make_response(_SENSORS_RESULT),
            _make_response(page1),
            _make_response(page2),
        ]
        df = client.fetch_measurements(
            location_ids=[1], parameters=["pm25"], limit_per_page=3
        )

    assert len(df) == 3
    assert mock_get.call_count == 3  # sensors + 2 measurement pages


def test_rate_limit_retries():
    """Client should back off and retry on HTTP 429."""
    rate_limit_resp = _make_response({}, status_code=429)
    success_sensors = _make_response(_SENSORS_RESULT)
    success_meas = _make_response(_SAMPLE_MEASUREMENTS)
    empty = _make_response(_EMPTY_RESULT)

    client = OpenAQClient(max_retries=3)
    with (
        patch.object(client.session, "get") as mock_get,
        patch("time.sleep"),
    ):
        mock_get.side_effect = [
            rate_limit_resp,  # 429 on sensors → retry
            success_sensors,  # sensors ok
            success_meas,  # measurements page 1
            empty,  # measurements page 2 (stop)
        ]
        df = client.fetch_measurements(location_ids=[1], parameters=["pm25"])

    assert not df.empty


def test_no_location_ids_returns_empty():
    client = OpenAQClient()
    df = client.fetch_measurements(location_ids=[], parameters=["pm25"])
    assert df.empty


def test_fetch_locations_uses_iso_param():
    """fetch_locations should pass `iso` not `country` to the v3 API."""
    loc_result = {
        "results": [
            {"id": 1, "name": "Test Station", "country": {"code": "US"}},
        ]
    }
    client = OpenAQClient()
    with patch.object(client.session, "get") as mock_get:
        mock_get.return_value = _make_response(loc_result)
        df = client.fetch_locations(country="US", limit=5)

    assert not df.empty
    # Verify iso= was used in the request params
    called_params = mock_get.call_args[1]["params"]
    assert "iso" in called_params
    assert called_params["iso"] == "US"
    assert "country" not in called_params
