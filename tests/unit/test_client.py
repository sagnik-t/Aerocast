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


_SAMPLE_RESULT = {
    "results": [
        {
            "date": {"utc": "2024-01-01T00:00:00Z"},
            "value": 12.5,
            "unit": "µg/m³",
            "coordinates": {"latitude": 40.71, "longitude": -74.01},
        }
    ]
}

_EMPTY_RESULT = {"results": []}


# ── tests ────────────────────────────────────────────────────────────────────


def test_fetch_measurements_returns_dataframe():
    client = OpenAQClient()
    with patch.object(client.session, "get") as mock_get:
        # First call returns one result; second call (next page check) returns empty
        mock_get.side_effect = [
            _make_response(_SAMPLE_RESULT),
            _make_response(_EMPTY_RESULT),
        ]
        df = client.fetch_measurements(location_ids=[1], parameters=["pm25"])

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == _RAW_COLUMNS
    assert df.iloc[0]["location_id"] == 1
    assert df.iloc[0]["parameter"] == "pm25"
    assert df.iloc[0]["value"] == 12.5


def test_fetch_measurements_empty_response_returns_empty_df():
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
    """If first page is full (limit items), a second page request is made."""
    page1 = {
        "results": [
            {
                "date": {"utc": f"2024-01-01T0{i}:00:00Z"},
                "value": float(i),
                "unit": "µg/m³",
                "coordinates": None,
            }
            for i in range(3)
        ]
    }
    page2 = {"results": []}

    client = OpenAQClient()
    with patch.object(client.session, "get") as mock_get:
        mock_get.side_effect = [
            _make_response(page1),
            _make_response(page2),
        ]
        df = client.fetch_measurements(
            location_ids=[1], parameters=["pm25"], limit_per_page=3
        )

    assert len(df) == 3
    assert mock_get.call_count == 2  # two page requests


def test_rate_limit_retries():
    """Client should back off and retry on HTTP 429."""
    rate_limit_resp = _make_response({}, status_code=429)
    success_resp = _make_response(_SAMPLE_RESULT)

    client = OpenAQClient(max_retries=3)
    with (
        patch.object(client.session, "get") as mock_get,
        patch("time.sleep"),
    ):  # don't actually sleep in tests
        mock_get.side_effect = [
            rate_limit_resp,
            success_resp,
            _make_response(_EMPTY_RESULT),
        ]
        df = client.fetch_measurements(location_ids=[1], parameters=["pm25"])

    assert not df.empty


def test_no_location_ids_returns_empty():
    client = OpenAQClient()
    df = client.fetch_measurements(location_ids=[], parameters=["pm25"])
    assert df.empty
