"""
AeroCast smoke test — validates a live deployment.

Usage
-----
    # Against Railway (or any remote):
    BASE_URL=https://your-service.up.railway.app python scripts/smoke_test.py

    # Against the local Docker Compose stack:
    BASE_URL=http://localhost:8000 python scripts/smoke_test.py

Exits 0 on success, 1 on the first failure.
"""

from __future__ import annotations

import os
import sys

import requests

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 15  # seconds

SAMPLE_FEATURES = {
    "pm25": 12.3,
    "pm10": 20.1,
    "o3": 31.5,
    "no2": 14.0,
    "aqi_lag_1h": 48.0,
    "hour": 14,
    "day_of_week": 2,
}


def check(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not ok:
        sys.exit(1)


def test_health() -> None:
    r = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    check(
        "GET /health → 200",
        r.status_code == 200,
        f"status={r.status_code}",
    )
    body = r.json()
    check(
        "GET /health body has status=ok",
        body.get("status") == "ok",
        str(body),
    )


def test_ready() -> None:
    r = requests.get(f"{BASE_URL}/ready", timeout=TIMEOUT)
    check(
        "GET /ready → 200",
        r.status_code == 200,
        f"status={r.status_code}",
    )
    body = r.json()
    check(
        "GET /ready body has status=ready",
        body.get("status") == "ready",
        str(body),
    )


def test_predict() -> None:
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"features": SAMPLE_FEATURES},
        timeout=TIMEOUT,
    )
    check(
        "POST /predict → 200",
        r.status_code == 200,
        f"status={r.status_code} body={r.text[:200]}",
    )
    body = r.json()
    check(
        "POST /predict returns aqi (float)",
        isinstance(body.get("aqi"), (int, float)),
        str(body),
    )
    check(
        "POST /predict returns model_kind (str)",
        isinstance(body.get("model_kind"), str) and body["model_kind"] != "",
        str(body),
    )


def test_metrics() -> None:
    r = requests.get(f"{BASE_URL}/metrics", timeout=TIMEOUT)
    check(
        "GET /metrics → 200",
        r.status_code == 200,
        f"status={r.status_code}",
    )
    check(
        "GET /metrics contains Prometheus counter",
        "aerocast_predictions_total" in r.text,
        "metric name not found in response",
    )


def main() -> None:
    print(f"Smoke-testing {BASE_URL}\n")
    test_health()
    test_ready()
    test_predict()
    test_metrics()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
