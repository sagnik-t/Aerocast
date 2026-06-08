"""
Tests for Phase 7 monitoring configuration files.

Validates that:
- Prometheus scrape config is well-formed and targets the API
- Grafana datasource config points at the correct Prometheus URL
- Grafana dashboard provider config has the correct path
- Grafana dashboard JSON has the expected panels and UID
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MONITORING = REPO_ROOT / "monitoring"


# ── Prometheus ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def prometheus_cfg() -> dict:
    cfg_path = MONITORING / "prometheus" / "prometheus.yml"
    assert cfg_path.exists(), f"Missing {cfg_path}"
    return yaml.safe_load(cfg_path.read_text())


def test_prometheus_has_global_scrape_interval(prometheus_cfg):
    assert "global" in prometheus_cfg
    assert "scrape_interval" in prometheus_cfg["global"]


def test_prometheus_has_aerocast_scrape_job(prometheus_cfg):
    jobs = {j["job_name"] for j in prometheus_cfg.get("scrape_configs", [])}
    assert "aerocast-api" in jobs, f"Expected job 'aerocast-api', found: {jobs}"


def test_prometheus_scrape_targets_api(prometheus_cfg):
    job = next(
        j for j in prometheus_cfg["scrape_configs"] if j["job_name"] == "aerocast-api"
    )
    targets = job["static_configs"][0]["targets"]
    assert any(
        "8000" in t for t in targets
    ), f"Expected target on port 8000, got: {targets}"


def test_prometheus_metrics_path(prometheus_cfg):
    job = next(
        j for j in prometheus_cfg["scrape_configs"] if j["job_name"] == "aerocast-api"
    )
    assert job.get("metrics_path", "/metrics") == "/metrics"


# ── Grafana datasource ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def grafana_datasource() -> dict:
    ds_path = MONITORING / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
    assert ds_path.exists(), f"Missing {ds_path}"
    return yaml.safe_load(ds_path.read_text())


def test_grafana_datasource_api_version(grafana_datasource):
    assert grafana_datasource.get("apiVersion") == 1


def test_grafana_datasource_is_prometheus(grafana_datasource):
    ds_list = grafana_datasource.get("datasources", [])
    types = {ds["type"] for ds in ds_list}
    assert "prometheus" in types


def test_grafana_datasource_url(grafana_datasource):
    ds_list = grafana_datasource["datasources"]
    prom = next(ds for ds in ds_list if ds["type"] == "prometheus")
    assert (
        "prometheus" in prom["url"]
    ), f"Prometheus URL should contain 'prometheus', got: {prom['url']}"
    assert (
        "9090" in prom["url"]
    ), f"Prometheus URL should reference port 9090, got: {prom['url']}"


def test_grafana_datasource_is_default(grafana_datasource):
    ds_list = grafana_datasource["datasources"]
    prom = next(ds for ds in ds_list if ds["type"] == "prometheus")
    assert prom.get("isDefault") is True


# ── Grafana dashboard provider ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def grafana_dashboard_provider() -> dict:
    provider_path = (
        MONITORING / "grafana" / "provisioning" / "dashboards" / "dashboard.yml"
    )
    assert provider_path.exists(), f"Missing {provider_path}"
    return yaml.safe_load(provider_path.read_text())


def test_grafana_provider_api_version(grafana_dashboard_provider):
    assert grafana_dashboard_provider.get("apiVersion") == 1


def test_grafana_provider_has_file_type(grafana_dashboard_provider):
    providers = grafana_dashboard_provider.get("providers", [])
    types = {p["type"] for p in providers}
    assert "file" in types


def test_grafana_provider_path(grafana_dashboard_provider):
    providers = grafana_dashboard_provider["providers"]
    file_provider = next(p for p in providers if p["type"] == "file")
    path = file_provider["options"]["path"]
    assert (
        "grafana" in path.lower() or "dashboard" in path.lower()
    ), f"Unexpected dashboard path: {path}"


# ── Grafana dashboard JSON ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def grafana_dashboard() -> dict:
    dash_path = MONITORING / "grafana" / "dashboards" / "aerocast.json"
    assert dash_path.exists(), f"Missing {dash_path}"
    return json.loads(dash_path.read_text())


def test_grafana_dashboard_has_uid(grafana_dashboard):
    assert grafana_dashboard.get("uid"), "Dashboard must have a non-empty uid"


def test_grafana_dashboard_has_title(grafana_dashboard):
    assert grafana_dashboard.get("title"), "Dashboard must have a title"


def test_grafana_dashboard_has_panels(grafana_dashboard):
    panels = grafana_dashboard.get("panels", [])
    assert len(panels) >= 4, f"Expected at least 4 panels, got {len(panels)}"


def test_grafana_dashboard_has_prediction_rate_panel(grafana_dashboard):
    titles = {p.get("title", "").lower() for p in grafana_dashboard["panels"]}
    assert any(
        "rate" in t or "prediction" in t for t in titles
    ), f"No prediction-rate panel found. Titles: {titles}"


def test_grafana_dashboard_has_latency_panel(grafana_dashboard):
    titles = {p.get("title", "").lower() for p in grafana_dashboard["panels"]}
    assert any(
        "latency" in t for t in titles
    ), f"No latency panel found. Titles: {titles}"


def test_grafana_dashboard_panels_have_targets(grafana_dashboard):
    for panel in grafana_dashboard["panels"]:
        targets = panel.get("targets", [])
        assert targets, f"Panel '{panel.get('title')}' has no Prometheus targets"


def test_grafana_dashboard_refresh_set(grafana_dashboard):
    assert grafana_dashboard.get("refresh"), "Dashboard should have a refresh interval"
