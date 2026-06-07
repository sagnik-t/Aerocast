"""
Unit tests for Airflow DAG structure.

Uses ``DagBag`` to parse each DAG file without spinning up a real Airflow
instance. Verifies:
  - No import errors
  - All expected DAG IDs present
  - Correct schedules
  - Expected task IDs exist in each DAG
"""

from __future__ import annotations

from pathlib import Path

import pytest

DAG_DIR = str(Path(__file__).parents[2] / "airflow" / "dags")


@pytest.fixture(scope="module")
def dagbag():
    # Import here so Airflow's module registry is only touched once per session
    from airflow.models import DagBag

    return DagBag(dag_folder=DAG_DIR, include_examples=False)


# ── Import health ──────────────────────────────────────────────────────────


class TestDagBagHealth:
    def test_no_import_errors(self, dagbag):
        assert dagbag.import_errors == {}, f"DAG import errors: {dagbag.import_errors}"

    def test_expected_dag_ids_present(self, dagbag):
        expected = {"ingestion", "drift_detection", "retraining"}
        assert expected.issubset(
            dagbag.dags.keys()
        ), f"Missing DAGs: {expected - dagbag.dags.keys()}"


# ── ingestion DAG ──────────────────────────────────────────────────────────


class TestIngestionDag:
    @pytest.fixture(autouse=True)
    def dag(self, dagbag):
        self.dag = dagbag.dags["ingestion"]

    def test_schedule_is_hourly(self):
        assert str(self.dag.schedule_interval) == "@hourly"

    def test_catchup_disabled(self):
        assert self.dag.catchup is False

    def test_expected_task_ids(self):
        expected = {"ingest", "trigger_drift_check"}
        assert expected.issubset(self.dag.task_ids)

    def test_ingest_upstream_of_trigger(self):
        trigger = self.dag.get_task("trigger_drift_check")
        upstream_ids = {t.task_id for t in trigger.upstream_list}
        assert "ingest" in upstream_ids


# ── drift_detection DAG ────────────────────────────────────────────────────


class TestDriftDag:
    @pytest.fixture(autouse=True)
    def dag(self, dagbag):
        self.dag = dagbag.dags["drift_detection"]

    def test_schedule_is_none(self):
        assert self.dag.schedule_interval is None

    def test_catchup_disabled(self):
        assert self.dag.catchup is False

    def test_expected_task_ids(self):
        expected = {
            "check_and_alert",
            "route",
            "trigger_retraining",
            "log_no_drift",
        }
        assert expected.issubset(self.dag.task_ids)

    def test_check_upstream_of_route(self):
        route = self.dag.get_task("route")
        upstream_ids = {t.task_id for t in route.upstream_list}
        assert "check_and_alert" in upstream_ids


# ── retraining DAG ─────────────────────────────────────────────────────────


class TestRetrainingDag:
    @pytest.fixture(autouse=True)
    def dag(self, dagbag):
        self.dag = dagbag.dags["retraining"]

    def test_schedule_is_none(self):
        assert self.dag.schedule_interval is None

    def test_catchup_disabled(self):
        assert self.dag.catchup is False

    def test_expected_task_ids(self):
        expected = {
            "find_latest_features",
            "train_and_promote",
            "redeploy_api",
            "notify_slack",
        }
        assert expected.issubset(self.dag.task_ids)

    def test_find_features_upstream_of_train(self):
        train = self.dag.get_task("train_and_promote")
        upstream_ids = {t.task_id for t in train.upstream_list}
        assert "find_latest_features" in upstream_ids

    def test_train_upstream_of_notify(self):
        notify = self.dag.get_task("notify_slack")
        upstream_ids = {t.task_id for t in notify.upstream_list}
        assert "redeploy_api" in upstream_ids
