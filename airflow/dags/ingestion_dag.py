"""
Ingestion DAG — runs every hour.

Tasks
-----
ingest
    Calls ``aerocast.data.pipeline.run_ingestion()``:
    fetch OpenAQ → validate → preprocess → DVC push.

trigger_drift_check
    Fires the ``drift_detection`` DAG on success so drift is checked
    against every fresh batch of data.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

_DEFAULT_ARGS = {
    "owner": "aerocast",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


@dag(
    dag_id="ingestion",
    description="Hourly OpenAQ ingest → preprocess → validate → DVC push",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=_DEFAULT_ARGS,
    tags=["aerocast", "data"],
)
def ingestion_dag() -> None:
    @task()
    def ingest() -> dict[str, str]:
        """Fetch, preprocess, validate, DVC-push. Returns saved file paths."""
        from aerocast.data.pipeline import run_ingestion

        paths = run_ingestion(dvc_push=True, fail_on_validation_error=True)
        # Serialise Paths to strings for XCom
        return {k: str(v) for k, v in paths.items()}

    trigger_drift = TriggerDagRunOperator(
        task_id="trigger_drift_check",
        trigger_dag_id="drift_detection",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    ingest() >> trigger_drift


ingestion_dag()
