"""
Drift detection DAG — triggered by the ingestion DAG.

Tasks
-----
check_and_alert
    Loads the latest processed CSV and the reference dataset, runs
    ``detect_drift()``, and fires a Slack alert if drift is detected.
    Returns ``True`` / ``False`` (drift detected).

route
    Branches on the drift flag:
      - drift     → ``trigger_retraining``
      - no drift  → ``log_no_drift``

trigger_retraining
    ``TriggerDagRunOperator`` that fires the ``retraining`` DAG.

log_no_drift
    Logs an INFO message and exits cleanly.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

_DEFAULT_ARGS = {
    "owner": "aerocast",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}


@dag(
    dag_id="drift_detection",
    description="Compare latest data to reference; trigger retraining on drift",
    schedule=None,  # triggered by ingestion_dag
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=_DEFAULT_ARGS,
    tags=["aerocast", "drift"],
)
def drift_dag() -> None:
    @task()
    def check_and_alert() -> bool:
        """
        Load latest processed file + reference, run detect_drift(),
        send Slack alert if drift is detected. Returns drift_detected bool.
        """
        import logging

        import pandas as pd

        from aerocast.config import settings
        from aerocast.drift.detector import detect_drift
        from aerocast.notify import send_slack

        log = logging.getLogger(__name__)

        processed_dir = Path(settings.data_processed_dir)
        reference_path = processed_dir / "reference.csv"

        csvs = sorted(processed_dir.glob("features_*.csv"))
        if not csvs:
            log.warning("No processed data found — skipping drift check.")
            return False

        if not reference_path.exists():
            log.warning("reference.csv not found — skipping drift check.")
            return False

        current_df = pd.read_csv(csvs[-1])
        reference_df = pd.read_csv(reference_path)

        result = detect_drift(current_df, reference_df)
        drift_detected: bool = result["drift_detected"]
        score: float = result.get("drift_score", 0.0)

        if drift_detected:
            send_slack(
                f":rotating_light: *AeroCast* — data drift detected "
                f"(score={score:.4f}). Retraining triggered automatically."
            )
            log.warning("Drift detected (score=%.4f). Retraining queued.", score)
        else:
            log.info("No drift detected (score=%.4f).", score)

        return drift_detected

    @task.branch()
    def route(drift_detected: bool) -> str:
        return "trigger_retraining" if drift_detected else "log_no_drift"

    @task()
    def log_no_drift() -> None:
        import logging

        logging.getLogger(__name__).info("Drift check passed — no retraining needed.")

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="retraining",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    drift_flag = check_and_alert()
    choice = route(drift_flag)
    no_drift_task = log_no_drift()

    choice >> trigger_retraining
    choice >> no_drift_task


drift_dag()
