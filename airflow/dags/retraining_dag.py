"""
Retraining DAG — triggered by the drift_detection DAG (or manually).

Tasks
-----
find_latest_features
    Globs ``data/processed/features_*.csv`` and returns the path to the
    most recent file.

train_and_promote
    Calls ``run_training(data_path=...)`` which trains MLP + LightGBM,
    logs both runs to W&B, and promotes the lower-RMSE model as champion
    in the W&B registry.

redeploy_api
    Stub — Phase 6 wires in the real serving container restart / Railway
    deploy webhook. Currently emits a log line only.

notify_slack
    Posts a Slack summary with champion name, val RMSE for both models,
    and a link to the W&B run.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task

_DEFAULT_ARGS = {
    "owner": "aerocast",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


@dag(
    dag_id="retraining",
    description="Retrain MLP + LightGBM, promote champion, redeploy API",
    schedule=None,  # triggered by drift_detection DAG
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=_DEFAULT_ARGS,
    tags=["aerocast", "training"],
)
def retraining_dag() -> None:
    @task()
    def find_latest_features() -> str:
        """Return path to the most recent processed features CSV."""
        from pathlib import Path

        from aerocast.config import settings

        processed_dir = Path(settings.data_processed_dir)
        csvs = sorted(processed_dir.glob("features_*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No processed feature files found in {processed_dir}. "
                "Run the ingestion DAG first."
            )
        return str(csvs[-1])

    @task()
    def train_and_promote(features_path: str) -> dict:
        """
        Train MLP + LightGBM, log to W&B, promote champion.
        Returns results dict with champion name and val RMSEs.
        """
        from aerocast.models.train import run_training

        results = run_training(
            data_path=features_path,
            wandb_tags=["phase-4", "auto-retrain"],
        )
        return results

    @task()
    def redeploy_api(results: dict) -> None:
        """
        Restart the serving container with the new champion model.

        Phase 6 implements the real restart / Railway webhook call.
        This stub logs only so the DAG wiring is complete from day one.
        """
        import logging

        logging.getLogger(__name__).info(
            "redeploy_api: stub — champion=%s. "
            "Phase 6 will wire in the serving container restart.",
            results.get("champion"),
        )

    @task()
    def notify_slack(results: dict) -> None:
        """Post a Slack summary of the retraining run."""
        from aerocast.notify import send_slack

        champion = results.get("champion", "unknown")
        mlp_rmse = results.get("mlp_val_rmse", float("nan"))
        lgbm_rmse = results.get("lgbm_val_rmse", float("nan"))

        emoji = ":brain:" if champion == "mlp" else ":deciduous_tree:"
        msg = (
            f"{emoji} *AeroCast retraining complete*\n"
            f"• Champion: *{champion}*\n"
            f"• MLP val RMSE:  `{mlp_rmse:.4f}`\n"
            f"• LGBM val RMSE: `{lgbm_rmse:.4f}`\n"
            f"Serving will reload the new champion on next request."
        )
        send_slack(msg)

    path = find_latest_features()
    results = train_and_promote(path)
    redeploy_api(results) >> notify_slack(results)


retraining_dag()
