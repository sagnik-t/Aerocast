"""
Central configuration via pydantic-settings.
All values are read from environment variables (or .env file).
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenAQ ──────────────────────────────────────────────────────────
    openaq_api_key: str = Field(default="", alias="OPENAQ_API_KEY")
    openaq_base_url: str = "https://api.openaq.org/v2"

    # ── Data ingestion ───────────────────────────────────────────────────
    # Comma-separated list of OpenAQ location IDs to ingest
    openaq_location_ids: str = Field(default="", alias="OPENAQ_LOCATION_IDS")
    # Parameters to fetch (all standard AQI pollutants)
    openaq_parameters: list[str] = ["pm25", "pm10", "o3", "no2", "so2", "co"]
    request_timeout: int = 30
    max_retries: int = 3
    # Forecast horizon (hours ahead to predict)
    forecast_horizon: int = 1

    # ── Weights & Biases ─────────────────────────────────────────────────
    wandb_api_key: str = Field(default="", alias="WANDB_API_KEY")
    wandb_project: str = Field(default="aerocast", alias="WANDB_PROJECT")
    wandb_entity: str = Field(default="", alias="WANDB_ENTITY")

    # ── DagsHub / DVC ────────────────────────────────────────────────────
    dagshub_user_token: str = Field(default="", alias="DAGSHUB_USER_TOKEN")
    dagshub_repo_owner: str = Field(default="", alias="DAGSHUB_REPO_OWNER")
    dagshub_repo_name: str = Field(default="aerocast", alias="DAGSHUB_REPO_NAME")

    # ── FastAPI serving ───────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    model_alias: str = Field(default="champion", alias="MODEL_ALIAS")

    # ── Slack ─────────────────────────────────────────────────────────────
    slack_webhook_url: str = Field(default="", alias="SLACK_WEBHOOK_URL")

    # ── Helpers ───────────────────────────────────────────────────────────
    @property
    def location_id_list(self) -> list[int]:
        """Parse OPENAQ_LOCATION_IDS env var into a list of ints."""
        if not self.openaq_location_ids:
            return []
        return [
            int(x.strip()) for x in self.openaq_location_ids.split(",") if x.strip()
        ]


settings = Settings()
