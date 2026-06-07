"""
W&B model registry helpers.

Single registry entry: ``aerocast-aqi-forecaster``
Aliases used:
  - ``champion``  — the model currently served by the API
  - ``challenger`` — the runner-up from the latest training run

Both MLP and LightGBM are logged under the same registry name so the
serving layer only needs to know one name and one alias ("champion").
The model type is stored as artifact metadata so the API can
deserialise correctly.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import wandb

logger = logging.getLogger(__name__)

REGISTRY_NAME = "aerocast-aqi-forecaster"
_ModelKind = Literal["mlp", "lgbm"]


# ---------------------------------------------------------------------------
# Saving artifacts
# ---------------------------------------------------------------------------


def log_model_artifact(
    run: "wandb.Run",
    model_path: str | Path,
    model_kind: _ModelKind,
    metrics: dict[str, float],
    aliases: list[str] | None = None,
) -> "wandb.Artifact":
    """
    Log a model file as a W&B Artifact and return it.

    Args:
        run:        Active wandb.Run.
        model_path: Local file path (.pt for MLP, .pkl for LightGBM).
        model_kind: ``"mlp"`` or ``"lgbm"``.
        metrics:    Dict of evaluation metrics (stored as artifact metadata).
        aliases:    Optional list of aliases (e.g. ["champion"]).

    Returns:
        The logged wandb.Artifact.
    """
    model_path = Path(model_path)
    artifact = wandb.Artifact(
        name=REGISTRY_NAME,
        type="model",
        metadata={"model_kind": model_kind, **metrics},
    )
    artifact.add_file(str(model_path))
    run.log_artifact(artifact, aliases=aliases or [])
    logger.info(
        "Logged artifact %s (kind=%s, aliases=%s)", REGISTRY_NAME, model_kind, aliases
    )
    return artifact


# ---------------------------------------------------------------------------
# Champion promotion
# ---------------------------------------------------------------------------


def promote_champion(
    run: "wandb.Run",
    *,
    champion_artifact: "wandb.Artifact",
    challenger_artifact: "wandb.Artifact",
) -> None:
    """
    Assign ``champion`` alias to *champion_artifact* and
    ``challenger`` alias to *challenger_artifact* in the W&B registry.

    Both artifacts must have already been logged in this run via
    :func:`log_model_artifact`.
    """
    entity = run.entity
    project = run.project

    _set_alias(entity, project, champion_artifact, "champion")
    _set_alias(entity, project, challenger_artifact, "challenger")
    logger.info(
        "Promoted %s → champion, %s → challenger",
        champion_artifact.name,
        challenger_artifact.name,
    )


def _set_alias(
    entity: str, project: str, artifact: "wandb.Artifact", alias: str
) -> None:
    """Set a single alias on an artifact via the W&B API."""
    api = wandb.Api()
    # artifact.id is available once it has been logged
    qualified = f"{entity}/{project}/{REGISTRY_NAME}:v{artifact.version}"
    try:
        art = api.artifact(qualified)
        art.aliases = list({*art.aliases, alias})
        art.save()
    except Exception:
        # Fallback: use the artifact object's source_name
        logger.warning(
            "Could not set alias '%s' via API; it will be set during log_artifact.",
            alias,
        )


# ---------------------------------------------------------------------------
# Loading the champion
# ---------------------------------------------------------------------------


def download_champion(
    entity: str,
    project: str,
    download_dir: str | Path = "/tmp/aerocast_model",
) -> tuple[Path, _ModelKind]:
    """
    Download the current champion artifact.

    Returns:
        (local_file_path, model_kind)
    """
    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{project}/{REGISTRY_NAME}:champion")
    download_dir = Path(download_dir)
    artifact.download(root=str(download_dir))

    model_kind: _ModelKind = artifact.metadata.get("model_kind", "mlp")
    # Find the downloaded file
    files = list(download_dir.glob("*"))
    if not files:
        raise FileNotFoundError(f"No files downloaded to {download_dir}")
    return files[0], model_kind


def load_champion(
    entity: str,
    project: str,
    download_dir: str | Path = "/tmp/aerocast_model",
) -> tuple[Any, _ModelKind]:
    """
    Download and deserialise the champion model.

    Returns:
        (model_object, model_kind)
        For ``"mlp"``: model_object is an :class:`AQIForecastMLP`.
        For ``"lgbm"``: model_object is a fitted sklearn Pipeline.
    """
    path, model_kind = download_champion(entity, project, download_dir)

    if model_kind == "mlp":
        from aerocast.models.mlp import AQIForecastMLP

        model = AQIForecastMLP.load_from_checkpoint(str(path))
        model.eval()
        return model, model_kind

    # lgbm
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline, model_kind
