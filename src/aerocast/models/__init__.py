"""
AeroCast model layer.

Primary model:   AQIForecastMLP   (PyTorch Lightning)
Challenger:      build_lgbm_pipeline  (sklearn Pipeline)
Registry:        load_champion, promote_champion
Training:        run_training
"""

from aerocast.models.lgbm import build_lgbm_pipeline, evaluate
from aerocast.models.mlp import AQIForecastMLP, make_dataloaders, make_tensor_dataset
from aerocast.models.registry import load_champion, log_model_artifact, promote_champion
from aerocast.models.train import run_training

__all__ = [
    "AQIForecastMLP",
    "make_tensor_dataset",
    "make_dataloaders",
    "build_lgbm_pipeline",
    "evaluate",
    "log_model_artifact",
    "promote_champion",
    "load_champion",
    "run_training",
]
