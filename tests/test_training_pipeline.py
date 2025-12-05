"""Integration tests for the training and inference pipeline."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from src.models.predict import NetworkLoadForecaster
from src.models.train import run_training
from src.utils.config import apply_overrides, load_config


@pytest.fixture()
def tmp_config(tmp_path: Path) -> tuple[dict, Path]:
    base_config = copy.deepcopy(load_config("configs/base.yaml"))
    repo_root = Path(__file__).resolve().parents[1]
    overrides = {
        "project_root": str(repo_root),
        "data": {
            "processed_dir": str(tmp_path / "processed"),
            "processed_filename": "clean.csv",
        },
        "features": {
            "output_path": str(tmp_path / "features.csv"),
        },
        "artifacts": {
            "model_path": str(tmp_path / "model.joblib"),
            "metadata_path": str(tmp_path / "metadata.json"),
            "metrics_path": str(tmp_path / "metrics.json"),
        },
        "model": {
            "params": {
                "n_estimators": 50,
                "max_depth": 8,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
            }
        },
    }
    config = apply_overrides(base_config, overrides)
    config_path = tmp_path / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return config, config_path


def test_training_produces_artifacts(tmp_config: tuple[dict, Path]) -> None:
    config, config_path = tmp_config
    result = run_training(config)

    for key in ("model_path", "metadata_path", "metrics_path"):
        assert Path(result[key]).exists()

    with Path(result["metrics_path"]).open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    assert "incoming_gbps" in metrics
    assert "outgoing_gbps" in metrics
    assert metrics["aggregate_mae"] >= 0

    forecaster = NetworkLoadForecaster(config_path=str(config_path))
    prediction = forecaster.predict_single("2023-12-01")
    assert set(prediction) == {"incoming_gbps", "outgoing_gbps"}
    assert prediction["incoming_gbps"] >= 0
    assert prediction["outgoing_gbps"] >= 0
