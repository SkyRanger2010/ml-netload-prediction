"""Train the forecasting model and persist artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.prepare import prepare_dataset
from src.features.build_features import build_feature_table
from src.utils.config import apply_overrides, ensure_parent, load_config, to_path
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _build_model(params: Dict[str, Any]) -> Pipeline:
    estimator = RandomForestRegressor(**params)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _split_dataset(
    dataset: pd.DataFrame, target_columns: list[str], test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_df = dataset.drop(columns=target_columns)
    feature_df = feature_df.drop(columns=["date"], errors="ignore")
    target_df = dataset[target_columns]

    split_idx = max(int(len(dataset) * (1 - test_size)), 1)
    split_idx = min(split_idx, len(dataset) - 1)

    X_train = feature_df.iloc[:split_idx]
    X_test = feature_df.iloc[split_idx:]
    y_train = target_df.iloc[:split_idx]
    y_test = target_df.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def _collect_metrics(
    y_true: pd.DataFrame, y_pred: np.ndarray, target_columns: list[str]
) -> Dict[str, Any]:
    mae_values = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    rmse_values = np.sqrt(
        mean_squared_error(y_true, y_pred, multioutput="raw_values")
    )

    metrics = {}
    for idx, target in enumerate(target_columns):
        metrics[target] = {
            "mae": float(mae_values[idx]),
            "rmse": float(rmse_values[idx]),
        }

    metrics["aggregate_mae"] = float(np.mean(mae_values))
    metrics["aggregate_rmse"] = float(np.mean(rmse_values))
    return metrics


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    project_root = Path(config["project_root"])

    processed_path = prepare_dataset(config)
    dataset, reference_date = build_feature_table(config, processed_path)
    target_columns = config["model"]["target_columns"]
    test_size = float(config["model"].get("test_size", 0.2))

    X_train, X_test, y_train, y_test = _split_dataset(
        dataset, target_columns, test_size
    )

    params = config["model"].get("params", {})
    model = _build_model(params)
    LOGGER.info("Training RandomForest on %s samples", len(X_train))
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = _collect_metrics(y_test, predictions, target_columns)

    model_path = to_path(config["artifacts"]["model_path"], project_root)
    metadata_path = to_path(config["artifacts"]["metadata_path"], project_root)
    metrics_path = to_path(config["artifacts"]["metrics_path"], project_root)

    ensure_parent(model_path)
    joblib.dump(model, model_path)

    metadata = {
        "feature_columns": list(X_train.columns),
        "target_columns": target_columns,
        "reference_date": reference_date.strftime("%Y-%m-%d"),
        "model_class": "RandomForestRegressor",
        "model_params": params,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }

    ensure_parent(metadata_path)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    ensure_parent(metrics_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    LOGGER.info("Artifacts saved: %s", model_path)

    maybe_log_mlflow(config, metrics, params, model_path)

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def maybe_log_mlflow(
    config: Dict[str, Any], metrics: Dict[str, Any], params: Dict[str, Any], model_path: Path
) -> None:
    settings = config.get("mlflow", {})
    if not settings.get("enabled"):
        return

    try:
        import mlflow
    except ImportError:  # pragma: no cover
        LOGGER.warning("MLflow requested but not installed")
        return

    tracking_uri = settings.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = settings.get("experiment_name", "network-load")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="training-run"):
        mlflow.log_params(params)
        flat_metrics = {
            f"{target}_{metric}": value
            for target, target_metrics in metrics.items()
            if isinstance(target_metrics, dict)
            for metric, value in target_metrics.items()
        }
        flat_metrics.update(
            {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (float, int))
            }
        )
        mlflow.log_metrics(flat_metrics)
        mlflow.log_artifact(str(model_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train forecasting model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config)
    run_training(config)


if __name__ == "__main__":
    main()
