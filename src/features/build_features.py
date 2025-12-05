"""Feature generation utilities for the forecasting model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.config import apply_overrides, ensure_parent, load_config, to_path
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _time_features(
    dates: pd.Series, reference_date: pd.Timestamp | None = None
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    dates = pd.to_datetime(dates).dt.tz_localize(None)
    if reference_date is None:
        reference_date = dates.min()

    df = pd.DataFrame(index=dates.index)
    df["day_of_week"] = dates.dt.weekday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = dates.dt.month
    df["quarter"] = dates.dt.quarter
    df["day_of_year"] = dates.dt.dayofyear
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["trend_days"] = (dates - reference_date).dt.days

    # Cyclical encodings
    df["sin_day_of_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_day_of_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    return df, reference_date


def build_feature_table(
    config: Dict[str, Any], processed_path: Path | None = None
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    project_root = Path(config["project_root"])
    processed_path = processed_path or (
        to_path(config["data"]["processed_dir"], project_root)
        / config["data"].get("processed_filename", "clean.csv")
    )

    LOGGER.info("Loading processed data from %s", processed_path)
    df = pd.read_csv(processed_path, parse_dates=["date"])
    features, reference_date = _time_features(df["date"])

    target_columns = config["model"]["target_columns"]
    dataset = pd.concat([features, df[target_columns].reset_index(drop=True)], axis=1)
    dataset.insert(0, "date", df["date"].dt.strftime("%Y-%m-%d"))

    output_path = to_path(config["features"]["output_path"], project_root)
    ensure_parent(output_path)
    dataset.to_csv(output_path, index=False)
    LOGGER.info("Persisted feature table to %s", output_path)
    return dataset, reference_date


def build_features_for_dates(
    dates: pd.Series | list[str], reference_date: pd.Timestamp
) -> pd.DataFrame:
    """Generate the same deterministic features for inference."""
    if isinstance(dates, list):
        dates = pd.Series(dates)
    features, _ = _time_features(dates, reference_date)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build engineered features")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config)
    build_feature_table(config)


if __name__ == "__main__":
    main()
