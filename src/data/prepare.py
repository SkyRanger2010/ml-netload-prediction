"""Data preparation script for network load forecasting."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.utils.config import apply_overrides, ensure_parent, load_config, to_path
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)
NUMERIC_COLUMNS = ["incoming_gbps", "outgoing_gbps"]


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS:
        if column not in df:
            raise KeyError(f"Column '{column}' is missing in the raw dataset")

    df = df.replace([pd.NA, pd.NaT, float("inf"), float("-inf")], pd.NA)
    df = df.interpolate(method="linear")
    df = df.bfill().ffill()
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].clip(lower=0)
    return df


def prepare_dataset(config: Dict[str, Any]) -> Path:
    """Load raw CSV, clean it and persist the processed dataset."""
    project_root = Path(config["project_root"])
    raw_path = to_path(config["data"]["raw_path"], project_root)
    processed_dir = to_path(config["data"]["processed_dir"], project_root)
    processed_filename = config["data"].get("processed_filename", "clean.csv")
    processed_path = processed_dir / processed_filename

    LOGGER.info("Loading raw data from %s", raw_path)
    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date")
    df = _clean_numeric(df)

    ensure_parent(processed_path)
    df.to_csv(processed_path, index=False)
    LOGGER.info("Saved cleaned dataset to %s", processed_path)
    return processed_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML config",
    )
    args, unknown = parser.parse_known_args()

    config = load_config(args.config)
    overrides = {"cli_args": unknown} if unknown else None
    config = apply_overrides(config, overrides)

    prepare_dataset(config)


if __name__ == "__main__":
    main()
