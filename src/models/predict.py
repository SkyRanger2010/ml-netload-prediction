"""Inference helpers for the FastAPI service."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import build_features_for_dates
from src.utils.config import load_config, to_path
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class NetworkLoadForecaster:
    """Wrapper around the trained scikit-learn pipeline."""

    def __init__(self, config_path: str = "configs/base.yaml") -> None:
        self.config = load_config(config_path)
        project_root = Path(self.config["project_root"])
        self.model_path = to_path(self.config["artifacts"]["model_path"], project_root)
        self.metadata_path = to_path(
            self.config["artifacts"]["metadata_path"], project_root
        )
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                "Model artifact is missing. Run src/models/train.py first."
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                "Metadata artifact is missing. Run src/models/train.py first."
            )

        self.model = joblib.load(self.model_path)
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        self.reference_date = pd.to_datetime(metadata["reference_date"])
        self.feature_columns = metadata["feature_columns"]
        self.target_columns = metadata["target_columns"]

    def _prepare_features(self, dates: Iterable[pd.Timestamp | str]) -> pd.DataFrame:
        frame = build_features_for_dates(list(dates), self.reference_date)
        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing features during inference: {missing}")
        frame = frame[self.feature_columns]
        return frame

    def predict_dates(self, dates: Iterable[pd.Timestamp | str]) -> List[dict]:
        features = self._prepare_features(dates)
        predictions = self.model.predict(features)
        output = []
        for idx, pred in enumerate(predictions):
            target_payload = {
                target: float(value)
                for target, value in zip(self.target_columns, np.atleast_1d(pred))
            }
            output.append(target_payload)
        return output

    def predict_single(self, date: pd.Timestamp | str) -> dict:
        return self.predict_dates([date])[0]
