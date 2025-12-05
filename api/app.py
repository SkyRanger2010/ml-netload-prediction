"""FastAPI inference service for network load forecasting."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import NetworkLoadForecaster
from src.utils.config import load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)
PROJECT_CONFIG = load_config("configs/base.yaml")
app = FastAPI(
    title="Network Load Forecaster",
    version=PROJECT_CONFIG["project"]["version"],
    description="REST API over the trained forecasting model.",
)


class PredictRequest(BaseModel):
    date: datetime = Field(..., description="Timestamp for which to forecast load")


class PredictResponse(BaseModel):
    requested_date: datetime
    incoming_gbps: float
    outgoing_gbps: float
    model_version: str


@app.on_event("startup")
async def startup_event() -> None:
    LOGGER.info("Loading forecasting model on startup")
    app.state.forecaster = NetworkLoadForecaster()


def _get_forecaster() -> NetworkLoadForecaster:
    forecaster = getattr(app.state, "forecaster", None)
    if forecaster is None:
        forecaster = NetworkLoadForecaster()
        app.state.forecaster = forecaster
    return forecaster


@app.get("/health", tags=["system"])
def healthcheck() -> Dict[str, Any]:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(payload: PredictRequest) -> PredictResponse:
    forecaster = _get_forecaster()
    try:
        prediction = forecaster.predict_single(payload.date)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(
        requested_date=payload.date,
        incoming_gbps=prediction.get("incoming_gbps", 0.0),
        outgoing_gbps=prediction.get("outgoing_gbps", 0.0),
        model_version=PROJECT_CONFIG["project"]["version"],
    )
