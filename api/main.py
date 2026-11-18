"""FastAPI service: 12-month rainfall forecasts with 80% intervals."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).resolve().parents[1] / "models" / "forecaster.joblib"))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        state.update(joblib.load(MODEL_PATH))
    yield
    state.clear()


app = FastAPI(title="Rainfall forecast API", version="1.0.0", lifespan=lifespan,
              description="Changi (Singapore) monthly rainfall: total and highest single-day rainfall, up to 12 months ahead.")

Target = Literal["total", "max_daily"]


class ForecastPoint(BaseModel):
    month: str
    forecast_mm: float
    lower_80_mm: float
    upper_80_mm: float


class ForecastResponse(BaseModel):
    target: Target
    unit: str = "mm"
    data_through: str
    points: list[ForecastPoint]


def _require_model() -> dict:
    if "forecasters" not in state:
        raise HTTPException(503, "model not loaded; run `python scripts/train.py` first")
    return state


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if "forecasters" in state else "model_not_loaded"}


@app.get("/model-info")
def model_info() -> dict:
    return _require_model()["info"]


@app.get("/forecast", response_model=ForecastResponse)
def forecast(target: Target = "total", horizon: int = Query(12, ge=1, le=12)) -> ForecastResponse:
    s = _require_model()
    fc = s["forecasters"][target].predict(horizon)
    points = [ForecastPoint(month=m.strftime("%Y-%m"), forecast_mm=round(float(r.forecast), 1),
                            lower_80_mm=round(float(r.lower_80), 1), upper_80_mm=round(float(r.upper_80), 1))
              for m, r in fc.iterrows()]
    return ForecastResponse(target=target, data_through=s["info"]["data_through"], points=points)
