from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model import FEATURE_COLUMNS, clean_data, load_model
from .schemas import PredictionResponse, ReturnRequest
from .ui_config import (
    CATEGORICAL_OPTIONS,
    FORM_DEFAULTS,
    MODEL_LABEL,
    MODEL_METRICS,
    MODEL_NAME,
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/return_model.joblib"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}. Run `python -m returns_app.train` first.")
    app.state.model = load_model(MODEL_PATH)
    yield


STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Returns Prediction API", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "model": "loaded" if hasattr(request.app.state, "model") else "missing"}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    return {
        "model": MODEL_NAME,
        "model_label": MODEL_LABEL,
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "metrics": MODEL_METRICS,
        "categorical_options": CATEGORICAL_OPTIONS,
        "defaults": FORM_DEFAULTS,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ReturnRequest, request: Request) -> PredictionResponse:
    try:
        row = pd.DataFrame([payload.model_dump()])
        row = clean_data(row)
        model = request.app.state.model
        probability = float(model.predict_proba(row)[0, 1])
        return PredictionResponse(returned=probability >= 0.5, probability=probability)
    except Exception as error:
        raise HTTPException(status_code=500, detail="Prediction failed") from error
