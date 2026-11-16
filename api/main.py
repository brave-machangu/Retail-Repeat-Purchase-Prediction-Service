"""FastAPI prediction service for the Retail Repeat-Purchase model.

Endpoints:
  GET  /health            -> liveness + model version
  POST /predict           -> score PRECOMPUTED customer features
  POST /predict/from-history -> score a customer's RAW purchase lines (reuses training feature code)

Run (venv):  uvicorn api.main:app --reload --port 8000
Then open http://127.0.0.1:8000/docs for interactive docs.
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# make `src` importable whether launched from repo root or elsewhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retail_repeat import config, serving  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("retail_repeat.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        b = serving.load_bundle()
        logger.info("model loaded: version=%s trained_at=%s", b["model_version"], b.get("trained_at"))
    except FileNotFoundError as e:
        # Don't crash the server; /health reports degraded and /predict returns 503.
        logger.warning("model not loaded: %s", e)
    yield


app = FastAPI(
    title="Retail Repeat-Purchase Prediction Service",
    version=config.MODEL_VERSION,
    description="Predicts whether an existing customer will purchase again within 30 days.",
    lifespan=lifespan,
)


# ---- Schemas (also serve as input validation + API docs) -------------------
class CustomerFeatures(BaseModel):
    """Precomputed 90-day-lookback features. Must be produced with the project's
    feature code (retail_repeat.features.compute_features_at_cutoff) to stay consistent."""
    recency_days: float = Field(..., ge=0, le=config.LOOKBACK_DAYS, examples=[12])
    n_orders: int = Field(..., ge=1, examples=[3])
    total_spend: float = Field(..., gt=0, examples=[742.50])
    avg_order_value: float = Field(..., gt=0, examples=[247.50])
    n_distinct_products: int = Field(..., ge=1, examples=[28])


class PurchaseLine(BaseModel):
    InvoiceNo: str
    StockCode: str
    InvoiceDate: str  # ISO date/datetime
    Quantity: int
    UnitPrice: float


class HistoryRequest(BaseModel):
    cutoff: str = Field(..., description="Prediction date (ISO). Features use the 90 days before it.",
                        examples=["2011-06-01"])
    lines: list[PurchaseLine] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    probability: float
    will_repeat: bool
    threshold: float
    model_version: str


class HistoryResponse(PredictionResponse):
    features: dict


# ---- Lifecycle -------------------------------------------------------------
def _require_model():
    try:
        return serving.get_bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---- Endpoints -------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    try:
        b = serving.get_bundle()
        return {"status": "ok", "model_loaded": True, "model_version": b["model_version"]}
    except FileNotFoundError:
        return {"status": "degraded", "model_loaded": False, "model_version": None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> dict:
    _require_model()
    t0 = time.perf_counter()
    result = serving.predict_from_features(features.model_dump())
    logger.info("predict prob=%.3f repeat=%s %.1fms",  # no customer identifiers logged
                result["probability"], result["will_repeat"], 1000 * (time.perf_counter() - t0))
    return result


@app.post("/predict/from-history", response_model=HistoryResponse)
def predict_from_history(req: HistoryRequest) -> dict:
    _require_model()
    try:
        result = serving.predict_from_history([ln.model_dump() for ln in req.lines], req.cutoff)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    logger.info("predict/from-history prob=%.3f repeat=%s", result["probability"], result["will_repeat"])
    return result
