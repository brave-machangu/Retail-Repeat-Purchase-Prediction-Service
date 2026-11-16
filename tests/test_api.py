"""API + serialization tests. Requires processed splits (python scripts/build_datasets.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from retail_repeat import config, serving, training  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # ensure a real artifact exists, then let the app load it on startup
    if not config.MODEL_PATH.exists():
        training.train_final()
    serving._bundle = None  # force a fresh load from disk
    from api.main import app
    with TestClient(app) as c:
        yield c


VALID = {"recency_days": 12, "n_orders": 3, "total_spend": 742.5,
         "avg_order_value": 247.5, "n_distinct_products": 28}


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"] == config.MODEL_VERSION


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["will_repeat"], bool)
    assert body["model_version"] == config.MODEL_VERSION


def test_predict_rejects_invalid_value(client):
    bad = {**VALID, "n_orders": 0}      # violates ge=1
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in VALID.items() if k != "total_spend"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_from_history(client):
    lines = [
        {"InvoiceNo": "536365", "StockCode": "85123A", "InvoiceDate": "2011-05-10",
         "Quantity": 6, "UnitPrice": 2.55},
        {"InvoiceNo": "536400", "StockCode": "71053", "InvoiceDate": "2011-05-22",
         "Quantity": 3, "UnitPrice": 3.39},
    ]
    r = client.post("/predict/from-history", json={"cutoff": "2011-06-01", "lines": lines})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["features"]["n_orders"] == 2          # two distinct invoices
    assert body["features"]["recency_days"] == 10     # 2011-06-01 minus 2011-05-22


def test_from_history_no_qualifying_purchase_is_422(client):
    lines = [{"InvoiceNo": "C999", "StockCode": "85123A", "InvoiceDate": "2011-05-10",
              "Quantity": 6, "UnitPrice": 2.55}]  # cancellation -> nothing qualifies
    r = client.post("/predict/from-history", json={"cutoff": "2011-06-01", "lines": lines})
    assert r.status_code == 422


def test_bundle_serialization_roundtrip(tmp_path):
    train_df = pd.read_parquet(config.DATA_PROCESSED / "train.parquet")
    val_df = pd.read_parquet(config.DATA_PROCESSED / "val.parquet")
    bundle = training.build_bundle(train_df, val_df)
    path = tmp_path / "model.joblib"
    training.save_bundle(bundle, path)
    serving._bundle = None               # clear cache
    loaded = serving.load_bundle(path)   # real loader sets the global cache
    assert loaded["features"] == config.FEATURE_COLUMNS
    out = serving.predict_from_features(VALID)
    assert 0.0 <= out["probability"] <= 1.0
