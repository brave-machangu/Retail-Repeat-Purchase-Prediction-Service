"""Inference helpers shared by the API and the tests.

Keeps all model-loading and feature-reuse logic in one place so the FastAPI layer
stays thin. `predict_from_history` reuses the SAME feature function as training
(`features.compute_features_at_cutoff`), which is the structural guarantee that
serving features cannot drift from training features.
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from . import config, data, features

_bundle = None  # process-wide cache


def load_bundle(path=None) -> dict:
    """Load (and cache) the model bundle. Raises FileNotFoundError with a clear hint."""
    global _bundle
    path = path or config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Train it first:  python scripts/train.py"
        )
    _bundle = joblib.load(path)
    return _bundle


def get_bundle() -> dict:
    if _bundle is None:
        load_bundle()
    return _bundle


def set_bundle(bundle: dict) -> None:
    """Inject a bundle directly (used by tests to avoid touching disk)."""
    global _bundle
    _bundle = bundle


def predict_from_features(feature_values: dict) -> dict:
    """Score one precomputed feature vector. Returns probability, decision, and metadata."""
    b = get_bundle()
    X = pd.DataFrame([[feature_values[c] for c in b["features"]]], columns=b["features"])
    prob = float(b["pipeline"].predict_proba(X)[0, 1])
    return {
        "probability": prob,
        "will_repeat": bool(prob >= b["threshold"]),
        "threshold": b["threshold"],
        "model_version": b["model_version"],
    }


def predict_from_history(lines: list[dict], cutoff) -> dict:
    """Score one customer from their raw purchase lines, reusing the training feature code.

    `lines` items need: InvoiceNo, StockCode, InvoiceDate, Quantity, UnitPrice.
    Returns the same shape as predict_from_features, plus the computed `features`.
    Raises ValueError if the customer has no qualifying purchase in the lookback window.
    """
    df = pd.DataFrame(lines)
    required = {"InvoiceNo", "StockCode", "InvoiceDate", "Quantity", "UnitPrice"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"history lines missing fields: {sorted(missing)}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceNo"] = df["InvoiceNo"].astype("string")
    df["StockCode"] = df["StockCode"].astype("string")
    # same qualifying-purchase filter as training (cancellations/qty/price/product)
    inv_special = df["InvoiceNo"].str.match(r"^[A-Za-z]").fillna(False)
    keep = (~inv_special) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0) & data.is_product_code(df["StockCode"])
    df = df[keep].assign(
        CustomerID=0,
        line_total=lambda d: d["Quantity"] * d["UnitPrice"],
    )

    feats = features.compute_features_at_cutoff(df, cutoff)  # <-- identical to training
    if feats.empty:
        raise ValueError("no qualifying purchase in the 90-day lookback window before the cutoff")
    feature_values = {c: float(feats.iloc[0][c]) for c in config.FEATURE_COLUMNS}
    result = predict_from_features(feature_values)
    result["features"] = feature_values
    return result
