"""Customer-level feature and label construction at a prediction cutoff.

This is the leakage-critical heart of the project and is SHARED by training,
the EDA notebook, and serving. The rules it enforces:

  * Features for cutoff C use ONLY purchases in [C - LOOKBACK_DAYS, C)  (strictly before C).
  * The label for cutoff C uses ONLY purchases in [C, C + HORIZON_DAYS).
  * A customer is eligible at C iff they have >=1 qualifying purchase in the lookback window.
  * CustomerID is an index/identifier, never a feature.
"""
from __future__ import annotations

import pandas as pd

from . import config


def compute_features_at_cutoff(clean_df: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Return one feature row per *eligible* customer, from the lookback window only.

    `clean_df` must be the output of data.clean_transactions (qualifying lines only,
    with a `line_total` column). `cutoff` is a date-like value.
    """
    cutoff = pd.Timestamp(cutoff)
    lb_start = cutoff - pd.Timedelta(days=config.LOOKBACK_DAYS)

    # Only purchases strictly before the cutoff and within the lookback window.
    look = clean_df[(clean_df["InvoiceDate"] >= lb_start) & (clean_df["InvoiceDate"] < cutoff)]
    if look.empty:
        return pd.DataFrame(columns=["CustomerID", *config.FEATURE_COLUMNS]).set_index("CustomerID")

    g = look.groupby("CustomerID")
    feats = pd.DataFrame({
        # days since last qualifying purchase, measured from the cutoff (0..LOOKBACK_DAYS)
        "recency_days": (cutoff - g["InvoiceDate"].max()).dt.days,
        "n_orders": g["InvoiceNo"].nunique(),
        "total_spend": g["line_total"].sum(),
        "n_distinct_products": g["StockCode"].nunique(),
    })
    feats["avg_order_value"] = feats["total_spend"] / feats["n_orders"]
    # column order fixed to config.FEATURE_COLUMNS
    feats = feats[config.FEATURE_COLUMNS]
    return feats


def compute_labels_at_cutoff(clean_df: pd.DataFrame, cutoff) -> pd.Series:
    """Return the set of CustomerIDs who purchase in [C, C + HORIZON_DAYS), as a boolean-ready index."""
    cutoff = pd.Timestamp(cutoff)
    horizon_end = cutoff + pd.Timedelta(days=config.HORIZON_DAYS)
    fut = clean_df[(clean_df["InvoiceDate"] >= cutoff) & (clean_df["InvoiceDate"] < horizon_end)]
    return pd.Index(fut["CustomerID"].unique(), name="CustomerID")


def build_snapshot(clean_df: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Build the labelled customer table for one cutoff.

    Returns a DataFrame indexed by CustomerID with FEATURE_COLUMNS + `label` + `cutoff`.
    Rows are the eligible customers (>=1 purchase in the lookback window).
    """
    cutoff = pd.Timestamp(cutoff)
    feats = compute_features_at_cutoff(clean_df, cutoff)
    future_buyers = compute_labels_at_cutoff(clean_df, cutoff)
    out = feats.copy()
    out[config.LABEL_COLUMN] = out.index.isin(future_buyers).astype("int64")
    out["cutoff"] = cutoff.normalize()
    return out


def features_from_customer_history(history: pd.DataFrame, cutoff) -> dict:
    """Serving helper: compute the feature vector for ONE customer's raw history.

    `history` is that customer's qualifying purchase lines (same schema as clean_df).
    Returns a plain dict of FEATURE_COLUMNS. Used by the API so serving reuses the
    exact training-time feature logic. Returns None if the customer is not eligible.
    """
    feats = compute_features_at_cutoff(history, cutoff)
    if feats.empty:
        return None
    row = feats.iloc[0]
    return {c: float(row[c]) for c in config.FEATURE_COLUMNS}
