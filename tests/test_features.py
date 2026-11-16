"""Correctness + leakage tests for the feature/label builder.

These use a tiny hand-computed synthetic dataset so every expected value is
verifiable by eye. Run: pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_repeat import config, features  # noqa: E402

CUTOFF = pd.Timestamp("2011-06-01")
# Lookback window is [2011-03-03, 2011-06-01); horizon is [2011-06-01, 2011-07-01).


def _row(cust, date, inv, code, qty, price):
    return {
        "InvoiceNo": inv, "StockCode": code, "CustomerID": cust,
        "InvoiceDate": pd.Timestamp(date), "Quantity": qty, "UnitPrice": price,
        "line_total": qty * price,
    }


@pytest.fixture()
def clean_df():
    rows = [
        # Customer 1: two orders in lookback (one with 2 products), plus a purchase AFTER cutoff
        _row(1, "2011-05-02", "100", "10001", 2, 5.0),   # order 100, product 10001
        _row(1, "2011-05-02", "100", "10002", 1, 20.0),  # same order, product 10002
        _row(1, "2011-05-22", "101", "10001", 1, 5.0),   # order 101, repeat product
        _row(1, "2011-06-10", "102", "10001", 1, 5.0),   # HORIZON purchase -> label 1, must NOT affect features
        # Customer 2: one order early in lookback, nothing after -> label 0
        _row(2, "2011-03-10", "200", "10005", 3, 2.0),
        # Customer 3: only a purchase BEFORE the lookback window -> NOT eligible, excluded
        _row(3, "2011-01-15", "300", "10009", 1, 9.0),
    ]
    return pd.DataFrame(rows)


def test_eligibility_excludes_customers_with_no_lookback_purchase(clean_df):
    snap = features.build_snapshot(clean_df, CUTOFF)
    assert set(snap.index) == {1, 2}        # customer 3 has no purchase in the lookback window
    assert 3 not in snap.index


def test_feature_values_customer1(clean_df):
    snap = features.build_snapshot(clean_df, CUTOFF)
    c1 = snap.loc[1]
    assert c1["n_orders"] == 2                      # invoices 100 and 101
    assert c1["n_distinct_products"] == 2           # 10001 and 10002
    assert c1["total_spend"] == pytest.approx(35.0)  # 10 + 20 + 5
    assert c1["avg_order_value"] == pytest.approx(17.5)  # 35 / 2
    # recency measured from the cutoff, last lookback purchase 2011-05-22 -> 10 days
    assert c1["recency_days"] == 10


def test_label_uses_only_the_horizon_window(clean_df):
    snap = features.build_snapshot(clean_df, CUTOFF)
    assert snap.loc[1, "label"] == 1   # bought again on 2011-06-10 (inside horizon)
    assert snap.loc[2, "label"] == 0   # never bought after the cutoff


def test_no_leakage_future_purchase_does_not_change_features(clean_df):
    """Removing customer 1's post-cutoff purchase must leave the FEATURES identical
    (only the label may change). This is the core leakage guarantee."""
    with_future = features.build_snapshot(clean_df, CUTOFF)
    without_future = features.build_snapshot(
        clean_df[clean_df["InvoiceDate"] < CUTOFF], CUTOFF
    )
    for col in config.FEATURE_COLUMNS:
        assert with_future.loc[1, col] == without_future.loc[1, col]


def test_recency_is_within_the_lookback_window(clean_df):
    snap = features.build_snapshot(clean_df, CUTOFF)
    assert (snap["recency_days"] >= 0).all()
    assert (snap["recency_days"] < config.LOOKBACK_DAYS).all()
