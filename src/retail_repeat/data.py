"""Load and clean the Online Retail transactions.

`clean_transactions` implements the cleaning contract from docs/DATA.md and
returns ONLY qualifying purchase lines, with a `line_total` column added.
It also returns a stats dict so callers can report exactly what was removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config


@dataclass
class CleaningStats:
    rows_in: int = 0
    dropped_missing_customer: int = 0
    dropped_duplicates: int = 0
    dropped_cancellations: int = 0
    dropped_nonpositive_qty: int = 0
    dropped_nonpositive_price: int = 0
    dropped_non_product: int = 0
    rows_out: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__


def load_transactions(path=None) -> pd.DataFrame:
    """Load the cached parquet (fast) or the raw Excel if the parquet is absent."""
    parquet = config.DATA_INTERIM if path is None else path
    if config.DATA_INTERIM.exists():
        df = pd.read_parquet(config.DATA_INTERIM)
    else:  # fall back to the raw Excel
        df = pd.read_excel(config.DATA_RAW, engine="openpyxl")
        for c in ["InvoiceNo", "StockCode", "Description", "Country"]:
            df[c] = df[c].astype("string")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def is_product_code(stockcode: pd.Series) -> pd.Series:
    """True where the StockCode looks like a real product (not postage/fees/vouchers)."""
    return stockcode.astype("string").str.match(config.PRODUCT_CODE_RE).fillna(False)


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """Apply the cleaning contract. Returns (qualifying_purchase_lines, stats).

    Steps (order matters only for the per-step counts, not the final result):
      1. drop rows with missing CustomerID
      2. drop exact duplicate rows
      3. exclude cancellation invoices (InvoiceNo starts with 'C') and 'A' adjustments
      4. require Quantity > 0
      5. require UnitPrice > 0
      6. keep only real product codes
    """
    stats = CleaningStats(rows_in=len(df))
    d = df.copy()

    # 1) missing CustomerID -> cannot attribute to a customer
    mask = d["CustomerID"].notna()
    stats.dropped_missing_customer = int((~mask).sum())
    d = d[mask]

    # 2) exact duplicate rows
    before = len(d)
    d = d.drop_duplicates()
    stats.dropped_duplicates = before - len(d)

    # 3) cancellations / adjustments (non-numeric InvoiceNo prefixes)
    inv = d["InvoiceNo"].astype("string")
    is_special = inv.str.match(r"^[A-Za-z]").fillna(False)  # 'C...' and 'A...'
    stats.dropped_cancellations = int(is_special.sum())
    d = d[~is_special]

    # 4) non-positive quantity (returns, zero lines)
    before = len(d)
    d = d[d["Quantity"] > 0]
    stats.dropped_nonpositive_qty = before - len(d)

    # 5) non-positive price (free gifts, adjustments, bad data)
    before = len(d)
    d = d[d["UnitPrice"] > 0]
    stats.dropped_nonpositive_price = before - len(d)

    # 6) real products only
    before = len(d)
    d = d[is_product_code(d["StockCode"])]
    stats.dropped_non_product = before - len(d)

    # derived column + tidy types
    d = d.assign(
        line_total=(d["Quantity"] * d["UnitPrice"]).astype(float),
        CustomerID=d["CustomerID"].astype("int64"),
    )
    stats.rows_out = len(d)
    stats.extra = {
        "distinct_customers": int(d["CustomerID"].nunique()),
        "distinct_orders": int(d["InvoiceNo"].nunique()),
        "date_min": str(d["InvoiceDate"].min()),
        "date_max": str(d["InvoiceDate"].max()),
    }
    return d, stats
