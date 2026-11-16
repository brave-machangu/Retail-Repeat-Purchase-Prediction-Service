"""Central configuration: paths, windows, split cutoffs, seeds, cleaning rules.

Everything leakage-sensitive is a named constant here so training, the notebook,
and the serving code all agree on the same definitions.
"""
from __future__ import annotations

import re
from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "Online Retail.xlsx"
DATA_INTERIM = ROOT / "data" / "interim" / "online_retail.parquet"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Reproducibility -----------------------------------------------------
RANDOM_SEED = 42

# --- Final (deployed) model ---------------------------------------------
# Chosen on the validation split in Milestone 4; honest test estimate in Milestone 5.
FINAL_MODEL = {"C": 10.0, "class_weight": None}
MODEL_VERSION = "0.1.0"
MODEL_PATH = ROOT / "models" / "model.joblib"

# --- Prediction-window design (see docs/DATA.md) -------------------------
LOOKBACK_DAYS = 90   # feature window [C - 90d, C)
HORIZON_DAYS = 30    # label window   [C, C + 30d)

# Single chronological snapshot per split. Chosen from the real date range
# (data spans 2010-12-01 .. 2011-12-09); valid cutoffs are 2011-03-01 .. 2011-11-09.
CUTOFFS = {
    "train": "2011-06-01",
    "val": "2011-08-01",
    "test": "2011-10-01",
}

# The last cutoff whose 30-day label window is fully observed in the data.
DATA_END = "2011-12-09"
LAST_VALID_CUTOFF = "2011-11-09"  # DATA_END minus HORIZON_DAYS

# --- Cleaning rules ------------------------------------------------------
# A real product code looks like 5 digits, optionally followed by a letter
# (e.g. 85123A, 71053, 84029G). Everything else (POST, DOT, M, BANK CHARGES,
# AMAZONFEE, CRUK, gift vouchers, discount/sample codes...) is admin/fees, not
# a product sale, and is excluded from the definition of a purchase.
PRODUCT_CODE_RE = re.compile(r"^\d{5}[A-Za-z]?$")

# Feature columns the model will see (CustomerID is deliberately NOT here).
FEATURE_COLUMNS = [
    "recency_days",        # days since last qualifying purchase (0..LOOKBACK_DAYS)
    "n_orders",            # distinct invoices in the lookback window
    "total_spend",         # sum of line totals (Quantity * UnitPrice)
    "avg_order_value",     # total_spend / n_orders
    "n_distinct_products",  # distinct product codes purchased
]
LABEL_COLUMN = "label"
