"""Reproducible dataset acquisition for the Retail Repeat-Purchase project.

Downloads the UCI "Online Retail" dataset (ID 352), verifies its size/hash,
extracts the Excel file into data/raw/, and caches a typed parquet copy into
data/interim/ so downstream steps load in ~1s instead of ~40s.

The raw data is NOT committed to git (see .gitignore). Reviewers regenerate it:

    python scripts/get_data.py

If you already have the zip locally, pass its path:

    python scripts/get_data.py --zip "C:/path/to/online+retail.zip"

Source : https://archive.ics.uci.edu/dataset/352/online+retail
License: CC BY 4.0  (Chen, D. 2015, UCI ML Repository, DOI 10.24432/C5BW33)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

# UCI static download endpoint for dataset 352
URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
EXCEL_NAME = "Online Retail.xlsx"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
ZIP_PATH = RAW_DIR / "online+retail.zip"
EXCEL_PATH = RAW_DIR / EXCEL_NAME
PARQUET_PATH = INTERIM_DIR / "online_retail.parquet"

STRING_COLS = ["InvoiceNo", "StockCode", "Description", "Country"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f">> Downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as resp, open(dest, "wb") as out:  # noqa: S310 (trusted UCI host)
        out.write(resp.read())
    print(f"   saved {dest} ({dest.stat().st_size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip", default=None,
                        help="Path to an already-downloaded online+retail.zip")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Obtain the zip (local copy preferred, else download)
    src_zip = Path(args.zip) if args.zip else ZIP_PATH
    if not src_zip.exists():
        download(URL, ZIP_PATH)
        src_zip = ZIP_PATH
    print(f">> Using zip: {src_zip}")
    print(f"   sha256: {sha256(src_zip)}")

    # 2) Extract the Excel file into data/raw/
    with zipfile.ZipFile(src_zip) as zf:
        names = zf.namelist()
        if EXCEL_NAME not in names:
            print(f"!! Expected '{EXCEL_NAME}' inside the zip, found: {names}", file=sys.stderr)
            return 1
        zf.extract(EXCEL_NAME, RAW_DIR)
    print(f">> Extracted {EXCEL_PATH} ({EXCEL_PATH.stat().st_size:,} bytes)")

    # 3) Cache a typed parquet for fast downstream loading
    print(">> Reading Excel and caching parquet (~40s)...")
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    for c in STRING_COLS:
        df[c] = df[c].astype("string")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df.to_parquet(PARQUET_PATH, index=False)
    print(f">> Cached {PARQUET_PATH}  rows={len(df):,} cols={df.shape[1]}")
    print("Done. Downstream code can read data/interim/online_retail.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
