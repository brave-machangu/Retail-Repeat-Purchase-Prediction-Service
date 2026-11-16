"""Build the labelled train/validation/test customer tables and report on them.

Run from the project root:
    python scripts/build_datasets.py

Outputs (git-ignored): data/processed/{train,val,test}.parquet
Prints: cleaning stats, eligible-customer counts, and class balance per split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# make `src` importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_repeat import config, data, features  # noqa: E402


def main() -> int:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("== LOAD ==")
    raw = data.load_transactions()
    print(f"  raw rows: {len(raw):,}")

    print("\n== CLEAN (qualifying purchases only) ==")
    clean, stats = data.clean_transactions(raw)
    for k, v in stats.as_dict().items():
        if k == "extra":
            continue
        print(f"  {k:<28} {v:>10,}")
    for k, v in stats.extra.items():
        print(f"  {k:<28} {v!s:>10}")
    kept_pct = 100 * stats.rows_out / stats.rows_in
    print(f"  --> kept {kept_pct:.1f}% of rows as qualifying purchase lines")

    print("\n== SNAPSHOTS ==")
    header = f"  {'split':<6} {'cutoff':<12} {'eligible':>9} {'pos':>7} {'neg':>7} {'pos_rate':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    summary = []
    for split, cutoff in config.CUTOFFS.items():
        snap = features.build_snapshot(clean, cutoff)
        n = len(snap)
        pos = int(snap[config.LABEL_COLUMN].sum())
        neg = n - pos
        rate = pos / n if n else float("nan")
        print(f"  {split:<6} {cutoff:<12} {n:>9,} {pos:>7,} {neg:>7,} {rate:>8.1%}")
        out_path = config.DATA_PROCESSED / f"{split}.parquet"
        snap.reset_index().to_parquet(out_path, index=False)
        summary.append((split, cutoff, n, pos, neg, rate))

    print("\n== FEATURE SANITY (train snapshot) ==")
    train = pd.read_parquet(config.DATA_PROCESSED / "train.parquet")
    print(train[config.FEATURE_COLUMNS].describe().round(2).to_string())
    # recency must live in [0, LOOKBACK_DAYS)
    rec = train["recency_days"]
    print(f"\n  recency_days range: [{rec.min()}, {rec.max()}]  "
          f"(must be within [0, {config.LOOKBACK_DAYS}))  -> "
          f"{'OK' if rec.min() >= 0 and rec.max() < config.LOOKBACK_DAYS else 'PROBLEM'}")
    print(f"\n  wrote: {[str(config.DATA_PROCESSED / f'{s}.parquet') for s, *_ in summary]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
