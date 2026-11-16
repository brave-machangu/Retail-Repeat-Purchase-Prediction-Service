r"""Milestone 3 exercise: what if we pick the recency threshold to maximise ACCURACY
instead of F1? Run from project root (venv):

    .\.venv\Scripts\python.exe scripts\exercise_threshold.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retail_repeat import config, evaluate  # noqa: E402

tr = pd.read_parquet(config.DATA_PROCESSED / "train.parquet")
va = pd.read_parquet(config.DATA_PROCESSED / "val.parquet")

# --- pick the threshold that maximises ACCURACY on train ---
best_days, best_acc = None, -1.0
for t in range(1, 90):
    acc = ((tr.recency_days <= t).astype(int) == tr.label).mean()
    if acc > best_acc:
        best_days, best_acc = t, acc

# --- apply that threshold to validation ---
pred = (va.recency_days <= best_days).astype(int)
m = evaluate.threshold_metrics(va.label.to_numpy(), pred.to_numpy())

print(f"accuracy-optimal threshold (from train) = {best_days} days  (train accuracy {best_acc:.1%})")
print(f"on validation: accuracy={m['accuracy']:.1%}  recall={m['recall']:.1%}  precision={m['precision']:.1%}")
print(f"predicted positives on val: {int(pred.sum())} of {len(pred)}")
print("\nCompare to the F1-optimal rule (<= 48 days): recall was 77.5%, precision 37.2%.")
