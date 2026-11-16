r"""Milestone 3: baselines. The honest floor the ML models must beat.

Run from project root (venv):  .\.venv\Scripts\python.exe scripts\run_baselines.py

Baseline 1 - Dummy: predicts the majority class / constant base rate, ignoring every feature.
Baseline 2 - Recency rule: ranks customers by recency_days only (recent buyer => more likely to return).

We SELECT the recency rule's day-threshold on TRAIN, then report it on VALIDATION.
The TEST split is left untouched until the final-evaluation milestone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_repeat import config, evaluate  # noqa: E402

LABEL = config.LABEL_COLUMN


def load(split: str) -> pd.DataFrame:
    return pd.read_parquet(config.DATA_PROCESSED / f"{split}.parquet")


def main() -> int:
    train, val = load("train"), load("val")
    y_tr, y_val = train[LABEL].to_numpy(), val[LABEL].to_numpy()
    base_rate_tr = y_tr.mean()
    print(f"train positives: {y_tr.mean():.1%}   val positives: {y_val.mean():.1%}\n")

    # ---- Baseline 1: Dummy ------------------------------------------------
    # 1a. majority-class prediction (everyone = 0) -> the 'accuracy trap'
    dummy_pred_val = np.zeros_like(y_val)
    dummy_hard = evaluate.threshold_metrics(y_val, dummy_pred_val)
    # 1b. constant base-rate score -> ranking metrics floor
    dummy_score_val = np.full_like(y_val, fill_value=base_rate_tr, dtype=float)
    dummy_rank = evaluate.ranking_metrics(y_val, dummy_score_val)

    print("== Baseline 1: Dummy (ignores features) ==")
    print(f"  majority-class accuracy (predict all 0): {dummy_hard['accuracy']:.1%}"
          f"   but recall for returners = {dummy_hard['recall']:.0%}  <-- useless")
    print(f"  ROC-AUC = {dummy_rank['roc_auc']:.3f} (=0.5, no ranking skill)"
          f"   AVG-PRECISION = {dummy_rank['avg_precision']:.3f} (= base rate)\n")

    # ---- Baseline 2: Recency rule ----------------------------------------
    # Score: more recent (smaller recency_days) => higher repurchase likelihood => higher score.
    score_tr = -train["recency_days"].to_numpy(dtype=float)
    score_val = -val["recency_days"].to_numpy(dtype=float)
    rank_val = evaluate.ranking_metrics(y_val, score_val)

    # Pick a day-threshold on TRAIN maximising F1, then apply to VALIDATION.
    # Rule: predict repeat if recency_days <= t  <=>  score >= -t.
    best_score_t, best_f1_tr = evaluate.pick_threshold(y_tr, score_tr)
    best_days = int(-best_score_t)
    val_pred = (val["recency_days"].to_numpy() <= best_days).astype(int)
    rule_val = evaluate.threshold_metrics(y_val, val_pred)

    print("== Baseline 2: Recency rule (recency_days only) ==")
    print(f"  ranking on validation:  ROC-AUC = {rank_val['roc_auc']:.3f}"
          f"   AVG-PRECISION = {rank_val['avg_precision']:.3f}")
    print(f"  chosen rule (from train): predict REPEAT if last purchase was <= {best_days} days ago"
          f"   (train F1={best_f1_tr:.3f})")
    print(f"  on validation at that threshold:")
    print(f"    accuracy={rule_val['accuracy']:.1%}  precision={rule_val['precision']:.1%}"
          f"  recall={rule_val['recall']:.1%}  f1={rule_val['f1']:.3f}")
    print(f"    confusion [tn={rule_val['tn']} fp={rule_val['fp']} | fn={rule_val['fn']} tp={rule_val['tp']}]\n")

    print("== Floor to beat (validation) ==")
    print(f"  ROC-AUC  >= {rank_val['roc_auc']:.3f}")
    print(f"  AVG-PREC >= {rank_val['avg_precision']:.3f}")
    print(f"  F1       >= {rule_val['f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
