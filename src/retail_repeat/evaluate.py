"""Shared evaluation helpers so every model (baselines and ML) is scored identically.

Two families of metric:
  * ranking metrics (roc_auc, average_precision): judge the predicted *scores/probabilities*
    across all thresholds. Threshold-free, good for imbalanced problems.
  * threshold metrics (precision/recall/f1/confusion): judge hard 0/1 predictions AT one chosen
    operating threshold.
Plus `pick_threshold` to choose that operating threshold by maximising F1 on a given split.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def ranking_metrics(y_true, y_score) -> dict:
    """Threshold-free quality of a score/probability. AP baseline == positive rate; AUC baseline == 0.5."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "avg_precision": float(average_precision_score(y_true, y_score)),
    }


def threshold_metrics(y_true, y_pred) -> dict:
    """Quality of hard 0/1 predictions at a fixed threshold."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def pick_threshold(y_true, y_score, grid=None) -> tuple[float, float]:
    """Return (best_threshold, best_f1): the score threshold that maximises F1 on this data.

    A prediction is positive when `y_score >= threshold`. Choose this on TRAIN or VALIDATION,
    never on the test set.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if grid is None:
        grid = np.unique(y_score)
    best_t, best_f1 = grid[0], -1.0
    for t in grid:
        pred = (y_score >= t).astype(int)
        _, _, f, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if f > best_f1:
            best_t, best_f1 = float(t), float(f)
    return best_t, best_f1
