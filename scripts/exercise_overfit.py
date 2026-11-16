r"""Milestone 4 exercise: see overfitting directly.

Fits two decision trees on the SAME data:
  * shallow, regularised:  max_depth=4, min_samples_leaf=20
  * deep, unconstrained:   max_depth=None, min_samples_leaf=1
and prints TRAIN vs VALIDATION ROC-AUC for each. The gap between train and
validation is the fingerprint of overfitting.

Run (venv):  .\.venv\Scripts\python.exe scripts\exercise_overfit.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retail_repeat import config, evaluate, modeling  # noqa: E402

FEATURES, LABEL = config.FEATURE_COLUMNS, config.LABEL_COLUMN


def load(split):
    df = pd.read_parquet(config.DATA_PROCESSED / f"{split}.parquet")
    return df[FEATURES], df[LABEL].to_numpy()


X_tr, y_tr = load("train")
X_val, y_val = load("val")

trees = {
    "shallow (depth=4, leaf=20)": modeling.make_tree(max_depth=4, min_samples_leaf=20),
    "deep    (depth=None, leaf=1)": modeling.make_tree(max_depth=None, min_samples_leaf=1),
}

print(f"{'tree':<30} {'train AUC':>10} {'val AUC':>9} {'gap':>7}")
print("-" * 60)
for name, est in trees.items():
    est.fit(X_tr, y_tr)
    tr_auc = evaluate.ranking_metrics(y_tr, est.predict_proba(X_tr)[:, 1])["roc_auc"]
    va_auc = evaluate.ranking_metrics(y_val, est.predict_proba(X_val)[:, 1])["roc_auc"]
    depth = est.get_depth()
    print(f"{name:<30} {tr_auc:>10.3f} {va_auc:>9.3f} {tr_auc - va_auc:>7.3f}   (actual depth={depth})")

print("\nBig train-AUC, smaller val-AUC, wide gap = memorising the training set, not learning.")
