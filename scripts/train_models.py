r"""Milestone 4: train and compare logistic regression, decision tree, random forest.

Tuning is done on the chronological VALIDATION split (not random CV) to respect time.
Selection metric: validation ROC-AUC. The TEST split is NOT touched here.

Run (venv):  .\.venv\Scripts\python.exe scripts\train_models.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_repeat import config, evaluate, modeling  # noqa: E402

FEATURES = config.FEATURE_COLUMNS
LABEL = config.LABEL_COLUMN


def load(split):
    df = pd.read_parquet(config.DATA_PROCESSED / f"{split}.parquet")
    return df[FEATURES], df[LABEL].to_numpy()


def grid(builder, **param_lists):
    """Yield (params, estimator) for every combination in param_lists."""
    keys = list(param_lists)
    for combo in itertools.product(*param_lists.values()):
        params = dict(zip(keys, combo))
        yield params, builder(**params)


def search(name, candidates, X_tr, y_tr, X_val, y_val):
    """Fit every candidate on train, score on validation, return the best by ROC-AUC."""
    best = None
    for params, est in candidates:
        est.fit(X_tr, y_tr)
        val_score = est.predict_proba(X_val)[:, 1]
        m = evaluate.ranking_metrics(y_val, val_score)
        if best is None or m["roc_auc"] > best["roc_auc"]:
            best = {"name": name, "params": params, "est": est, **m}
    print(f"  {name:<20} best val ROC-AUC={best['roc_auc']:.3f}  AP={best['avg_precision']:.3f}  "
          f"params={best['params']}")
    return best


def main() -> int:
    X_tr, y_tr = load("train")
    X_val, y_val = load("val")
    print(f"train n={len(X_tr):,} ({y_tr.mean():.1%} pos)   val n={len(X_val):,} ({y_val.mean():.1%} pos)\n")

    print("== Tuning on the temporal validation split (selection metric: ROC-AUC) ==")
    results = []
    results.append(search(
        "LogisticRegression",
        grid(modeling.make_logreg, C=[0.01, 0.1, 1.0, 10.0], class_weight=[None, "balanced"]),
        X_tr, y_tr, X_val, y_val))
    results.append(search(
        "DecisionTree",
        grid(modeling.make_tree, max_depth=[2, 3, 4, 5, 8, None],
             min_samples_leaf=[1, 20, 50], class_weight=[None, "balanced"]),
        X_tr, y_tr, X_val, y_val))
    results.append(search(
        "RandomForest",
        grid(modeling.make_forest, n_estimators=[300], max_depth=[4, 6, 8, None],
             min_samples_leaf=[1, 5, 20], class_weight=[None, "balanced"]),
        X_tr, y_tr, X_val, y_val))

    # --- Comparison vs the baseline floor (Milestone 3) ---
    print("\n== Comparison (validation) ==")
    print(f"  {'model':<20} {'ROC-AUC':>8} {'AvgPrec':>8}")
    print(f"  {'Recency baseline':<20} {0.617:>8.3f} {0.405:>8.3f}   <- floor")
    for r in results:
        print(f"  {r['name']:<20} {r['roc_auc']:>8.3f} {r['avg_precision']:>8.3f}")

    # --- Best overall model: pick an operating threshold on validation (max F1) ---
    best = max(results, key=lambda r: r["roc_auc"])
    val_score = best["est"].predict_proba(X_val)[:, 1]
    thr, f1_val = evaluate.pick_threshold(y_val, val_score)
    hard = evaluate.threshold_metrics(y_val, (val_score >= thr).astype(int))
    print(f"\n== Best model: {best['name']} {best['params']} ==")
    print(f"  operating threshold (max-F1 on val) = {thr:.3f}")
    print(f"  val: accuracy={hard['accuracy']:.1%}  precision={hard['precision']:.1%}  "
          f"recall={hard['recall']:.1%}  f1={hard['f1']:.3f}")
    print(f"  confusion [tn={hard['tn']} fp={hard['fp']} | fn={hard['fn']} tp={hard['tp']}]")

    imp = modeling.feature_importance(best["est"], FEATURES)
    print("\n== Feature importance (best model) ==")
    for k, v in sorted(imp.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:<22} {v:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
