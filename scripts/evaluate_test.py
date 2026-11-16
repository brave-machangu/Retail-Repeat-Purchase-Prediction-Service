r"""Milestone 5: honest final evaluation on the UNTOUCHED test split.

Pipeline:
  1. Fit the chosen model (LogisticRegression, C=10) on TRAIN only.
  2. Choose the operating threshold on VALIDATION (max F1).
  3. Evaluate ONCE on TEST at that frozen threshold -> honest generalization estimate.
  4. Probability calibration (Brier score + reliability table).
  5. Error analysis (feature profiles of TP/FP/FN/TN + representative mistakes).

Writes reports/test_metrics.json and reports/figures/calibration_logreg.png.
Run (venv):  python scripts\evaluate_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retail_repeat import config, evaluate, modeling  # noqa: E402

FEATURES, LABEL = config.FEATURE_COLUMNS, config.LABEL_COLUMN
CHOSEN = dict(C=10.0, class_weight=None)   # selected on validation in Milestone 4


def load(split):
    df = pd.read_parquet(config.DATA_PROCESSED / f"{split}.parquet")
    return df, df[FEATURES], df[LABEL].to_numpy()


def main() -> int:
    tr_df, X_tr, y_tr = load("train")
    va_df, X_va, y_va = load("val")
    te_df, X_te, y_te = load("test")

    # 1) fit on train only
    model = modeling.make_logreg(**CHOSEN)
    model.fit(X_tr, y_tr)

    # 2) threshold chosen on validation (max F1)
    va_score = model.predict_proba(X_va)[:, 1]
    thr, va_f1 = evaluate.pick_threshold(y_va, va_score)

    # scores
    va_rank = evaluate.ranking_metrics(y_va, va_score)
    te_score = model.predict_proba(X_te)[:, 1]
    te_rank = evaluate.ranking_metrics(y_te, te_score)
    te_hard = evaluate.threshold_metrics(y_te, (te_score >= thr).astype(int))
    va_hard = evaluate.threshold_metrics(y_va, (va_score >= thr).astype(int))

    print("== Final model: LogisticRegression (C=10), threshold frozen from validation ==")
    print(f"  operating threshold = {thr:.3f}\n")
    print(f"  {'metric':<14} {'validation':>11} {'TEST':>8}")
    print(f"  {'ROC-AUC':<14} {va_rank['roc_auc']:>11.3f} {te_rank['roc_auc']:>8.3f}")
    print(f"  {'Avg-Precision':<14} {va_rank['avg_precision']:>11.3f} {te_rank['avg_precision']:>8.3f}")
    print(f"  {'Precision':<14} {va_hard['precision']:>11.1%} {te_hard['precision']:>8.1%}")
    print(f"  {'Recall':<14} {va_hard['recall']:>11.1%} {te_hard['recall']:>8.1%}")
    print(f"  {'F1':<14} {va_hard['f1']:>11.3f} {te_hard['f1']:>8.3f}")
    print(f"  {'Accuracy':<14} {va_hard['accuracy']:>11.1%} {te_hard['accuracy']:>8.1%}")
    print(f"\n  TEST confusion [tn={te_hard['tn']} fp={te_hard['fp']} | fn={te_hard['fn']} tp={te_hard['tp']}]")
    print(f"  TEST positives (actual): {y_te.mean():.1%}   baseline floor to beat: AUC 0.617 / AP 0.405")

    # 4) calibration on test
    brier = brier_score_loss(y_te, te_score)
    prob_true, prob_pred = calibration_curve(y_te, te_score, n_bins=5, strategy="quantile")
    print(f"\n== Calibration (test) ==  Brier score = {brier:.3f}  (lower is better; 0=perfect)")
    print(f"  {'predicted avg':>13} {'actual rate':>12}")
    for pp, pt in zip(prob_pred, prob_true):
        print(f"  {pp:>13.1%} {pt:>12.1%}")

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(prob_pred, prob_true, "o-", color="#4c72b0", label="logreg")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed repeat rate")
    ax.set_title("Reliability (test)"); ax.legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "calibration_logreg.png", dpi=110)

    # 5) error analysis on test
    te = te_df.copy()
    te["prob"] = te_score
    te["pred"] = (te_score >= thr).astype(int)
    te["kind"] = np.select(
        [(te[LABEL] == 1) & (te.pred == 1), (te[LABEL] == 0) & (te.pred == 1),
         (te[LABEL] == 1) & (te.pred == 0), (te[LABEL] == 0) & (te.pred == 0)],
        ["TP", "FP", "FN", "TN"], default="?")
    print("\n== Error analysis (test): mean feature value by outcome ==")
    prof = te.groupby("kind")[FEATURES].mean().round(1)
    print(prof.to_string())
    print("\n  FN = returners we MISSED. FP = flagged but did NOT return.")
    print("  Representative false negatives (missed returners, lowest predicted prob):")
    fn = te[te.kind == "FN"].nsmallest(3, "prob")[FEATURES + ["prob"]]
    print(fn.round(2).to_string(index=False))

    # coefficients (standardised) for interpretability
    coefs = modeling.feature_importance(model, FEATURES)
    print("\n== Logistic-regression coefficients (standardised; + => more likely to repeat) ==")
    for k, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:<22} {v:+.3f}")

    # persist a metrics summary
    out = {
        "model": "LogisticRegression", "params": CHOSEN, "threshold": thr,
        "validation": {**va_rank, **{k: va_hard[k] for k in ("precision", "recall", "f1", "accuracy")}},
        "test": {**te_rank, **{k: te_hard[k] for k in ("precision", "recall", "f1", "accuracy")},
                 "confusion": {k: te_hard[k] for k in ("tn", "fp", "fn", "tp")}},
        "brier_test": brier,
        "coefficients": coefs,
    }
    (config.REPORTS_DIR / "test_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {config.REPORTS_DIR / 'test_metrics.json'} and figures/calibration_logreg.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
