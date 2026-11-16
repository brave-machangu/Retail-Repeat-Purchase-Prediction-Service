"""Build and persist the deployable model bundle.

`build_bundle` fits the chosen logistic regression on TRAIN, freezes the decision
threshold on VALIDATION, then refits on TRAIN+VAL for deployment. The returned bundle
is a plain dict so it serialises cleanly with joblib and is easy to inspect/test.
"""
from __future__ import annotations

from datetime import datetime, timezone

import joblib
import pandas as pd

from . import config, evaluate, modeling

FEATURES = config.FEATURE_COLUMNS
LABEL = config.LABEL_COLUMN


def build_bundle(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Return a deployable bundle: pipeline + threshold + metadata."""
    # 1) fit on train, choose the operating threshold on validation (untainted)
    selector = modeling.make_logreg(**config.FINAL_MODEL)
    selector.fit(train_df[FEATURES], train_df[LABEL])
    val_score = selector.predict_proba(val_df[FEATURES])[:, 1]
    threshold, _ = evaluate.pick_threshold(val_df[LABEL].to_numpy(), val_score)

    # 2) refit on train+val for the deployed artifact (more data, both precede test)
    full = pd.concat([train_df, val_df], ignore_index=True)
    pipeline = modeling.make_logreg(**config.FINAL_MODEL)
    pipeline.fit(full[FEATURES], full[LABEL])

    return {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "features": list(FEATURES),
        "model_version": config.MODEL_VERSION,
        "params": dict(config.FINAL_MODEL),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train_rows": int(len(full)),
        "lookback_days": config.LOOKBACK_DAYS,
        "horizon_days": config.HORIZON_DAYS,
    }


def save_bundle(bundle: dict, path=None) -> None:
    path = path or config.MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def train_final(path=None) -> dict:
    """Load processed splits, build the bundle, save it, and return it."""
    train_df = pd.read_parquet(config.DATA_PROCESSED / "train.parquet")
    val_df = pd.read_parquet(config.DATA_PROCESSED / "val.parquet")
    bundle = build_bundle(train_df, val_df)
    save_bundle(bundle, path)
    return bundle
