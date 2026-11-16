"""Smoke tests for the model builders: valid probabilities, scaler present, importances."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_repeat import config, modeling  # noqa: E402


@pytest.fixture()
def tiny_xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((40, len(config.FEATURE_COLUMNS))), columns=config.FEATURE_COLUMNS)
    y = (X["n_orders"] > 0.5).astype(int).to_numpy()  # learnable signal
    return X, y


@pytest.mark.parametrize("builder", ["make_logreg", "make_tree", "make_forest"])
def test_predict_proba_is_valid(tiny_xy, builder):
    X, y = tiny_xy
    model = getattr(modeling, builder)()
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_logreg_has_scaler():
    model = modeling.make_logreg()
    assert "scaler" in model.named_steps  # scaling travels with the model


def test_feature_importance_covers_all_features(tiny_xy):
    X, y = tiny_xy
    for builder in ("make_logreg", "make_tree", "make_forest"):
        model = getattr(modeling, builder)()
        model.fit(X, y)
        imp = modeling.feature_importance(model, config.FEATURE_COLUMNS)
        assert set(imp) == set(config.FEATURE_COLUMNS)
