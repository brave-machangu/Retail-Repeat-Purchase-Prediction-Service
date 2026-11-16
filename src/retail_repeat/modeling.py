"""Model builders. Each returns a fitted-ready estimator/pipeline with a uniform
.fit / .predict_proba interface so training, comparison, and serving treat them alike.

Logistic regression carries its own StandardScaler so scaling is fit on train data
only and can never be forgotten at serving time. Trees are scale-invariant and used bare.
All randomness is seeded from config.RANDOM_SEED for reproducibility.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config


def make_logreg(C: float = 1.0, class_weight=None) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, class_weight=class_weight, max_iter=2000,
            random_state=config.RANDOM_SEED,
        )),
    ])


def make_tree(max_depth=None, min_samples_leaf: int = 1, class_weight=None) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        class_weight=class_weight, random_state=config.RANDOM_SEED,
    )


def make_forest(n_estimators: int = 300, max_depth=None, min_samples_leaf: int = 1,
                class_weight=None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        class_weight=class_weight, max_features="sqrt",
        random_state=config.RANDOM_SEED, n_jobs=-1,
    )


def feature_importance(model, feature_names) -> dict:
    """Return a readable importance mapping for whichever model type was passed."""
    clf = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    if hasattr(clf, "coef_"):            # logistic regression: standardised coefficients
        vals = clf.coef_.ravel()
    elif hasattr(clf, "feature_importances_"):  # trees / forests
        vals = clf.feature_importances_
    else:
        return {}
    return dict(zip(feature_names, (float(v) for v in vals)))
