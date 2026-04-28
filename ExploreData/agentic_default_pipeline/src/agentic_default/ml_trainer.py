"""Train classification models and produce a JSON-friendly metrics report.

The ``train_and_evaluate`` function is the main entry point. It accepts the
arrays from :class:`agentic_default.data_loader.LoadedDataset` plus a list of
model names and returns a dictionary of metrics for each model. The dictionary
is what the CrewAI ExplainerAgent will consume.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier


SUPPORTED_MODELS = ("random_forest", "xgboost", "neural_network")


# ---------- result containers -----------------------------------------------


@dataclass
class ModelResult:
    """Per-model evaluation result."""

    model_name: str
    metrics: Dict[str, Any]
    feature_importance: List[Dict[str, Any]] = field(default_factory=list)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    train_seconds: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "confusion_matrix": self.confusion_matrix,
            "train_seconds": round(self.train_seconds, 3),
            "notes": self.notes,
        }


# ---------- model factories -------------------------------------------------


def _build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Construct a Random Forest with sensible defaults for this dataset."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def _build_xgboost(random_state: int = 42):
    """Construct an XGBoost classifier; falls back to GradientBoosting if absent.

    Returns
    -------
    object
        A fitted-or-fittable estimator with ``fit`` / ``predict_proba``.
    str
        A note describing which library is in use.
    """
    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier

        return (
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state,
            ),
            "xgboost not installed — using sklearn GradientBoostingClassifier as a substitute",
        )
    return (
        XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        ),
        "xgboost.XGBClassifier",
    )


def _build_neural_network(random_state: int = 42) -> MLPClassifier:
    """Construct a small MLP suitable for tabular binary classification."""
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=80,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
        verbose=False,
    )


# ---------- metric computation ----------------------------------------------


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Compute the standard binary-classification metric bundle."""
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["average_precision"] = float(average_precision_score(y_true, y_score))
    out["classification_report"] = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    return out


def _feature_importance_records(
    importances: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Return the top-k feature importances as JSON records."""
    order = np.argsort(importances)[::-1][:top_k]
    return [
        {
            "feature": feature_names[i],
            "importance": float(importances[i]),
            "rank": int(rank + 1),
        }
        for rank, i in enumerate(order)
    ]


# ---------- per-model training ----------------------------------------------


def _train_one(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    random_state: int,
) -> ModelResult:
    """Fit one model, evaluate on the test split, return a ModelResult."""
    note = ""
    if model_name == "random_forest":
        model = _build_random_forest(random_state)
    elif model_name == "xgboost":
        model, note = _build_xgboost(random_state)
    elif model_name == "neural_network":
        model = _build_neural_network(random_state)
    else:
        raise ValueError(
            f"Unknown model {model_name!r}. Choose from {SUPPORTED_MODELS}."
        )

    start = time.time()
    model.fit(x_train, y_train)
    train_seconds = time.time() - start

    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(x_test)
    else:
        y_score = None

    metrics = _compute_metrics(y_test, y_pred, y_score)

    importance: List[Dict[str, Any]] = []
    if hasattr(model, "feature_importances_"):
        importance = _feature_importance_records(
            np.asarray(model.feature_importances_), feature_names
        )
    elif hasattr(model, "coefs_"):
        # MLP: use first-layer absolute weight magnitude as a rough proxy.
        first_layer = np.abs(model.coefs_[0]).sum(axis=1)
        importance = _feature_importance_records(first_layer, feature_names)
        note = (note + "; " if note else "") + (
            "feature importance proxied via |W1| sum across hidden units"
        )

    cm = confusion_matrix(y_test, y_pred).tolist()

    return ModelResult(
        model_name=model_name,
        metrics=metrics,
        feature_importance=importance,
        confusion_matrix=cm,
        train_seconds=float(train_seconds),
        notes=note,
    )


# ---------- public API ------------------------------------------------------


def train_and_evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    models: Sequence[str] = SUPPORTED_MODELS,
    random_state: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train and evaluate one or more classifiers.

    Parameters
    ----------
    x_train, y_train, x_test, y_test : np.ndarray
        Train and test arrays.
    feature_names : sequence of str
        Names of the feature columns, used for importance reporting.
    models : sequence of str, default all supported
        Subset of ``SUPPORTED_MODELS`` to train.
    random_state : int, default 42
        Seed.
    output_dir : Path, optional
        If provided, the metric report is written to ``<dir>/metrics_report.json``.

    Returns
    -------
    dict
        Mapping with keys ``models`` (list of per-model dicts), ``leaderboard``
        (sorted by ROC-AUC descending), and ``best_model``.
    """
    results: List[ModelResult] = []
    for model_name in models:
        result = _train_one(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            feature_names=feature_names,
            random_state=random_state,
        )
        results.append(result)

    leaderboard = sorted(
        results,
        key=lambda r: (r.metrics.get("roc_auc", 0.0), r.metrics.get("f1", 0.0)),
        reverse=True,
    )
    best_model = leaderboard[0].model_name if leaderboard else None

    report = {
        "models": [r.to_dict() for r in results],
        "leaderboard": [
            {
                "model_name": r.model_name,
                "roc_auc": r.metrics.get("roc_auc"),
                "f1": r.metrics.get("f1"),
                "precision": r.metrics.get("precision"),
                "recall": r.metrics.get("recall"),
                "accuracy": r.metrics.get("accuracy"),
            }
            for r in leaderboard
        ],
        "best_model": best_model,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    return report
