"""Train classification models and produce a JSON-friendly metrics report.

The ``train_and_evaluate`` function is the main entry point. It accepts the
arrays from :class:`agentic_default.data_loader.LoadedDataset` plus a list of
model names and returns a dictionary of metrics for each model.

Hyperparameters can be overridden per-model via the ``hyperparameters`` arg
or via ``default_hyperparameters()`` (used by the UI form / chat tools).
"""

from __future__ import annotations

import copy
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


# ---------- defaults --------------------------------------------------------


def default_hyperparameters() -> Dict[str, Dict[str, Any]]:
    """Return a fresh dict of the default hyperparameters per model.

    The UI form populates itself from this; the chat coordinator merges
    user-requested overrides on top. Returning a deep copy each time avoids
    accidental shared mutation.
    """
    return copy.deepcopy(
        {
            "random_forest": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
            },
            "xgboost": {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": 5,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
            "neural_network": {
                "hidden_layer_sizes": [64, 32],
                "alpha": 1e-4,
                "learning_rate_init": 1e-3,
                "max_iter": 80,
                "batch_size": 128,
            },
        }
    )


def merge_hyperparameters(
    overrides: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Merge user overrides on top of the defaults, producing a full config."""
    merged = default_hyperparameters()
    if not overrides:
        return merged
    for model_name, params in overrides.items():
        if model_name not in merged:
            merged[model_name] = {}
        if not isinstance(params, dict):
            continue
        merged[model_name].update(params)
    return merged


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
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    #: Test-set predicted labels (0/1), aligned with x_test/y_test ordering.
    #: Used by the fairness pipeline; trimmed out of LLM-facing payloads.
    predictions: List[int] = field(default_factory=list)
    #: Predicted probabilities for class 1 (positive), aligned with predictions.
    #: Used by visualizations (PR curve, threshold analysis). Not serialised to JSON.
    probabilities: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary (probabilities excluded to keep files small)."""
        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "confusion_matrix": self.confusion_matrix,
            "train_seconds": round(self.train_seconds, 3),
            "notes": self.notes,
            "hyperparameters": self.hyperparameters,
            "predictions": self.predictions,
            # probabilities intentionally omitted from JSON output
        }


# ---------- model factories -------------------------------------------------


def _build_random_forest(random_state: int, params: Dict[str, Any]) -> RandomForestClassifier:
    """Construct a Random Forest using the supplied hyperparameters."""
    return RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        **params,
    )


def _build_xgboost(random_state: int, params: Dict[str, Any]):
    """Construct an XGBoost classifier; fall back to GradientBoosting if absent."""
    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier

        # GradientBoostingClassifier doesn't support all xgboost params — only
        # forward the overlap.
        compatible = {
            k: v
            for k, v in params.items()
            if k in {"n_estimators", "learning_rate", "max_depth"}
        }
        return (
            GradientBoostingClassifier(random_state=random_state, **compatible),
            "xgboost not installed — using sklearn GradientBoostingClassifier as a substitute",
        )
    return (
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            **params,
        ),
        "xgboost.XGBClassifier",
    )


def _build_neural_network(random_state: int, params: Dict[str, Any]) -> MLPClassifier:
    """Construct an MLP using the supplied hyperparameters."""
    # hidden_layer_sizes can come in as a list (JSON-friendly) — sklearn
    # accepts a tuple or a list, but normalise for cleanliness.
    if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], list):
        params = dict(params)
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    return MLPClassifier(
        activation="relu",
        solver="adam",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
        verbose=False,
        **params,
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
    params: Dict[str, Any],
    model_pickle_dir: Optional[Path] = None,
) -> ModelResult:
    """Fit one model, evaluate on the test split, return a ModelResult.

    If ``model_pickle_dir`` is given, the fitted estimator is pickled to
    ``<dir>/<model_name>.pkl`` so the Streamlit UI can offer a download
    button. Pickle errors are caught and surfaced in ``notes`` rather
    than failing the whole run.
    """
    note = ""
    if model_name == "random_forest":
        model = _build_random_forest(random_state, params)
    elif model_name == "xgboost":
        model, note = _build_xgboost(random_state, params)
    elif model_name == "neural_network":
        model = _build_neural_network(random_state, params)
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
        first_layer = np.abs(model.coefs_[0]).sum(axis=1)
        importance = _feature_importance_records(first_layer, feature_names)
        note = (note + "; " if note else "") + (
            "feature importance proxied via |W1| sum across hidden units"
        )

    cm = confusion_matrix(y_test, y_pred).tolist()

    if model_pickle_dir is not None:
        try:
            import pickle
            model_pickle_dir = Path(model_pickle_dir)
            model_pickle_dir.mkdir(parents=True, exist_ok=True)
            pkl_path = model_pickle_dir / f"{model_name}.pkl"
            with open(pkl_path, "wb") as fh:
                pickle.dump(model, fh)
        except Exception as exc:  # noqa: BLE001
            note = (note + "; " if note else "") + f"pickle failed: {exc!r}"

    return ModelResult(
        model_name=model_name,
        metrics=metrics,
        feature_importance=importance,
        confusion_matrix=cm,
        train_seconds=float(train_seconds),
        notes=note,
        hyperparameters=params,
        predictions=[int(v) for v in y_pred],
        probabilities=[float(v) for v in y_score] if y_score is not None else [],
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
    hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Train and evaluate one or more classifiers.

    Parameters
    ----------
    x_train, y_train, x_test, y_test : np.ndarray
        Train and test arrays.
    feature_names : sequence of str
        Names of the feature columns.
    models : sequence of str, default all supported
        Subset of ``SUPPORTED_MODELS`` to train.
    random_state : int, default 42
        Seed.
    output_dir : Path, optional
        If provided, the metric report is written to ``<dir>/metrics_report.json``.
    hyperparameters : dict, optional
        Per-model overrides (merged on top of :func:`default_hyperparameters`).
        Example::

            {"random_forest": {"n_estimators": 500, "max_depth": 12}}

    Returns
    -------
    dict
        Mapping with keys ``models`` (list of per-model dicts), ``leaderboard``
        (sorted by ROC-AUC descending), ``best_model``, and
        ``hyperparameters_used``.
    """
    full_params = merge_hyperparameters(hyperparameters)

    # Pickle download is disabled (workspace storage limits) — we no longer
    # persist trained estimators to disk. The optional `_train_one` hook is
    # retained for callers that pass their own dir, but train_and_evaluate
    # itself does not opt in.
    pickle_dir: Optional[Path] = None

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
            params=full_params.get(model_name, {}),
            model_pickle_dir=pickle_dir,
        )
        results.append(result)

    leaderboard = sorted(
        results,
        key=lambda r: (r.metrics.get("roc_auc", 0.0), r.metrics.get("f1", 0.0)),
        reverse=True,
    )
    best_model = leaderboard[0].model_name if leaderboard else None

    # Build the in-memory report. Probabilities are injected here but excluded
    # from to_dict() so they are never written to the on-disk JSON.
    model_dicts: List[Dict[str, Any]] = []
    for r in results:
        d = r.to_dict()
        if r.probabilities:
            d["probabilities"] = r.probabilities
        model_dicts.append(d)

    report = {
        "models": model_dicts,
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
        "hyperparameters_used": full_params,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Strip predictions/probabilities from the on-disk JSON — they are
        # large and only useful for downstream in-process consumers (the
        # fairness pipeline reads them straight from PipelineState).
        on_disk = {
            **report,
            "models": [
                {k: v for k, v in m.items() if k not in {"predictions", "probabilities"}}
                for m in report["models"]
            ],
        }
        (output_dir / "metrics_report.json").write_text(
            json.dumps(on_disk, indent=2), encoding="utf-8"
        )

    return report
