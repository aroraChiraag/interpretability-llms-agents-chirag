"""Optimizer — apply SMOTE / Focal-Loss reweighting / GridSearch and re-evaluate.

Designed to slot into the existing pipeline: each public function returns a
``ModelResult``-shaped dict that the Streamlit UI can drop straight into the
``metrics_report["models"]`` list, replacing the prior entry for the same
model name.

We deliberately keep all of this in its own module so ``ml_trainer.py``
stays the single source of truth for the *baseline* training loop. The
optimizer reuses ``ml_trainer``'s metric / importance helpers and model
factories — no duplication.

Three techniques are supported (per model):

- **SMOTE** — synthetic minority over-sampling on the training set
  (`imblearn.over_sampling.SMOTE`). Requires the optional ``imbalanced-learn``
  dependency. Re-trains the chosen model on the resampled set.

- **Focal-Loss reweighting** — approximates focal loss via
  ``sample_weight``: `w_i = α (1 - p_class)^γ` for positives, mirrored for
  negatives. This is a frequency-based proxy: it does not implement the
  per-example modulating factor of true focal loss (we don't have model
  probabilities at fit time), but it preserves focal loss's intent —
  up-weight the rare class. Works for RF, XGBoost, and (via ``fit``'s
  optional ``sample_weight``) MLPClassifier.

- **GridSearch** — wraps ``sklearn.model_selection.GridSearchCV`` with a
  per-model default param grid and a user-chosen number of CV folds /
  scoring metric. Returns the best estimator's metrics plus the chosen
  hyperparameters.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GridSearchCV

from .ml_trainer import (
    SUPPORTED_MODELS,
    _build_neural_network,
    _build_random_forest,
    _build_xgboost,
    _compute_metrics,
    _feature_importance_records,
    default_hyperparameters,
)


SUPPORTED_TECHNIQUES = (
    "smote",
    "focal_loss",
    "grid_search",
    "optuna",
    "class_weight",
    "early_stopping",
    "threshold_tuning",
    "feature_selection",
)


# ---------- helpers ---------------------------------------------------------


def _instantiate_model(model_name: str, hyperparameters: Dict[str, Dict[str, Any]],
                       random_state: int) -> Tuple[Any, str]:
    """Build a fresh sklearn-style estimator for the given model_name."""
    params = hyperparameters.get(model_name, {})
    if model_name == "random_forest":
        return _build_random_forest(random_state, params), ""
    if model_name == "xgboost":
        return _build_xgboost(random_state, params)  # already returns (model, note)
    if model_name == "neural_network":
        return _build_neural_network(random_state, params), ""
    raise ValueError(f"Unknown model {model_name!r}; choose from {SUPPORTED_MODELS}.")


def _evaluate(model: Any,
              x_test: np.ndarray,
              y_test: np.ndarray,
              feature_names: Sequence[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]],
                                                     List[List[int]], List[int]]:
    """Compute metrics, feature-importance records, confusion matrix, predictions."""
    from sklearn.metrics import confusion_matrix
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
        importance = _feature_importance_records(
            np.abs(model.coefs_[0]).sum(axis=1), feature_names
        )
    cm = confusion_matrix(y_test, y_pred).tolist()
    return metrics, importance, cm, [int(v) for v in y_pred]


def _result_dict(model_name: str, metrics, importance, cm, train_seconds,
                 notes, hyperparameters, predictions) -> Dict[str, Any]:
    """Shape an optimization result so it slots into metrics_report['models']."""
    return {
        "model_name": model_name,
        "metrics": metrics,
        "feature_importance": importance,
        "confusion_matrix": cm,
        "train_seconds": round(float(train_seconds), 3),
        "notes": notes,
        "hyperparameters": hyperparameters,
        "predictions": predictions,
    }


# ---------- SMOTE ----------------------------------------------------------


def apply_smote(x_train: np.ndarray, y_train: np.ndarray,
                random_state: int = 42, k_neighbors: int = 5,
                sampling_strategy: str | float = "auto") -> Tuple[np.ndarray, np.ndarray]:
    """Resample a training set using SMOTE.

    Raises a friendly RuntimeError if ``imbalanced-learn`` is not installed.
    """
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "SMOTE requires the imbalanced-learn package. "
            "Install with: uv pip install imbalanced-learn"
        ) from exc
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors,
                  sampling_strategy=sampling_strategy)
    return smote.fit_resample(x_train, y_train)


def smote_optimize(model_name: str,
                   x_train: np.ndarray, y_train: np.ndarray,
                   x_test: np.ndarray, y_test: np.ndarray,
                   feature_names: Sequence[str],
                   hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
                   random_state: int = 42,
                   k_neighbors: int = 5,
                   sampling_strategy: str | float = "auto") -> Dict[str, Any]:
    """SMOTE-resample then re-train ``model_name`` on the balanced set."""
    hp = hyperparameters or default_hyperparameters()
    model, note = _instantiate_model(model_name, hp, random_state)
    x_res, y_res = apply_smote(x_train, y_train,
                               random_state=random_state,
                               k_neighbors=k_neighbors,
                               sampling_strategy=sampling_strategy)
    start = time.time()
    model.fit(x_res, y_res)
    train_seconds = time.time() - start
    metrics, importance, cm, predictions = _evaluate(model, x_test, y_test, feature_names)
    notes_parts = [note] if note else []
    notes_parts.append(
        f"Trained with SMOTE (k_neighbors={k_neighbors}, "
        f"sampling_strategy={sampling_strategy!r}); "
        f"resampled training size {len(y_train)} → {len(y_res)}"
    )
    return _result_dict(
        model_name=model_name,
        metrics=metrics,
        importance=importance,
        cm=cm,
        train_seconds=train_seconds,
        notes="; ".join(notes_parts),
        hyperparameters=hp.get(model_name, {}),
        predictions=predictions,
    )


# ---------- Focal-loss reweighting -----------------------------------------


def focal_loss_sample_weights(y_train: np.ndarray,
                              alpha: float = 0.25,
                              gamma: float = 2.0) -> np.ndarray:
    """Frequency-based focal-loss approximation as per-row sample weights.

    True focal loss reweights each example by ``(1 - p_t)^γ`` where ``p_t``
    is the predicted probability of the *true* class. We don't have those
    probabilities at fit-time, so we substitute the *class* frequency as a
    coarse proxy. This preserves focal loss's intent — up-weight the rare
    class — without needing a custom training loop.
    """
    y = np.asarray(y_train).astype(int)
    pos_freq = float((y == 1).mean()) if len(y) else 0.5
    neg_freq = 1.0 - pos_freq
    pos_w = float(alpha * ((1.0 - pos_freq) ** gamma))
    neg_w = float((1.0 - alpha) * ((1.0 - neg_freq) ** gamma))
    # Avoid all-zero weights if alpha or freq is degenerate.
    if pos_w == 0 and neg_w == 0:
        pos_w = neg_w = 1.0
    return np.where(y == 1, pos_w, neg_w).astype(float)


def focal_loss_optimize(model_name: str,
                        x_train: np.ndarray, y_train: np.ndarray,
                        x_test: np.ndarray, y_test: np.ndarray,
                        feature_names: Sequence[str],
                        hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
                        random_state: int = 42,
                        alpha: float = 0.25,
                        gamma: float = 2.0) -> Dict[str, Any]:
    """Re-train ``model_name`` with focal-loss-style sample weights."""
    hp = hyperparameters or default_hyperparameters()
    model, note = _instantiate_model(model_name, hp, random_state)
    weights = focal_loss_sample_weights(y_train, alpha=alpha, gamma=gamma)

    fit_kwargs = {"sample_weight": weights}
    notes_parts = [note] if note else []

    start = time.time()
    try:
        model.fit(x_train, y_train, **fit_kwargs)
    except TypeError:
        # Some versions of MLPClassifier do not accept sample_weight; fall back.
        model.fit(x_train, y_train)
        notes_parts.append(
            "this estimator does not accept sample_weight; trained without focal weighting"
        )
    train_seconds = time.time() - start

    metrics, importance, cm, predictions = _evaluate(model, x_test, y_test, feature_names)
    notes_parts.append(
        f"Focal-loss approx via sample_weight (alpha={alpha}, gamma={gamma}); "
        f"freq-proxy weights pos={weights[y_train == 1].mean():.4f} "
        f"neg={weights[y_train == 0].mean():.4f}"
    )
    return _result_dict(
        model_name=model_name,
        metrics=metrics,
        importance=importance,
        cm=cm,
        train_seconds=train_seconds,
        notes="; ".join(notes_parts),
        hyperparameters=hp.get(model_name, {}),
        predictions=predictions,
    )


# ---------- GridSearch -----------------------------------------------------


# Modest default grids — small enough to finish in a Streamlit session.
DEFAULT_PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 4],
    },
    "xgboost": {
        "n_estimators": [200, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 7],
    },
    "neural_network": {
        "hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
        "alpha": [1e-4, 1e-3],
        "learning_rate_init": [1e-3, 5e-4],
    },
}


def _build_estimator_for_grid(model_name: str, random_state: int) -> Tuple[Any, str]:
    """Build a fresh estimator with default hyperparameters for grid search."""
    hp = default_hyperparameters()
    return _instantiate_model(model_name, hp, random_state)


def grid_search_optimize(model_name: str,
                         x_train: np.ndarray, y_train: np.ndarray,
                         x_test: np.ndarray, y_test: np.ndarray,
                         feature_names: Sequence[str],
                         param_grid: Optional[Dict[str, list]] = None,
                         cv_folds: int = 3,
                         scoring: str = "roc_auc",
                         random_state: int = 42,
                         n_jobs: int = -1) -> Dict[str, Any]:
    """Run sklearn's GridSearchCV for ``model_name`` and return the best fit."""
    estimator, note = _build_estimator_for_grid(model_name, random_state)
    grid = param_grid or DEFAULT_PARAM_GRIDS.get(model_name, {})
    if not grid:
        raise ValueError(f"No default param grid for {model_name!r}; pass param_grid.")
    notes_parts = [note] if note else []

    start = time.time()
    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(x_train, y_train)
    train_seconds = time.time() - start
    best_model = search.best_estimator_

    metrics, importance, cm, predictions = _evaluate(best_model, x_test, y_test, feature_names)
    notes_parts.append(
        f"GridSearchCV cv={cv_folds} scoring={scoring} "
        f"best_cv_score={round(search.best_score_, 4)} "
        f"best_params={search.best_params_}"
    )

    chosen_hp = dict(search.best_params_)
    if "hidden_layer_sizes" in chosen_hp and isinstance(chosen_hp["hidden_layer_sizes"], tuple):
        chosen_hp["hidden_layer_sizes"] = list(chosen_hp["hidden_layer_sizes"])

    return _result_dict(
        model_name=model_name,
        metrics=metrics,
        importance=importance,
        cm=cm,
        train_seconds=train_seconds,
        notes="; ".join(notes_parts),
        hyperparameters=chosen_hp,
        predictions=predictions,
    )



# ---------- Bayesian (Optuna) search ---------------------------------------
#
# This block integrates a teammate's Optuna-based contribution. The parameter
# spaces (RF / XGBoost / NN) are credited to the original `optimize.py` they
# wrote; we kept their search ranges and re-mapped the result back through
# our existing model factories so the xgboost-missing fallback, prediction
# capture, and result-dict shape stay consistent with the other techniques.


def optuna_optimize(model_name: str,
                    x_train: np.ndarray, y_train: np.ndarray,
                    x_test: np.ndarray, y_test: np.ndarray,
                    feature_names: Sequence[str],
                    hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
                    random_state: int = 42,
                    n_trials: int = 15,
                    timeout: int = 600,
                    scoring: str = "f1",
                    cv_folds: int = 3) -> Dict[str, Any]:
    """Run Bayesian hyperparameter search via Optuna, then re-train and evaluate.

    Search spaces (per teammate contribution):

    - random_forest : n_estimators 100–500 step 100, max_depth 5–20,
                       min_samples_split 2–10, class_weight=balanced.
    - xgboost       : n_estimators 100–500 step 100, learning_rate 0.01–0.2
                       log-uniform, max_depth 3–10.
    - neural_network: 1 or 2 hidden layers, 32–128 units each, alpha
                       1e-4–1e-2 log-uniform, learning_rate_init 1e-3–1e-2
                       log-uniform, max_iter 200, early_stopping=True.
    """
    try:
        import optuna  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is not installed. Install with: uv pip install optuna"
        ) from exc
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    def objective(trial: "optuna.Trial") -> float:
        if model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "class_weight": "balanced",
            }
            clf = RandomForestClassifier(random_state=random_state, n_jobs=-1, **params)
        elif model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
            }
            try:
                from xgboost import XGBClassifier  # type: ignore
                clf = XGBClassifier(
                    random_state=random_state,
                    n_jobs=-1,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    **params,
                )
            except ImportError:
                # Fallback to sklearn so the bootcamp env without xgboost still works.
                from sklearn.ensemble import GradientBoostingClassifier
                clf = GradientBoostingClassifier(random_state=random_state, **params)
        elif model_name == "neural_network":
            n_layers = trial.suggest_int("n_layers", 1, 2)
            layers = tuple(
                trial.suggest_int(f"n_units_l{i}", 32, 128) for i in range(n_layers)
            )
            params = {
                "hidden_layer_sizes": layers,
                "alpha": trial.suggest_float("alpha", 1e-4, 1e-2, log=True),
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 1e-3, 1e-2, log=True
                ),
                "max_iter": 200,
                "early_stopping": True,
            }
            clf = MLPClassifier(random_state=random_state, **params)
        else:
            return 0.0
        return float(
            cross_val_score(
                clf, x_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=-1
            ).mean()
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Reconstruct the teammate's per-trial layer encoding back into a single
    # `hidden_layer_sizes` tuple before handing the result to our factories.
    final_params: Dict[str, Any] = dict(study.best_params)
    if model_name == "neural_network" and "n_layers" in final_params:
        n_layers = final_params.pop("n_layers")
        layers: List[int] = []
        for i in range(n_layers):
            key = f"n_units_l{i}"
            if key in final_params:
                layers.append(int(final_params.pop(key)))
        final_params["hidden_layer_sizes"] = layers
        # Match MLPClassifier defaults set inside our factory.
        final_params.setdefault("max_iter", 200)

    # Build the final estimator through our shared factories so xgboost-fallback
    # / prediction capture / result shape stay consistent across techniques.
    hp = hyperparameters or default_hyperparameters()
    hp = {**hp, model_name: {**hp.get(model_name, {}), **final_params}}
    model, note = _instantiate_model(model_name, hp, random_state)
    notes_parts = [note] if note else []

    start = time.time()
    model.fit(x_train, y_train)
    train_seconds = time.time() - start
    metrics, importance, cm, predictions = _evaluate(model, x_test, y_test, feature_names)

    notes_parts.append(
        f"Optuna Bayesian search (teammate-contributed search space): "
        f"trials={n_trials} timeout={timeout}s scoring={scoring} cv={cv_folds} "
        f"best_cv_score={round(study.best_value, 4)} best_params={final_params}"
    )
    return _result_dict(
        model_name=model_name,
        metrics=metrics,
        importance=importance,
        cm=cm,
        train_seconds=train_seconds,
        notes="; ".join(notes_parts),
        hyperparameters=final_params,
        predictions=predictions,
    )


# ---------- Class Weight Balancing -----------------------------------------


def class_weight_optimize(
    model_name: str,
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    feature_names: Sequence[str],
    hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Re-train with class_weight='balanced' (RF) or scale_pos_weight (XGBoost).

    Not supported for Neural Network — MLPClassifier has no class_weight param.
    """
    if model_name not in ("random_forest", "xgboost"):
        raise ValueError(
            f"Class Weight Balancing is only supported for random_forest and xgboost, "
            f"not {model_name!r}."
        )
    hp = hyperparameters or default_hyperparameters()
    notes_parts: List[str] = []

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        params = {**hp.get("random_forest", {}), "class_weight": "balanced"}
        model: Any = RandomForestClassifier(random_state=random_state, n_jobs=-1, **params)
        notes_parts.append("Trained with class_weight='balanced'")
    else:  # xgboost
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        spw = neg / max(pos, 1)
        params_xgb = {**hp.get("xgboost", {}), "scale_pos_weight": spw}
        try:
            from xgboost import XGBClassifier  # type: ignore
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
                **params_xgb,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            compatible = {
                k: v for k, v in hp.get("xgboost", {}).items()
                if k in {"n_estimators", "learning_rate", "max_depth"}
            }
            model = GradientBoostingClassifier(random_state=random_state, **compatible)
            notes_parts.append(
                "xgboost not installed — using sklearn GradientBoostingClassifier as substitute"
            )
        notes_parts.append(f"Trained with scale_pos_weight={spw:.3f} (neg/pos ratio)")

    start = time.time()
    model.fit(x_train, y_train)
    train_seconds = time.time() - start
    metrics, importance, cm, predictions = _evaluate(model, x_test, y_test, feature_names)

    return _result_dict(
        model_name=model_name,
        metrics=metrics,
        importance=importance,
        cm=cm,
        train_seconds=train_seconds,
        notes="; ".join(filter(None, notes_parts)),
        hyperparameters=hp.get(model_name, {}),
        predictions=predictions,
    )


# ---------- Early Stopping -------------------------------------------------


def early_stopping_optimize(
    model_name: str,
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    feature_names: Sequence[str],
    hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
    random_state: int = 42,
    early_stopping_rounds: int = 10,
    validation_fraction: float = 0.1,
) -> Dict[str, Any]:
    """Re-train with early stopping.

    XGBoost: uses eval_set + early_stopping_rounds in fit().
    Neural Network: sets early_stopping=True + n_iter_no_change.
    Not supported for Random Forest — trees are not trained iteratively.
    """
    if model_name not in ("xgboost", "neural_network"):
        raise ValueError(
            f"Early Stopping is only supported for xgboost and neural_network, "
            f"not {model_name!r}."
        )
    hp = hyperparameters or default_hyperparameters()
    notes_parts: List[str] = []

    if model_name == "xgboost":
        from sklearn.model_selection import train_test_split
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train, y_train,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=y_train,
        )
        try:
            from xgboost import XGBClassifier  # type: ignore
            model: Any = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
                **hp.get("xgboost", {}),
            )
            start = time.time()
            try:
                model.fit(
                    x_tr, y_tr,
                    eval_set=[(x_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
            except TypeError:
                # Some XGBoost versions use the constructor parameter instead.
                model.set_params(early_stopping_rounds=early_stopping_rounds)
                model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)
            train_seconds = time.time() - start
            notes_parts.append(
                f"XGBoost early_stopping_rounds={early_stopping_rounds}, "
                f"val_fraction={validation_fraction:.0%}"
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            compatible = {
                k: v for k, v in hp.get("xgboost", {}).items()
                if k in {"n_estimators", "learning_rate", "max_depth"}
            }
            model = GradientBoostingClassifier(random_state=random_state, **compatible)
            start = time.time()
            model.fit(x_tr, y_tr)
            train_seconds = time.time() - start
            notes_parts.append(
                "xgboost not installed — early stopping unavailable; used GradientBoosting"
            )
        metrics, importance, cm, predictions = _evaluate(model, x_test, y_test, feature_names)

    else:  # neural_network
        params = {
            **hp.get("neural_network", {}),
            "early_stopping": True,
            "n_iter_no_change": early_stopping_rounds,
            "validation_fraction": validation_fraction,
        }
        model = _build_neural_network(random_state, params)
        start = time.time()
        model.fit(x_train, y_train)
        train_seconds = time.time() - start
        metrics, importance, cm, predictions = _evaluate(model, x_test, y_test, feature_names)
        notes_parts.append(
            f"Neural Network early_stopping=True, "
            f"n_iter_no_change={early_stopping_rounds}, "
            f"validation_fraction={validation_fraction:.0%}"
        )

    return _result_dict(
        model_name=model_name,
        metrics=metrics,
        importance=importance,
        cm=cm,
        train_seconds=train_seconds,
        notes="; ".join(filter(None, notes_parts)),
        hyperparameters=hp.get(model_name, {}),
        predictions=predictions,
    )


# ---------- Threshold Tuning -----------------------------------------------


def threshold_tuning_optimize(
    model_name: str,
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    feature_names: Sequence[str],
    hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
    random_state: int = 42,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Train model normally, then apply a custom decision threshold at prediction time."""
    from sklearn.metrics import confusion_matrix as _cm_fn
    hp = hyperparameters or default_hyperparameters()
    model, note = _instantiate_model(model_name, hp, random_state)
    notes_parts: List[str] = [note] if note else []

    start = time.time()
    model.fit(x_train, y_train)
    train_seconds = time.time() - start

    if hasattr(model, "predict_proba"):
        y_score: Optional[np.ndarray] = model.predict_proba(x_test)[:, 1]
        y_pred = (y_score >= threshold).astype(int)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(x_test)
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_score = None
        y_pred = model.predict(x_test)

    metrics = _compute_metrics(y_test, y_pred, y_score)
    importance: List[Dict[str, Any]] = []
    if hasattr(model, "feature_importances_"):
        importance = _feature_importance_records(
            np.asarray(model.feature_importances_), feature_names
        )
    elif hasattr(model, "coefs_"):
        importance = _feature_importance_records(
            np.abs(model.coefs_[0]).sum(axis=1), feature_names
        )
    cm = _cm_fn(y_test, y_pred).tolist()
    predictions = [int(v) for v in y_pred]

    direction = (
        "↑ precision bias" if threshold > 0.5
        else "↑ recall bias" if threshold < 0.5
        else "default (no bias)"
    )
    notes_parts.append(
        f"Decision threshold={threshold:.2f} ({direction})"
    )
    return _result_dict(
        model_name=model_name,
        metrics=metrics,
        importance=importance,
        cm=cm,
        train_seconds=train_seconds,
        notes="; ".join(notes_parts),
        hyperparameters=hp.get(model_name, {}),
        predictions=predictions,
    )


# ---------- Feature Selection ----------------------------------------------


def feature_selection_optimize(
    model_name: str,
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    feature_names: Sequence[str],
    hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
    random_state: int = 42,
    excluded_features: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Drop specified features by name then retrain the model."""
    hp = hyperparameters or default_hyperparameters()
    excluded_upper = {f.upper() for f in (excluded_features or [])}
    feature_names_list = list(feature_names)

    keep_idx = [
        i for i, name in enumerate(feature_names_list)
        if name.upper() not in excluded_upper
    ]
    kept_names = [feature_names_list[i] for i in keep_idx]

    if not kept_names:
        raise ValueError("All features were excluded — at least one feature must remain.")

    x_tr = x_train[:, keep_idx]
    x_te = x_test[:, keep_idx]

    model, note = _instantiate_model(model_name, hp, random_state)
    notes_parts: List[str] = [note] if note else []

    start = time.time()
    model.fit(x_tr, y_train)
    train_seconds = time.time() - start
    metrics, importance, cm, predictions = _evaluate(model, x_te, y_test, kept_names)

    dropped = sorted(excluded_upper) or ["none"]
    notes_parts.append(
        f"Feature selection: excluded [{', '.join(dropped)}]; "
        f"trained on {len(kept_names)}/{len(feature_names_list)} features"
    )
    return _result_dict(
        model_name=model_name,
        metrics=metrics,
        importance=importance,
        cm=cm,
        train_seconds=train_seconds,
        notes="; ".join(filter(None, notes_parts)),
        hyperparameters=hp.get(model_name, {}),
        predictions=predictions,
    )


# ---------- Unified entry point --------------------------------------------


def optimize(technique: str,
             model_name: str,
             x_train: np.ndarray, y_train: np.ndarray,
             x_test: np.ndarray, y_test: np.ndarray,
             feature_names: Sequence[str],
             hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
             random_state: int = 42,
             technique_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dispatch to the correct optimizer technique."""
    technique = (technique or "").lower()
    technique_params = technique_params or {}
    if technique not in SUPPORTED_TECHNIQUES:
        raise ValueError(
            f"Unknown technique {technique!r}. Choose from {SUPPORTED_TECHNIQUES}."
        )
    if technique == "smote":
        return smote_optimize(
            model_name=model_name,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            feature_names=feature_names,
            hyperparameters=hyperparameters,
            random_state=random_state,
            k_neighbors=int(technique_params.get("k_neighbors", 5)),
            sampling_strategy=technique_params.get("sampling_strategy", "auto"),
        )
    if technique == "focal_loss":
        return focal_loss_optimize(
            model_name=model_name,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            feature_names=feature_names,
            hyperparameters=hyperparameters,
            random_state=random_state,
            alpha=float(technique_params.get("alpha", 0.25)),
            gamma=float(technique_params.get("gamma", 2.0)),
        )
    if technique == "optuna":
        return optuna_optimize(
            model_name=model_name,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            feature_names=feature_names,
            hyperparameters=hyperparameters,
            random_state=random_state,
            n_trials=int(technique_params.get("n_trials", 15)),
            timeout=int(technique_params.get("timeout", 600)),
            scoring=technique_params.get("scoring", "f1"),
            cv_folds=int(technique_params.get("cv_folds", 3)),
        )
    if technique == "class_weight":
        return class_weight_optimize(
            model_name=model_name,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            feature_names=feature_names,
            hyperparameters=hyperparameters,
            random_state=random_state,
        )
    if technique == "early_stopping":
        return early_stopping_optimize(
            model_name=model_name,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            feature_names=feature_names,
            hyperparameters=hyperparameters,
            random_state=random_state,
            early_stopping_rounds=int(technique_params.get("early_stopping_rounds", 10)),
            validation_fraction=float(technique_params.get("validation_fraction", 0.1)),
        )
    if technique == "threshold_tuning":
        return threshold_tuning_optimize(
            model_name=model_name,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            feature_names=feature_names,
            hyperparameters=hyperparameters,
            random_state=random_state,
            threshold=float(technique_params.get("threshold", 0.5)),
        )
    if technique == "feature_selection":
        return feature_selection_optimize(
            model_name=model_name,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            feature_names=feature_names,
            hyperparameters=hyperparameters,
            random_state=random_state,
            excluded_features=technique_params.get("excluded_features", []),
        )
    return grid_search_optimize(
        model_name=model_name,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        feature_names=feature_names,
        param_grid=technique_params.get("param_grid"),
        cv_folds=int(technique_params.get("cv_folds", 3)),
        scoring=technique_params.get("scoring", "roc_auc"),
        random_state=random_state,
        n_jobs=int(technique_params.get("n_jobs", -1)),
    )
