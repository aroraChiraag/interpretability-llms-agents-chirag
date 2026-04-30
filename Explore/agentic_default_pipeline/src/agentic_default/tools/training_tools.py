"""CrewAI tools that wrap model training, metric extraction, and tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..ml_trainer import (
    SUPPORTED_MODELS,
    default_hyperparameters,
    merge_hyperparameters,
    train_and_evaluate,
)
from ..state import get_state


# ---------------------------------------------------------------------------
# Train models
# ---------------------------------------------------------------------------


class TrainModelsInput(BaseModel):
    """Input schema for :class:`TrainModelsTool`."""

    models: List[str] = Field(
        default=list(SUPPORTED_MODELS),
        description=(
            "Subset of supported model names to train. "
            f"Choose from {list(SUPPORTED_MODELS)}."
        ),
    )
    random_state: int = Field(default=42)
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory where metrics_report.json should be written.",
    )
    state_handle: str = Field(default="default")
    hyperparameters: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Optional per-model hyperparameter overrides. If omitted, the tool "
            "uses whatever overrides are currently stored in pipeline state "
            "(or the package defaults if none have been set)."
        ),
    )


class TrainModelsTool(BaseTool):
    """Train classifiers and store the metric report in pipeline state."""

    name: str = "train_models"
    description: str = (
        "Train one or more classification models on the loaded dataset and "
        "return a JSON report containing per-model metrics (accuracy, "
        "precision, recall, F1, ROC-AUC, average precision, balanced "
        "accuracy), confusion matrices, top feature importances, and a "
        "leaderboard. Honours hyperparameter overrides stored in pipeline "
        "state, or the ones supplied directly via the `hyperparameters` arg."
    )
    args_schema: Type[BaseModel] = TrainModelsInput

    def _run(
        self,
        models: Optional[List[str]] = None,
        random_state: int = 42,
        output_dir: Optional[str] = None,
        state_handle: str = "default",
        hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        state = get_state(state_handle)
        if state.dataset is None:
            return json.dumps(
                {"error": "Dataset not loaded — call load_dataset before train_models."}
            )

        # Hyperparameter precedence: explicit arg > pipeline state > defaults.
        params = hyperparameters or state.hyperparameters or default_hyperparameters()

        chosen = models or list(SUPPORTED_MODELS)
        report = train_and_evaluate(
            x_train=state.dataset.x_train,
            y_train=state.dataset.y_train,
            x_test=state.dataset.x_test,
            y_test=state.dataset.y_test,
            feature_names=state.dataset.feature_names,
            models=chosen,
            random_state=random_state,
            output_dir=Path(output_dir) if output_dir else None,
            hyperparameters=params,
        )
        state.metrics_report = report
        compact = _compact_report(report)
        return json.dumps(compact, indent=2)


def _compact_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the verbose classification_report dict from each model entry."""
    return {
        "leaderboard": report["leaderboard"],
        "best_model": report["best_model"],
        "hyperparameters_used": report.get("hyperparameters_used", {}),
        "models": [
            {
                "model_name": m["model_name"],
                "metrics": {
                    k: v
                    for k, v in m["metrics"].items()
                    if k != "classification_report"
                },
                "feature_importance": m["feature_importance"],
                "confusion_matrix": m["confusion_matrix"],
                "train_seconds": m["train_seconds"],
                "notes": m["notes"],
                "hyperparameters": m.get("hyperparameters", {}),
                # `predictions` is intentionally omitted — it's too large for
                # an LLM context and is only consumed by the fairness
                # pipeline, which reads from PipelineState directly.
            }
            for m in report["models"]
        ],
    }


# ---------------------------------------------------------------------------
# Get metrics
# ---------------------------------------------------------------------------


class GetMetricsInput(BaseModel):
    """Input schema for :class:`GetMetricsTool`."""

    state_handle: str = Field(default="default")
    include_classification_report: bool = Field(default=False)


class GetMetricsTool(BaseTool):
    """Return the previously computed metric report from pipeline state."""

    name: str = "get_metrics"
    description: str = (
        "Return the metric report produced by train_models. Call this if you "
        "need to re-read the metrics; you must call train_models first."
    )
    args_schema: Type[BaseModel] = GetMetricsInput

    def _run(
        self,
        state_handle: str = "default",
        include_classification_report: bool = False,
    ) -> str:
        report = get_state(state_handle).metrics_report
        if not report:
            return json.dumps({"error": "No metrics yet — call train_models first."})
        if include_classification_report:
            return json.dumps(report, indent=2)
        return json.dumps(_compact_report(report), indent=2)


# ---------------------------------------------------------------------------
# Hyperparameter tools
# ---------------------------------------------------------------------------


class UpdateHyperparametersInput(BaseModel):
    """Input schema for :class:`UpdateHyperparametersTool`."""

    overrides: Dict[str, Dict[str, Any]] = Field(
        description=(
            "Per-model hyperparameter overrides to merge into pipeline state. "
            'Example: {"random_forest": {"n_estimators": 500, "max_depth": 12}}.'
        )
    )
    state_handle: str = Field(default="default")


class UpdateHyperparametersTool(BaseTool):
    """Merge user-supplied hyperparameter overrides into pipeline state."""

    name: str = "update_hyperparameters"
    description: str = (
        "Merge per-model hyperparameter overrides into pipeline state so the "
        "next call to train_models picks them up. Use this when the user asks "
        "to change a hyperparameter (e.g. 'use 500 trees in random forest')."
    )
    args_schema: Type[BaseModel] = UpdateHyperparametersInput

    def _run(
        self,
        overrides: Dict[str, Dict[str, Any]],
        state_handle: str = "default",
    ) -> str:
        state = get_state(state_handle)
        merged = merge_hyperparameters({**state.hyperparameters, **overrides})
        # Ensure every override actually lands on top of any prior value.
        for model_name, params in (overrides or {}).items():
            if isinstance(params, dict):
                merged.setdefault(model_name, {}).update(params)
        state.hyperparameters = merged
        return json.dumps(
            {"hyperparameters": merged, "applied_overrides": overrides},
            indent=2,
        )


class GetHyperparametersInput(BaseModel):
    state_handle: str = Field(default="default")


class GetHyperparametersTool(BaseTool):
    """Return the current per-model hyperparameter configuration."""

    name: str = "get_hyperparameters"
    description: str = (
        "Return the per-model hyperparameters currently stored in pipeline "
        "state. Useful before training to confirm what will be used."
    )
    args_schema: Type[BaseModel] = GetHyperparametersInput

    def _run(self, state_handle: str = "default") -> str:
        state = get_state(state_handle)
        return json.dumps(state.hyperparameters or default_hyperparameters(), indent=2)


class ResetHyperparametersInput(BaseModel):
    state_handle: str = Field(default="default")


class ResetHyperparametersTool(BaseTool):
    """Reset pipeline-state hyperparameters to the package defaults."""

    name: str = "reset_hyperparameters"
    description: str = (
        "Restore the default hyperparameters for every model. Use this when "
        "the user asks to 'go back to defaults' or 'reset hyperparameters'."
    )
    args_schema: Type[BaseModel] = ResetHyperparametersInput
    def _run(self, state_handle: str = "default") -> str:
        state = get_state(state_handle)
        state.hyperparameters = default_hyperparameters()
        return json.dumps({"hyperparameters": state.hyperparameters}, indent=2)
