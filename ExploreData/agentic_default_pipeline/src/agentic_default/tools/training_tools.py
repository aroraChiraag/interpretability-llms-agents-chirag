"""CrewAI tools that wrap model training and metric extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..ml_trainer import SUPPORTED_MODELS, train_and_evaluate
from ..state import get_state


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


class TrainModelsTool(BaseTool):
    """Train classifiers and store the metric report in pipeline state."""

    name: str = "train_models"
    description: str = (
        "Train one or more classification models on the loaded dataset and "
        "return a JSON report containing per-model metrics (accuracy, "
        "precision, recall, F1, ROC-AUC, average precision, balanced "
        "accuracy), confusion matrices, top feature importances, and a "
        "leaderboard."
    )
    args_schema: Type[BaseModel] = TrainModelsInput

    def _run(
        self,
        models: Optional[List[str]] = None,
        random_state: int = 42,
        output_dir: Optional[str] = None,
        state_handle: str = "default",
    ) -> str:
        state = get_state(state_handle)
        if state.dataset is None:
            return json.dumps(
                {"error": "Dataset not loaded — call load_dataset before train_models."}
            )

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
        )
        state.metrics_report = report
        # Trim the verbose classification_report dict from each model entry
        # before returning to the LLM — the explainer doesn't need that depth.
        compact = {
            "leaderboard": report["leaderboard"],
            "best_model": report["best_model"],
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
                }
                for m in report["models"]
            ],
        }
        return json.dumps(compact, indent=2)


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
            return json.dumps(
                {"error": "No metrics yet — call train_models first."}
            )
        if include_classification_report:
            return json.dumps(report, indent=2)
        compact = {
            "leaderboard": report["leaderboard"],
            "best_model": report["best_model"],
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
                }
                for m in report["models"]
            ],
        }
        return json.dumps(compact, indent=2)
