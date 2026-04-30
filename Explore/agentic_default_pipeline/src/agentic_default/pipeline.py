"""End-to-end CrewAI pipeline: Data → Trainer → Explainer.

Each phase is its own Crew with a single agent. We run them sequentially so
state passes through the in-memory ``PipelineState`` (large numpy arrays) and
through JSON strings between agents (small structured outputs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents.data_agent import DataAgent
from .agents.explainer_agent import ExplainerAgent
from .agents.trainer_agent import TrainerAgent
from .ml_trainer import SUPPORTED_MODELS
from .state import get_state, reset_state


@dataclass
class PipelineRun:
    """Aggregated outputs of one end-to-end run."""

    dataset_summary: Dict[str, Any] = field(default_factory=dict)
    metrics_report: Dict[str, Any] = field(default_factory=dict)
    explanation_markdown: str = ""
    raw_outputs: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_summary": self.dataset_summary,
            "metrics_report": self.metrics_report,
            "explanation_markdown": self.explanation_markdown,
        }


def _crew_output_text(crew_result: Any) -> str:
    """Extract the raw string from a CrewAI ``CrewOutput``-like object."""
    return getattr(crew_result, "raw", None) or str(crew_result)


def _maybe_parse_json(text: str) -> Any:
    """Best-effort JSON parse — returns dict or {} on failure."""
    if not text:
        return {}
    candidate = text.strip()
    # Strip markdown code fences if the agent ignored instructions.
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


def run_pipeline(
    csv_path: Optional[str] = None,
    models: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    gemini_model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    state_handle: str = "default",
    fresh_state: bool = True,
) -> PipelineRun:
    """Run the full Data → Trainer → Explainer crew.

    Parameters
    ----------
    csv_path : str, optional
        Override path to the source CSV.
    models : list of str, optional
        Which classifiers to train. Defaults to all supported.
    output_dir : str, optional
        Directory where dataset/metrics snapshots and the final
        ``explanation.md`` will be written.
    gemini_model : str, default 'gemini-2.5-flash'
        Gemini model name used by every agent.
    api_key : str, optional
        Gemini API key. Falls back to ``GEMINI_API_KEY`` env var.
    state_handle : str
        Pipeline state slot. Use distinct handles for parallel runs.
    fresh_state : bool, default True
        Clear any existing in-memory state for the handle before starting.

    Returns
    -------
    PipelineRun
        Aggregated outputs from the three agents.
    """
    if fresh_state:
        reset_state(state_handle)

    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    # ---- 1. Data agent --------------------------------------------------
    data_agent = DataAgent(
        model=gemini_model, api_key=api_key, state_handle=state_handle
    )
    data_out = data_agent.run(
        csv_path=csv_path,
        snapshot_dir=str(output_path) if output_path else None,
    )
    data_text = _crew_output_text(data_out)
    dataset_summary = _maybe_parse_json(data_text)

    # ---- 2. Trainer agent ------------------------------------------------
    trainer_agent = TrainerAgent(
        model=gemini_model, api_key=api_key, state_handle=state_handle
    )
    trainer_out = trainer_agent.run(
        models=models or list(SUPPORTED_MODELS),
        output_dir=str(output_path) if output_path else None,
    )
    trainer_text = _crew_output_text(trainer_out)
    metrics_report = _maybe_parse_json(trainer_text) or get_state(state_handle).metrics_report

    # ---- 3. Explainer agent ---------------------------------------------
    explainer_agent = ExplainerAgent(model=gemini_model, api_key=api_key)
    explainer_out = explainer_agent.run(
        metrics_report=metrics_report,
        dataset_summary=dataset_summary,
    )
    explanation_markdown = _crew_output_text(explainer_out)

    if output_path is not None:
        (output_path / "dataset_summary.json").write_text(
            json.dumps(dataset_summary, indent=2), encoding="utf-8"
        )
        (output_path / "metrics_report.json").write_text(
            json.dumps(metrics_report, indent=2), encoding="utf-8"
        )
        (output_path / "explanation.md").write_text(
            explanation_markdown, encoding="utf-8"
        )

    return PipelineRun(
        dataset_summary=dataset_summary,
        metrics_report=metrics_report,
        explanation_markdown=explanation_markdown,
        raw_outputs={
            "data": data_text,
            "trainer": trainer_text,
            "explainer": explanation_markdown,
        },
    )
