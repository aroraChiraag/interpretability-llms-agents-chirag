"""BiasAgent — pre/post-training bias audit, reporting only.

This agent is purely advisory: its output is NEVER consumed by the
TrainerAgent, ExplainerAgent, FairnessAgent, or CoordinatorAgent. It writes
a Markdown audit, surfaces it in the Streamlit *Bias* tab, and stops there.

It accepts:

* ``dataset_signals`` — the dict produced by
  :func:`agentic_default.bias_metrics.compute_dataset_bias_signals`.
  These are dataset-level signals available immediately after the data
  has been loaded (no training required).
* ``metrics_report`` — optional. If models have been trained, the agent
  will also reason over per-model feature-importance to flag direct
  demographic features and proxy features in any model's top-10.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from crewai import Agent, Task

from ._crew_helpers import build_crew, kickoff_quiet
from .llm import build_gemini_llm


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "bias_agent.txt"


class BiasAgent:
    """A CrewAI agent that produces the Markdown bias audit."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.llm = build_gemini_llm(model=model, api_key=api_key, temperature=0.1)

    def _build_agent(self) -> Agent:
        return Agent(
            role="Lead Bias Detection Auditor",
            goal=(
                "Surface every plausible source of bias in the dataset and "
                "(if available) the trained models. Flag direct demographic "
                "features, proxy variables, intersectional risks, and "
                "propose counterfactual tests."
            ),
            backstory=(
                "You are a pre-deployment bias auditor for a regulated bank. "
                "You are not concerned with overall model performance — you "
                "are concerned with whether the model encodes discrimination. "
                "You write findings clearly, never soften them, and quote "
                "numbers verbatim from the data you receive."
            ),
            llm=self.llm,
            tools=[],
            allow_delegation=False,
            verbose=False,
        )

    def run(
        self,
        dataset_signals: Any,
        metrics_report: Optional[Any] = None,
        prescan: Optional[Any] = None,
    ) -> Any:
        """Generate a Markdown bias audit."""
        if not isinstance(dataset_signals, str):
            dataset_signals = json.dumps(dataset_signals, indent=2)
        if metrics_report is not None and not isinstance(metrics_report, str):
            # Strip predictions before sending to the LLM — too large.
            mr = dict(metrics_report) if isinstance(metrics_report, dict) else {}
            if isinstance(mr.get("models"), list):
                mr["models"] = [
                    {k: v for k, v in m.items() if k != "predictions"}
                    for m in mr["models"]
                ]
            metrics_report = json.dumps(mr, indent=2)
        if prescan is not None and not isinstance(prescan, str):
            prescan = json.dumps(prescan, indent=2)

        prompt = PROMPT_PATH.read_text()

        ctx_blocks = ["", "DATASET-LEVEL BIAS SIGNALS (JSON):", dataset_signals]
        if metrics_report:
            ctx_blocks.extend(
                ["", "TRAINER METRICS REPORT (JSON, may be empty):", metrics_report]
            )
        if prescan:
            ctx_blocks.extend(
                ["", "PRE-SCANNED FEATURE-IMPORTANCE FLAGS (JSON):", prescan]
            )

        agent = self._build_agent()
        task = Task(
            description=prompt + "\n\n" + "\n".join(ctx_blocks),
            expected_output="A Markdown bias audit. No JSON.",
            agent=agent,
        )
        crew = build_crew(agents=[agent], tasks=[task], verbose=False)
        return kickoff_quiet(crew)
