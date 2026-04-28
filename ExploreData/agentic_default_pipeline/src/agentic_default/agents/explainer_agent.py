"""ExplainerAgent — turns the metric JSON into a Markdown interpretability brief."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from crewai import Agent, Task

from ._crew_helpers import build_crew, kickoff_quiet
from .llm import build_gemini_llm


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "explainer_agent.txt"


class ExplainerAgent:
    """A CrewAI agent that produces the human-readable explanation."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.llm = build_gemini_llm(model=model, api_key=api_key, temperature=0.2)

    def _build_agent(self) -> Agent:
        return Agent(
            role="Credit Default Explainability Reporter",
            goal=(
                "Translate the trainer's metric report into a clear Markdown "
                "interpretability brief covering metrics, confusion-matrix "
                "errors, feature importances, and caveats."
            ),
            backstory=(
                "You are an applied ML communicator. You write plain prose "
                "that respects the reader's time, you never fabricate numbers, "
                "and you flag ethical caveats (class imbalance, fairness, "
                "calibration) when they are relevant."
            ),
            llm=self.llm,
            tools=[],
            allow_delegation=False,
            verbose=False,
        )

    def run(self, metrics_report: Any, dataset_summary: Optional[Any] = None) -> Any:
        """Generate the Markdown explanation from the trainer's report."""
        if not isinstance(metrics_report, str):
            metrics_report = json.dumps(metrics_report, indent=2)
        if dataset_summary is not None and not isinstance(dataset_summary, str):
            dataset_summary = json.dumps(dataset_summary, indent=2)

        prompt = PROMPT_PATH.read_text()
        ctx_blocks = ["", "TRAINER REPORT (JSON):", metrics_report]
        if dataset_summary:
            ctx_blocks.extend(["", "DATA AGENT SUMMARY (JSON):", dataset_summary])

        agent = self._build_agent()
        task = Task(
            description=prompt + "\n\n" + "\n".join(ctx_blocks),
            expected_output="A Markdown interpretability brief.",
            agent=agent,
        )
        crew = build_crew(agents=[agent], tasks=[task], verbose=False)
        return kickoff_quiet(crew)
