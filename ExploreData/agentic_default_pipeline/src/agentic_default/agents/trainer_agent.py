"""TrainerAgent — runs the ML training tool and emits a JSON metric report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from crewai import Agent, Task

from ..ml_trainer import SUPPORTED_MODELS
from ..tools.training_tools import GetMetricsTool, TrainModelsTool
from ._crew_helpers import build_crew, kickoff_quiet
from .llm import build_gemini_llm


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "trainer_agent.txt"


class TrainerAgent:
    """A CrewAI agent that orchestrates ``train_models`` and reports results."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        state_handle: str = "default",
    ) -> None:
        self.state_handle = state_handle
        self.model = model
        self.llm = build_gemini_llm(model=model, api_key=api_key)
        self._train_tool = TrainModelsTool()
        self._metrics_tool = GetMetricsTool()

    def _build_agent(self) -> Agent:
        return Agent(
            role="Credit Default Model Trainer",
            goal=(
                "Train Random Forest, XGBoost, and a small Neural Network on "
                "the loaded dataset and produce a JSON metric report."
            ),
            backstory=(
                "You are a senior ML engineer with a tabular-data background. "
                "You report metrics faithfully and never invent numbers; you "
                "only output what the training tool actually returned."
            ),
            llm=self.llm,
            tools=[self._train_tool, self._metrics_tool],
            allow_delegation=False,
            verbose=False,
        )

    def run(
        self,
        models: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> Any:
        prompt = PROMPT_PATH.read_text()
        chosen = models or list(SUPPORTED_MODELS)
        ctx = (
            f"\n\nContext:\n  state_handle: {self.state_handle}\n"
            f"  models_to_train: {chosen}\n"
            f"  output_dir: {output_dir or '(none)'}\n"
        )
        agent = self._build_agent()
        task = Task(
            description=prompt + ctx,
            expected_output="A JSON object with leaderboard, best_model, and per-model metrics.",
            agent=agent,
        )
        crew = build_crew(agents=[agent], tasks=[task], verbose=False)
        return kickoff_quiet(crew)
