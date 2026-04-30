"""DataAgent — loads the credit-card-default dataset and emits a JSON summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from crewai import Agent, Task

from ..tools.dataset_tools import LoadDatasetTool, PreviewRecordsTool
from ._crew_helpers import build_crew, kickoff_quiet
from .llm import build_gemini_llm


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "data_agent.txt"


class DataAgent:
    """A CrewAI agent that drives the data-loading tools."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        state_handle: str = "default",
    ) -> None:
        self.state_handle = state_handle
        self.model = model
        self.llm = build_gemini_llm(model=model, api_key=api_key)
        self._load_tool = LoadDatasetTool()
        self._preview_tool = PreviewRecordsTool()

    def _build_agent(self) -> Agent:
        return Agent(
            role="Credit Default Data Agent",
            goal=(
                "Load the Taiwan default-of-credit-card-clients CSV, split it "
                "into train/test, and emit a JSON dataset summary the Trainer "
                "Agent can use."
            ),
            backstory=(
                "You are a careful data engineer. You always inspect a small "
                "preview of the data before declaring it ready, and you "
                "communicate strictly via JSON so downstream agents can parse "
                "your output."
            ),
            llm=self.llm,
            tools=[self._load_tool, self._preview_tool],
            allow_delegation=False,
            verbose=False,
        )

    def run(
        self,
        csv_path: Optional[str] = None,
        snapshot_dir: Optional[str] = None,
    ) -> Any:
        """Kick off the data-loading crew and return the raw output."""
        prompt = PROMPT_PATH.read_text()
        ctx = (
            f"\n\nContext:\n  state_handle: {self.state_handle}\n"
            f"  csv_path: {csv_path or '(use bundled default)'}\n"
            f"  snapshot_dir: {snapshot_dir or '(no snapshot)'}\n"
        )
        agent = self._build_agent()
        task = Task(
            description=prompt + ctx,
            expected_output="A JSON object summarizing the loaded dataset.",
            agent=agent,
        )
        crew = build_crew(agents=[agent], tasks=[task], verbose=False)
        return kickoff_quiet(crew)
