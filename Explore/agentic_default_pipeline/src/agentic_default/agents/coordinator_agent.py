"""CoordinatorAgent — the chat-driven tuning agent.

One CrewAI agent with all the pipeline tools attached. Each user message
becomes a fresh `Task`, with the conversation history rendered into the
description so the agent has context. Returns the agent's plain-text reply.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from crewai import Agent, Task

from ..tools.dataset_tools import LoadDatasetTool
from ..tools.training_tools import (
    GetHyperparametersTool,
    GetMetricsTool,
    ResetHyperparametersTool,
    TrainModelsTool,
    UpdateHyperparametersTool,
)
from ._crew_helpers import build_crew, kickoff_quiet
from .llm import build_gemini_llm


PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "coordinator_agent.txt"
)


class CoordinatorAgent:
    """Conversational agent that drives the pipeline tools on demand."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        state_handle: str = "default",
    ) -> None:
        self.state_handle = state_handle
        self.model = model
        self.llm = build_gemini_llm(model=model, api_key=api_key, temperature=0.1)
        self._tools = [
            LoadDatasetTool(),
            GetHyperparametersTool(),
            UpdateHyperparametersTool(),
            ResetHyperparametersTool(),
            TrainModelsTool(),
            GetMetricsTool(),
        ]

    # ---- prompt assembly ------------------------------------------------

    def _system_prompt(self) -> str:
        return PROMPT_PATH.read_text()

    @staticmethod
    def _format_history(messages: List[dict]) -> str:
        """Render prior turns as a compact transcript."""
        lines = []
        for m in messages[-10:]:  # last 10 turns is plenty of context
            role = m.get("role", "user").upper()
            content = m.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    # ---- agent build ----------------------------------------------------

    def _build_agent(self) -> Agent:
        return Agent(
            role="Pipeline Coordinator",
            goal=(
                "Translate natural-language pipeline requests into the right "
                "sequence of tool calls and report back in plain language."
            ),
            backstory=(
                "You are an experienced ML engineer pairing with a "
                "non-technical analyst. You explain what you did, you never "
                "fabricate metrics, and you ask one short clarifying question "
                "rather than guessing when intent is unclear."
            ),
            llm=self.llm,
            tools=self._tools,
            allow_delegation=False,
            verbose=False,
        )

    # ---- public API -----------------------------------------------------

    def chat(self, user_message: str, history: Optional[List[dict]] = None) -> str:
        """Send one user message and return the agent's reply."""
        history = history or []
        transcript = self._format_history(history)

        prompt_blocks = [
            self._system_prompt(),
            "",
            f"Pipeline state handle: {self.state_handle}",
        ]
        if transcript:
            prompt_blocks.extend(["", "CONVERSATION SO FAR:", transcript])
        prompt_blocks.extend(["", "USER MESSAGE:", user_message])
        full_prompt = "\n".join(prompt_blocks)

        agent = self._build_agent()
        task = Task(
            description=full_prompt,
            expected_output="A short plain-English reply, optionally after using tools.",
            agent=agent,
        )
        crew = build_crew(agents=[agent], tasks=[task], verbose=False)
        result = kickoff_quiet(crew)
        return getattr(result, "raw", None) or str(result)
