"""Tiny helpers for constructing and kicking off Crew objects.

Tracing is left at CrewAI's default (enabled). The ``kickoff_quiet`` helper
wraps ``crew.kickoff()`` in :func:`auto_decline_prompts` so the trace-viewer
y/N prompt that CrewAI emits at the end of a run is auto-declined in
non-interactive contexts (Jupyter, CI). The trace data itself is still
collected and (if configured) uploaded.
"""

from __future__ import annotations

from typing import Any

from crewai import Crew

from .._runtime import auto_decline_prompts


def build_crew(*, agents: list, tasks: list, verbose: bool = False) -> Crew:
    """Construct a Crew with default tracing behavior."""
    return Crew(agents=agents, tasks=tasks, verbose=verbose)


def kickoff_quiet(crew: Crew) -> Any:
    """Run ``crew.kickoff()`` while auto-declining any interactive prompts."""
    with auto_decline_prompts():
        return crew.kickoff()
