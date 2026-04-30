"""Runtime defaults applied before CrewAI is imported.

We keep CrewAI's tracing **on** — the user wants execution traces collected.
What we suppress is only the *interactive y/N prompt* CrewAI emits at the
end of a run asking whether to open the trace viewer. That prompt hangs in
non-interactive contexts (notebooks, CI, log pipelines).

Two complementary mechanisms:

1. Env-var hints CrewAI may honour to skip the prompt (set as defaults so
   the user can always override).
2. ``auto_decline_prompts`` — a context manager that monkey-patches
   ``builtins.input`` to return ``"n"`` whenever code inside the context
   calls it. Wrap each ``crew.kickoff()`` call with this and the prompt is
   silently declined while tracing itself stays enabled.
"""

from __future__ import annotations

import builtins
import contextlib
import os
import sys
from typing import Iterator


# Map of env var -> default value. We *only* set prompt-suppression hints —
# we do **not** disable tracing or telemetry.
_PROMPT_SUPPRESSION_DEFAULTS = {
    "CREWAI_NON_INTERACTIVE": "true",
    "CREWAI_DISABLE_TRACE_PROMPT": "true",
}


def apply_non_interactive_defaults() -> None:
    """Set prompt-suppression env vars to safe defaults if unset.

    Tracing and telemetry are intentionally left at CrewAI's defaults.
    """
    for key, default in _PROMPT_SUPPRESSION_DEFAULTS.items():
        os.environ.setdefault(key, default)


@contextlib.contextmanager
def auto_decline_prompts() -> Iterator[None]:
    """Auto-decline any ``input(...)`` call made inside the context.

    Replaces ``builtins.input`` with a stub that returns ``"n"``. The
    original prompt is echoed to ``stderr`` with an ``[auto-declined]``
    prefix so the user can see what was suppressed. Tracing itself is not
    affected — the trace data is still collected and (if configured)
    uploaded; only the interactive viewer prompt is short-circuited.
    """
    original_input = builtins.input

    def _silent_input(prompt: str = "") -> str:
        if prompt:
            print(f"[auto-declined] {prompt.rstrip()} -> n", file=sys.stderr)
        return "n"

    builtins.input = _silent_input  # type: ignore[assignment]
    try:
        yield
    finally:
        builtins.input = original_input


# Apply prompt-suppression env defaults on import.
apply_non_interactive_defaults()
