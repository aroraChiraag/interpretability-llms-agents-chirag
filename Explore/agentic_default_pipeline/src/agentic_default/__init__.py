"""Agentic credit-card default classification pipeline.

Public entry point: :func:`agentic_default.pipeline.run_pipeline`.
"""

# Apply non-interactive defaults BEFORE any submodule imports CrewAI.
from . import _runtime  # noqa: F401  (side effect: env vars)


__version__ = "0.1.0"
