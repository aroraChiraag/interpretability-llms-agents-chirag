"""Shared in-memory pipeline state.

Because CrewAI tools have to take JSON-friendly inputs/outputs but our ML
arrays are large numpy objects, we keep arrays in a process-local store and
only pass small string handles between agents. This avoids serializing tens
of thousands of rows into the LLM prompt.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .data_loader import LoadedDataset


@dataclass
class PipelineState:
    """Holds the cross-agent artifacts for a single pipeline run."""

    dataset: Optional[LoadedDataset] = None
    metrics_report: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


_STATE_LOCK = threading.Lock()
_STATE: Dict[str, PipelineState] = {}


def get_state(handle: str = "default") -> PipelineState:
    """Return the (lazily created) state object for the given handle."""
    with _STATE_LOCK:
        if handle not in _STATE:
            _STATE[handle] = PipelineState()
        return _STATE[handle]


def reset_state(handle: str = "default") -> None:
    """Clear state for the given handle. Useful between runs."""
    with _STATE_LOCK:
        _STATE.pop(handle, None)
