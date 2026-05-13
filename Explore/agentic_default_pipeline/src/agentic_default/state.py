"""Shared in-memory pipeline state.

Numpy arrays and per-run results live in process-local storage so they don't
have to be serialised through LLM prompts. The Streamlit app and the chat
coordinator both read/write this state.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .data_loader import LoadedDataset
from .ml_trainer import default_hyperparameters


@dataclass
class PipelineState:
    """Holds the cross-agent artifacts for a single pipeline run."""

    dataset: Optional[LoadedDataset] = None
    metrics_report: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    hyperparameters: Dict[str, Dict[str, Any]] = field(default_factory=default_hyperparameters)
    fairness_metrics: Dict[str, Any] = field(default_factory=dict)
    fairness_explanation: str = ""
    bias_signals: Dict[str, Any] = field(default_factory=dict)
    bias_explanation: str = ""


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
