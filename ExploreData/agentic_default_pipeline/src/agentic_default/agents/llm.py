"""Helpers for constructing the Gemini LLM used by every agent."""

from __future__ import annotations

import os
from typing import Optional

from crewai import LLM


DEFAULT_MODEL = "gemini-2.5-flash"


def build_gemini_llm(
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> LLM:
    """Build a CrewAI ``LLM`` configured for Gemini.

    Parameters
    ----------
    model : str
        Gemini model name (e.g., ``"gemini-2.5-flash"``).
    api_key : str, optional
        Gemini API key. Falls back to the ``GEMINI_API_KEY`` env var.
    temperature : float, default 0.0
        Sampling temperature. We default to deterministic output.

    Returns
    -------
    LLM
        A CrewAI LLM instance pointed at Gemini.

    Raises
    ------
    RuntimeError
        If no API key was provided and ``GEMINI_API_KEY`` is unset.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env or pass api_key=..."
        )
    return LLM(model=f"gemini/{model}", api_key=key, temperature=temperature)
