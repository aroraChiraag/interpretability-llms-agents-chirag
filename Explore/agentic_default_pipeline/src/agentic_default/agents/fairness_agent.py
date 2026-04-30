"""
fairness_agent.py

This module defines the Fairness Agent. It evaluates the operational fairness 
of three models (Neural Network, Random Forest, XGBoost) by analyzing 
disparities in their output predictions across protected demographic groups.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional
from crewai import Agent, Task

# Importing helper functions as per explainer_agent.py structure

from ._crew_helpers import build_crew, kickoff_quiet
from .llm import build_gemini_llm

# Path to the prompt text file
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "fairness_agent.txt"

class FairnessAgent:
    """A CrewAI agent that evaluates model-level fairness across multiple architectures."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.llm = build_gemini_llm(model=model, api_key=api_key, temperature=0.1)

    def _build_agent(self) -> Agent:
        return Agent(
            role="Lead Model Fairness Auditor",
            goal=(
                "Evaluate and compare the operational fairness of the Neural Network, "
                "Random Forest, and XGBoost models. Identify which model architecture "
                "minimizes algorithmic bias while maintaining credit risk predictive power."
            ),
            backstory=(
                "You are an expert in Algorithmic Accountability. You do not look at raw "
                "data bias; instead, you focus on 'Outcome Fairness.' You analyze how different "
                "model architectures—specifically Neural Networks, Random Forests, and Gradient "
                "Boosted Trees (XGBoost)—behave differently toward protected groups. "
                "You are tasked with ensuring the bank does not deploy a 'Black Box' model "
                "that inadvertently discriminates against 'SEX', 'AGE', or 'MARRIAGE' status."
            ),
            llm=self.llm,
            tools=[], # Focused on analytical reasoning over the trainer's metrics
            allow_delegation=False,
            verbose=False,
        )

    def run(self, metrics_report: Any, fairness_metrics: Optional[Any] = None) -> Any:
        """
        Generate a Markdown Fairness Audit comparing the three models.
        
        Args:
            metrics_report: The standard performance metrics (Recall, Precision, etc.)
            fairness_metrics: Specific fairness calculations (Disparate Impact, etc.)
        """
        if not isinstance(metrics_report, str):
            metrics_report = json.dumps(metrics_report, indent=2)
        if fairness_metrics is not None and not isinstance(fairness_metrics, str):
            fairness_metrics = json.dumps(fairness_metrics, indent=2)

        prompt = PROMPT_PATH.read_text()
        
        # Constructing context for the agent to compare the three models
        ctx_blocks = [
            "", 
            "PERFORMANCE METRICS (NN, RF, XGBOOST):", metrics_report
        ]
        if fairness_metrics:
            ctx_blocks.extend(["", "MODEL-LEVEL FAIRNESS SCORES:", fairness_metrics])

        agent = self._build_agent()
        task = Task(
            description=prompt + "\n\n" + "\n".join(ctx_blocks),
            expected_output="A Markdown Fairness Audit and Model Recommendation.",
            agent=agent,
        )
        crew = build_crew(agents=[agent], tasks=[task], verbose=False)
        return kickoff_quiet(crew)