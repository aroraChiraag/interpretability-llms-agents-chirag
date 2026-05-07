"""
fairness_glossary.py

Definitions for fairness and bias metrics used in the FairnessAgent tab.
Each entry contains a short label, a formal definition, and an in-context
explanation specific to the credit-default use case.
"""

from __future__ import annotations

from typing import TypedDict, Optional


class GlossaryEntry(TypedDict):
    term: str
    definition: str
    context: str
    requirement: Optional[str]


FAIRNESS_GLOSSARY: list[GlossaryEntry] = [
    {
        "term": "Selection Rate",
        "definition": (
            "The proportion of the population within a specific group that is "
            "classified as the 'positive' class by the model."
        ),
        "context": (
            "For the credit default use case, if the model 'positive' prediction "
            "is 'Grant Credit' (or conversely, 'Predict Default'), the Selection Rate "
            "is the percentage of a group (e.g., Women) that the model predicts will default."
        ),
        "requirement": None,
    },
    {
        "term": "Base Rate",
        "definition": (
            "The actual prevalence of the target condition within a specific group "
            "in the ground-truth data."
        ),
        "context": (
            "This is the percentage of a group that actually defaults on their payments. "
            "It represents the inherent risk profile of the raw data before any model is applied."
        ),
        "requirement": None,
    },
    {
        "term": "Disparate Impact (DI)",
        "definition": (
            "A measure of the relative Selection Rates between a protected group "
            "(unprivileged) and a reference group (privileged)."
        ),
        "context": (
            "It quantifies whether a model flags one group for default significantly "
            "more often than another."
        ),
        "requirement": None,
    },
    {
        "term": "Passes 80% Rule (Four-Fifths Rule)",
        "definition": (
            "A regulatory threshold derived from US EEOC guidelines stating that the "
            "Selection Rate for any group should be at least 80% of the rate for the "
            "group with the highest rate."
        ),
        "context": (
            "If the Disparate Impact ratio is less than 0.80, the model is statistically "
            "considered to have a 'disparate impact' and may be flagged as discriminatory "
            "by auditors."
        ),
        "requirement": None,
    },
    {
        "term": "TPR (True Positive Rate) / Recall",
        "definition": (
            "The proportion of actual positives that are correctly identified by the model."
        ),
        "context": (
            "The percentage of actual defaulters that the model correctly caught."
        ),
        "requirement": None,
    },
    {
        "term": "FPR (False Positive Rate)",
        "definition": (
            "The proportion of actual negatives that are incorrectly identified as positives."
        ),
        "context": (
            "The percentage of customers who would have paid on time but were incorrectly "
            "flagged as 'Defaulters.' This represents the 'insult rate' to good customers."
        ),
        "requirement": None,
    },
    {
        "term": "TPR Gap (Equal Opportunity Difference)",
        "definition": (
            "The absolute difference between the True Positive Rates of two different groups."
        ),
        "context": (
            "It measures if the model is better at catching defaulters in one group than "
            "another. A large gap suggests the model is 'under-detecting' risk in one "
            "demographic while 'over-detecting' it in another."
        ),
        "requirement": (
            "For Equal Opportunity Fairness, this gap should ideally be zero."
        ),
    },
    {
        "term": "FPR Gap (Predictive Equality Difference)",
        "definition": (
            "The absolute difference between the False Positive Rates of two different groups."
        ),
        "context": (
            "It measures if one group is being unfairly 'wrongly accused' of default more "
            "often than another. In banking, a higher FPR for a protected group (e.g., "
            "younger applicants) means they are being denied credit they actually deserve "
            "more often than older applicants."
        ),
        "requirement": None,
    },
]

# Keyed lookup for quick access by term name
GLOSSARY_BY_TERM: dict[str, GlossaryEntry] = {
    entry["term"]: entry for entry in FAIRNESS_GLOSSARY
}

# Short tooltip strings keyed by the column names used in the fairness summary DataFrame
COLUMN_HELP: dict[str, str] = {
    "model": "The model architecture being evaluated (Random Forest, XGBoost, Neural Network).",
    "attribute": "The protected demographic attribute (SEX, AGE band, MARRIAGE status).",
    "disparate_impact": (
        "Ratio of Selection Rates: unprivileged group ÷ privileged group. "
        "Values ≥ 0.80 satisfy the Four-Fifths Rule."
    ),
    "passes_80_pct_rule": (
        "True if Disparate Impact ≥ 0.80 (US EEOC Four-Fifths Rule). "
        "False indicates a statistically discriminatory outcome."
    ),
    "tpr_gap": (
        "Absolute difference in True Positive Rates between groups (Equal Opportunity). "
        "Closer to 0 is fairer."
    ),
    "fpr_gap": (
        "Absolute difference in False Positive Rates between groups (Predictive Equality). "
        "Closer to 0 means fewer wrongful credit denials across groups."
    ),
}
