"""
bias_glossary.py

Definitions, value-label maps, and bias-metric descriptions for the
BiasAgent tab. Based on the UCI Taiwan Credit Default dataset (Yeh, 2009).

Three public exports:
- FEATURE_GLOSSARY      — list of GlossaryEntry dicts for the 23 explanatory
                          variables (used for hover tooltips).
- BIAS_METRIC_GLOSSARY  — list of GlossaryEntry dicts for bias metric terms
                          shown in the Bias tab (e.g. Default Rate, Gap pp,
                          Proxy Feature, etc.).
- VALUE_LABELS          — dict of column → {code → human label} for decoding
                          raw integer codes in displayed tables.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class GlossaryEntry(TypedDict):
    term: str
    definition: str
    context: str
    requirement: Optional[str]


# ---------------------------------------------------------------------------
# Value-label maps for categorical columns
# ---------------------------------------------------------------------------

VALUE_LABELS: dict[str, dict[str, str]] = {
    "SEX": {
        "1": "Male",
        "2": "Female",
    },
    "EDUCATION": {
        "0": "Undocumented (0)",
        "1": "Graduate School",
        "2": "University",
        "3": "High School",
        "4": "Others",
        "5": "Undocumented (5)",
        "6": "Undocumented (6)",
    },
    "MARRIAGE": {
        "0": "Undocumented (0)",
        "1": "Married",
        "2": "Single",
        "3": "Others",
    },
    # Repayment status codes — shared by PAY_0 / PAY_2 … PAY_6
    "PAY_STATUS": {
        "-2": "No consumption",
        "-1": "Paid duly",
        "0":  "Revolving credit used",
        "1":  "Delay 1 month",
        "2":  "Delay 2 months",
        "3":  "Delay 3 months",
        "4":  "Delay 4 months",
        "5":  "Delay 5 months",
        "6":  "Delay 6 months",
        "7":  "Delay 7 months",
        "8":  "Delay 8 months",
        "9":  "Delay 9+ months",
    },
}


def decode_group_label(attribute: str, raw_code: str) -> str:
    """Return a human-readable label for a raw integer group code.

    Falls back to the raw code string if no mapping exists.
    """
    mapping = VALUE_LABELS.get(attribute, {})
    return mapping.get(str(raw_code), str(raw_code))


# ---------------------------------------------------------------------------
# Feature glossary — the 23 explanatory variables
# ---------------------------------------------------------------------------

FEATURE_GLOSSARY: list[GlossaryEntry] = [
    {
        "term": "LIMIT_BAL (X1)",
        "definition": (
            "Amount of the given credit in NT dollars. Includes both the individual "
            "consumer credit and supplementary family credit."
        ),
        "context": (
            "Used as a proxy-bias signal: groups with a mean LIMIT_BAL more than 20% "
            "below the overall mean may reflect historical under-lending to that demographic."
        ),
        "requirement": None,
    },
    {
        "term": "SEX (X2)",
        "definition": "Gender of the cardholder. Encoded as: 1 = Male, 2 = Female.",
        "context": (
            "A protected demographic attribute. Differences in default rates or credit "
            "limits across gender groups are flagged as potential discrimination signals."
        ),
        "requirement": None,
    },
    {
        "term": "EDUCATION (X3)",
        "definition": (
            "Highest education level. Encoded as: 1 = Graduate School, 2 = University, "
            "3 = High School, 4 = Others. Codes 0, 5, and 6 are undocumented in the "
            "original dataset specification."
        ),
        "context": (
            "Undocumented education codes (0, 5, 6) affect a meaningful fraction of the "
            "dataset and have different default rates from documented groups — a data quality "
            "and potential bias risk."
        ),
        "requirement": None,
    },
    {
        "term": "MARRIAGE (X4)",
        "definition": (
            "Marital status. Encoded as: 1 = Married, 2 = Single, 3 = Others. "
            "Code 0 is undocumented."
        ),
        "context": (
            "A protected demographic attribute. Disparities in default rates or credit "
            "limits across marital status groups are surfaced as bias signals."
        ),
        "requirement": None,
    },
    {
        "term": "AGE (X5)",
        "definition": "Age of the cardholder in years.",
        "context": (
            "Bucketed into bands (≤30, 31-45, 46-60, 61-70, 71-80, >80) for group analysis. "
            "A protected attribute in lending under age-discrimination regulations."
        ),
        "requirement": None,
    },
    {
        "term": "PAY_0 / PAY_2–PAY_6 (X6–X11)",
        "definition": (
            "History of past monthly repayment status from April to September 2005. "
            "PAY_0 = September 2005; PAY_2 = August 2005; … PAY_6 = April 2005. "
            "Scale: -1 = paid duly; 1 = 1-month delay; 2 = 2-month delay; … "
            "9 = 9+ month delay."
        ),
        "context": (
            "These ordinal repayment-history features are among the strongest predictors "
            "of default. If they correlate with protected attributes (SEX, AGE, MARRIAGE), "
            "they can act as indirect proxies for discrimination."
        ),
        "requirement": None,
    },
    {
        "term": "BILL_AMT1–6 (X12–X17)",
        "definition": (
            "Amount of bill statement in NT dollars. BILL_AMT1 = September 2005; "
            "BILL_AMT2 = August 2005; … BILL_AMT6 = April 2005. Negative values "
            "indicate overpayments or credit returns."
        ),
        "context": (
            "Median BILL_AMT is compared across demographic groups. A 15%+ deviation "
            "from the overall median is flagged as a bill-skew bias signal, suggesting "
            "one group systematically carries higher or lower balances."
        ),
        "requirement": None,
    },
    {
        "term": "PAY_AMT1–6 (X18–X23)",
        "definition": (
            "Amount of previous payment in NT dollars. PAY_AMT1 = September 2005; "
            "PAY_AMT2 = August 2005; … PAY_AMT6 = April 2005."
        ),
        "context": (
            "Payment amounts are proxy features — they correlate with income and wealth, "
            "which in turn may correlate with demographic attributes. Their presence in "
            "a model's top-10 features is flagged as a proxy-bias risk."
        ),
        "requirement": None,
    },
]


# ---------------------------------------------------------------------------
# Bias metric glossary — terms used in the Bias tab itself
# ---------------------------------------------------------------------------

BIAS_METRIC_GLOSSARY: list[GlossaryEntry] = [
    {
        "term": "Default Rate",
        "definition": (
            "The proportion of clients in a group who actually defaulted on their "
            "next payment (ground-truth label = 1)."
        ),
        "context": (
            "The overall dataset default rate is the baseline. Per-group default rates "
            "show the inherent risk profile before any model is applied. Differences "
            "across groups reflect real-world socioeconomic disparities in the raw data."
        ),
        "requirement": None,
    },
    {
        "term": "Gap vs Overall (pp)",
        "definition": (
            "The difference between a group's default rate and the overall dataset "
            "default rate, expressed in percentage points (pp)."
        ),
        "context": (
            "A positive gap means the group defaults more often than average; a negative "
            "gap means less often. Large gaps signal that a model trained on this data "
            "may inherit structural biases from the real-world lending history."
        ),
        "requirement": None,
    },
    {
        "term": "LIMIT_BAL Proxy Bias",
        "definition": (
            "A signal that average credit limits differ significantly across demographic "
            "groups — suggesting that historical lending decisions may have systematically "
            "under- or over-served certain populations."
        ),
        "context": (
            "If a group's mean LIMIT_BAL is more than 20% below the overall mean, it is "
            "flagged. Because LIMIT_BAL is also a model feature, this historical gap can "
            "be encoded directly into model predictions."
        ),
        "requirement": None,
    },
    {
        "term": "BILL_AMT Skew",
        "definition": (
            "A signal that the median average bill statement amount differs by more than "
            "15% from the overall median for a particular demographic group."
        ),
        "context": (
            "Persistent over- or under-billing across groups may reflect spending power "
            "differences linked to demographics, creating a proxy path through which "
            "protected attributes influence model outcomes."
        ),
        "requirement": None,
    },
    {
        "term": "Undocumented Education Codes",
        "definition": (
            "Rows where the EDUCATION field contains values (0, 5, or 6) that are not "
            "defined in the original dataset specification."
        ),
        "context": (
            "These rows cannot be reliably assigned to an education category. If their "
            "default rate differs from documented groups, including or excluding them "
            "can materially change model behaviour for an uncharacterised subpopulation."
        ),
        "requirement": None,
    },
    {
        "term": "Negative BILL_AMT Subgroup",
        "definition": (
            "Clients who have at least one negative bill statement amount — indicating "
            "a credit return, overpayment, or refund in that billing period."
        ),
        "context": (
            "This subgroup may behave differently from the general population. If their "
            "default rate diverges materially, it signals a data pattern the model "
            "could exploit in ways that are hard to audit."
        ),
        "requirement": None,
    },
    {
        "term": "Direct Demographic Feature",
        "definition": (
            "A protected-attribute column (SEX, AGE, EDUCATION, or MARRIAGE) that appears "
            "in a model's top-10 most important features."
        ),
        "context": (
            "Under fair-lending regulations, using protected attributes directly as "
            "predictive features is high-risk and requires explicit justification. "
            "Its appearance in the top-10 is flagged as an immediate audit concern."
        ),
        "requirement": (
            "Regulators and auditors typically require explicit justification if any "
            "protected attribute is used as a direct model input."
        ),
    },
    {
        "term": "Proxy Feature",
        "definition": (
            "A non-demographic feature (LIMIT_BAL, BILL_AMT*, PAY_AMT*) that "
            "correlates with a protected attribute and appears in a model's top-10 "
            "most important features."
        ),
        "context": (
            "Even when protected attributes are excluded from training, proxy features "
            "can transmit the same discriminatory signal indirectly. Their high importance "
            "scores may indicate the model is de-facto learning group membership."
        ),
        "requirement": None,
    },
    {
        "term": "Intersectional Bias",
        "definition": (
            "Bias that emerges at the intersection of two or more protected attributes "
            "(e.g., young + female + single) rather than from any single attribute alone."
        ),
        "context": (
            "Single-attribute fairness checks can miss subgroups that are uniquely "
            "disadvantaged only in combination. Intersectional analysis is a recommended "
            "next step when single-attribute gaps appear small."
        ),
        "requirement": None,
    },
]


# Column-level help text for bias summary dataframes
BIAS_COLUMN_HELP: dict[str, str] = {
    "attribute":         "Protected demographic attribute (SEX, AGE, EDUCATION, MARRIAGE).",
    "group":             "Specific group within the attribute (e.g. Male, Female, ≤30).",
    "default_rate":      "Proportion of clients in this group who actually defaulted (ground truth).",
    "count":             "Number of clients in this group in the dataset.",
    "gap_vs_overall_pp": (
        "Difference between this group's default rate and the overall default rate, "
        "in percentage points. Positive = higher-risk group; negative = lower-risk."
    ),
    "mean_LIMIT_BAL":    "Average credit limit (NT$) for this group.",
    "overall_mean":      "Average credit limit (NT$) across all clients.",
    "pct_gap_vs_overall":"Percentage difference between this group's mean and the overall mean.",
    "below_20_pct_flag": "True if this group's mean LIMIT_BAL is >20% below the overall mean — a proxy-bias flag.",
    "median_avg_bill":   "Median of the client's average monthly bill amount (NT$) for this group.",
    "overall_median":    "Median of the average monthly bill amount across all clients.",
    "exceeds_15_pct_flag":"True if this group's median deviates >15% from the overall median — a bill-skew flag.",
    "model":             "Model architecture (Random Forest, XGBoost, Neural Network).",
    "feature":           "Feature name flagged in the model's top-10 feature importances.",
    "rank":              "Rank of this feature in the model's importance list (1 = most important).",
    "importance":        "Importance score assigned to this feature by the model.",
}


# ---------------------------------------------------------------------------
# Full 23-variable feature table (per UCI Yeh, 2009 specification)
# ---------------------------------------------------------------------------
# One row per variable, with the X-numbering used in the source paper.
# Rendered as a table on the Run-tab "Load Data" stage so users can see
# every column up front.

FEATURE_TABLE_23: list[dict[str, str]] = [
    {
        "X#": "X1",
        "column": "LIMIT_BAL",
        "type": "numeric (NT$)",
        "description": (
            "Amount of the given credit. Includes both the individual "
            "consumer credit and the family (supplementary) credit."
        ),
    },
    {
        "X#": "X2",
        "column": "SEX",
        "type": "categorical",
        "description": "Gender. 1 = male, 2 = female.",
    },
    {
        "X#": "X3",
        "column": "EDUCATION",
        "type": "categorical",
        "description": (
            "1 = graduate school, 2 = university, 3 = high school, "
            "4 = others. (Codes 0/5/6 also appear in the raw data and are "
            "undocumented in the source paper.)"
        ),
    },
    {
        "X#": "X4",
        "column": "MARRIAGE",
        "type": "categorical",
        "description": "Marital status. 1 = married, 2 = single, 3 = others.",
    },
    {
        "X#": "X5",
        "column": "AGE",
        "type": "numeric (years)",
        "description": "Age of the cardholder in years.",
    },
    {
        "X#": "X6",
        "column": "PAY_0",
        "type": "ordinal",
        "description": (
            "Repayment status in **September 2005**. Scale: -1 = paid duly; "
            "1 = 1-month delay; 2 = 2-month delay; … ; 9 = 9+ month delay."
        ),
    },
    {
        "X#": "X7",
        "column": "PAY_2",
        "type": "ordinal",
        "description": "Repayment status in **August 2005** (same scale as PAY_0).",
    },
    {
        "X#": "X8",
        "column": "PAY_3",
        "type": "ordinal",
        "description": "Repayment status in **July 2005** (same scale as PAY_0).",
    },
    {
        "X#": "X9",
        "column": "PAY_4",
        "type": "ordinal",
        "description": "Repayment status in **June 2005** (same scale as PAY_0).",
    },
    {
        "X#": "X10",
        "column": "PAY_5",
        "type": "ordinal",
        "description": "Repayment status in **May 2005** (same scale as PAY_0).",
    },
    {
        "X#": "X11",
        "column": "PAY_6",
        "type": "ordinal",
        "description": "Repayment status in **April 2005** (same scale as PAY_0).",
    },
    {
        "X#": "X12",
        "column": "BILL_AMT1",
        "type": "numeric (NT$)",
        "description": "Amount of bill statement in **September 2005**.",
    },
    {
        "X#": "X13",
        "column": "BILL_AMT2",
        "type": "numeric (NT$)",
        "description": "Amount of bill statement in **August 2005**.",
    },
    {
        "X#": "X14",
        "column": "BILL_AMT3",
        "type": "numeric (NT$)",
        "description": "Amount of bill statement in **July 2005**.",
    },
    {
        "X#": "X15",
        "column": "BILL_AMT4",
        "type": "numeric (NT$)",
        "description": "Amount of bill statement in **June 2005**.",
    },
    {
        "X#": "X16",
        "column": "BILL_AMT5",
        "type": "numeric (NT$)",
        "description": "Amount of bill statement in **May 2005**.",
    },
    {
        "X#": "X17",
        "column": "BILL_AMT6",
        "type": "numeric (NT$)",
        "description": "Amount of bill statement in **April 2005**.",
    },
    {
        "X#": "X18",
        "column": "PAY_AMT1",
        "type": "numeric (NT$)",
        "description": "Amount paid in **September 2005**.",
    },
    {
        "X#": "X19",
        "column": "PAY_AMT2",
        "type": "numeric (NT$)",
        "description": "Amount paid in **August 2005**.",
    },
    {
        "X#": "X20",
        "column": "PAY_AMT3",
        "type": "numeric (NT$)",
        "description": "Amount paid in **July 2005**.",
    },
    {
        "X#": "X21",
        "column": "PAY_AMT4",
        "type": "numeric (NT$)",
        "description": "Amount paid in **June 2005**.",
    },
    {
        "X#": "X22",
        "column": "PAY_AMT5",
        "type": "numeric (NT$)",
        "description": "Amount paid in **May 2005**.",
    },
    {
        "X#": "X23",
        "column": "PAY_AMT6",
        "type": "numeric (NT$)",
        "description": "Amount paid in **April 2005**.",
    },
    {
        "X#": "Y",
        "column": "default payment next month",
        "type": "binary target",
        "description": (
            "Response variable. 1 = the customer defaulted on their next "
            "payment; 0 = the customer did not default."
        ),
    },
]
