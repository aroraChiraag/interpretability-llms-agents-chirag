"""Pure-Python fairness math for the credit-default pipeline.

The Fairness Agent only *interprets* numbers — these functions produce them.

Definitions (binary classifier, label 1 = "predicted to default" = "denied"):

- **Selection rate** for a group = fraction of group members the model
  predicted as class 1.
- **Disparate Impact (DI)** across groups = ``min(selection_rate) /
  max(selection_rate)``. The 80% rule (4/5ths rule) flags DI < 0.80 as a
  potential adverse-impact concern.
- **Equalized Odds gap** = ``max(TPR) - min(TPR)`` across groups, plus the
  same gap on FPR. Smaller is fairer.
- **Predictive Equality** = the FPR gap on its own (focuses on the cost of
  *wrongly* flagging safe customers — the relevant fairness concept for a
  bank denying credit).

Age is bucketed into four ordinal bands so the AGE attribute behaves like a
categorical group rather than 60+ singleton ages.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


SEX_LABELS = {1: "male", 2: "female"}
MARRIAGE_LABELS = {1: "married", 2: "single", 3: "other"}
AGE_BANDS = [
    (0, 30, "<=30"),
    (31, 45, "31-45"),
    (46, 60, "46-60"),
    (61, 999, ">60"),
]


# ---------- helpers ---------------------------------------------------------


def _bucket_age(ages: Iterable[float]) -> List[str]:
    """Bucket ages into the four bands defined in :data:`AGE_BANDS`."""
    out: List[str] = []
    for a in ages:
        try:
            value = float(a)
        except (TypeError, ValueError):
            out.append("unknown")
            continue
        for low, high, label in AGE_BANDS:
            if low <= value <= high:
                out.append(label)
                break
        else:
            out.append("unknown")
    return out


def _label_group(attr: str, value: Any) -> str:
    """Translate raw codes into human-readable group labels where useful."""
    if attr == "SEX":
        return SEX_LABELS.get(int(value), f"sex_{int(value)}")
    if attr == "MARRIAGE":
        return MARRIAGE_LABELS.get(int(value), f"marriage_{int(value)}")
    return str(value)


# ---------- core math -------------------------------------------------------


def _per_group_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: List[str],
) -> Dict[str, Dict[str, Any]]:
    """For each group, return count / selection_rate / TPR / FPR / base rate."""
    group_array = np.asarray(group_labels)
    out: Dict[str, Dict[str, Any]] = {}
    for group in sorted(set(group_array)):
        mask = group_array == group
        y_t = y_true[mask]
        y_p = y_pred[mask]
        n = int(mask.sum())
        if n == 0:
            continue
        positives = y_t == 1
        negatives = y_t == 0
        selection = float(y_p.mean()) if n > 0 else 0.0
        tpr = float(y_p[positives].mean()) if positives.any() else None
        fpr = float(y_p[negatives].mean()) if negatives.any() else None
        out[group] = {
            "count": n,
            "selection_rate": round(selection, 4),
            "tpr": round(tpr, 4) if tpr is not None else None,
            "fpr": round(fpr, 4) if fpr is not None else None,
            "base_rate": round(float(y_t.mean()), 4) if n > 0 else 0.0,
        }
    return out


def _disparate_impact(group_stats: Dict[str, Dict[str, Any]]) -> Optional[float]:
    rates = [g["selection_rate"] for g in group_stats.values() if g["selection_rate"] > 0]
    if len(rates) < 2:
        return None
    return round(min(rates) / max(rates), 4)


def _gap(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    return round(max(clean) - min(clean), 4)


def fairness_for_attribute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_values: Iterable[Any],
    attribute: str,
) -> Dict[str, Any]:
    """Compute fairness statistics for one sensitive attribute."""
    if attribute == "AGE":
        groups = _bucket_age(sensitive_values)
    else:
        groups = [_label_group(attribute, v) for v in sensitive_values]

    stats = _per_group_stats(y_true, y_pred, groups)
    di = _disparate_impact(stats)
    tpr_gap = _gap([g["tpr"] for g in stats.values()])
    fpr_gap = _gap([g["fpr"] for g in stats.values()])

    return {
        "attribute": attribute,
        "groups": stats,
        "disparate_impact": di,
        "passes_80_pct_rule": (di is not None and di >= 0.80),
        "equalized_odds": {"tpr_gap": tpr_gap, "fpr_gap": fpr_gap},
        "predictive_equality_fpr_gap": fpr_gap,
    }


# ---------- public API ------------------------------------------------------


def compute_fairness_for_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_df: pd.DataFrame,
    attributes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute fairness metrics for one model across each sensitive attribute."""
    attributes = attributes or list(sensitive_df.columns)
    return {
        attr: fairness_for_attribute(
            y_true=np.asarray(y_true),
            y_pred=np.asarray(y_pred),
            sensitive_values=sensitive_df[attr].tolist(),
            attribute=attr,
        )
        for attr in attributes
        if attr in sensitive_df.columns
    }


def compute_fairness_for_all_models(
    metrics_report: Dict[str, Any],
    y_true: np.ndarray,
    sensitive_df: pd.DataFrame,
    attributes: Optional[List[str]] = None,
    tuning_notes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Iterate over every model in a metrics_report and compute fairness.

    Parameters
    ----------
    metrics_report : dict
        Output of :func:`agentic_default.ml_trainer.train_and_evaluate`. Each
        entry under ``models`` must include a ``predictions`` list aligned
        with ``y_true`` / ``sensitive_df``.
    y_true : np.ndarray
        The y_test labels.
    sensitive_df : pd.DataFrame
        Test-side sensitive columns (e.g. ``LoadedDataset.test_sensitive``).
    attributes : list of str, optional
        Subset of sensitive columns to evaluate. Defaults to all columns.

    Returns
    -------
    dict
        ``{"per_model": {model_name: per_attribute_stats}, "summary": {...}}``
        where ``summary`` highlights worst-DI and best-DI per attribute.
    """
    per_model: Dict[str, Any] = {}
    for m in metrics_report.get("models", []):
        preds = m.get("predictions") or []
        if not preds:
            continue
        per_model[m["model_name"]] = compute_fairness_for_model(
            y_true=y_true,
            y_pred=np.asarray(preds, dtype=int),
            sensitive_df=sensitive_df,
            attributes=attributes,
        )
        per_model[m["model_name"]]["_tuning_notes"] = (tuning_notes or {}).get(m["model_name"], "")

    summary = _summarise(per_model)
    return {"per_model": per_model, "summary": summary}


def _summarise(per_model: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a small "who-wins-what" headline from the per-model results."""
    if not per_model:
        return {}
    # Skip internal keys (e.g. _tuning_notes) — only keep real attribute dicts.
    attributes = sorted({
        attr
        for v in per_model.values()
        for attr in v
        if not attr.startswith("_") and isinstance(v.get(attr), dict)
    })
    summary: Dict[str, Any] = {}
    for attr in attributes:
        # disparate-impact ranking (higher = fairer; ignore Nones)
        di_scores = [
            (model, v[attr].get("disparate_impact"))
            for model, v in per_model.items()
            if attr in v and isinstance(v[attr], dict)
            and v[attr].get("disparate_impact") is not None
        ]
        di_scores.sort(key=lambda t: t[1], reverse=True)
        # FPR-gap ranking (lower = fairer)
        fpr_scores = [
            (model, v[attr]["equalized_odds"].get("fpr_gap"))
            for model, v in per_model.items()
            if attr in v and isinstance(v[attr], dict)
            and v[attr].get("equalized_odds") is not None
            and v[attr]["equalized_odds"].get("fpr_gap") is not None
        ]
        fpr_scores.sort(key=lambda t: t[1])
        summary[attr] = {
            "best_disparate_impact": di_scores[0] if di_scores else None,
            "worst_disparate_impact": di_scores[-1] if di_scores else None,
            "best_fpr_gap": fpr_scores[0] if fpr_scores else None,
            "worst_fpr_gap": fpr_scores[-1] if fpr_scores else None,
        }
    return summary
