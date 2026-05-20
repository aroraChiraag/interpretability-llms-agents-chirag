"""Pre-training (and post-training) bias signals for the credit-default pipeline.

The :class:`BiasAgent` only **interprets** numbers — these functions produce
them. None of these signals affect the trainer / fairness / explainer; the
output is consumed by the BiasAgent and the Streamlit *Bias* tab only.

Two categories of signal:

**Dataset-level (always available after data load)**

- Per-demographic-group default rate (SEX, EDUCATION, MARRIAGE, AGE band).
- Undocumented EDUCATION values (codes 0, 5, 6) — count, default rate, gap.
- Average ``LIMIT_BAL`` per group (proxy-bias signal).
- Median ``BILL_AMT`` per group vs. the overall median (15% deviation flag).
- Subgroup of clients with any negative bill amount (returns / overpayments).

**Post-training (only available after at least one model has trained)**

- Direct demographic features in any model's top-10 importances.
- Proxy features (LIMIT_BAL, BILL_AMT*, PAY_AMT*) in any top-10.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


SENSITIVE_COLS = ["SEX", "AGE", "EDUCATION", "MARRIAGE"]
DOCUMENTED_EDUCATION = {1, 2, 3, 4}
BILL_AMT_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
PROXY_FEATURES = {"LIMIT_BAL", *BILL_AMT_COLS, *PAY_AMT_COLS}
DEMOGRAPHIC_FEATURES = {"SEX", "AGE", "EDUCATION", "MARRIAGE"}

AGE_BANDS = [
    (0, 30, "<=30"),
    (31, 45, "31-45"),
    (46, 60, "46-60"),
    (61, 70, "61-70"),
    (71, 80, "71-80"),
    (81, 999, ">80"),
]


# ---------- helpers ---------------------------------------------------------


def _bucket_age(values) -> List[str]:
    out: List[str] = []
    for v in values:
        try:
            value = float(v)
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


def _round(x: float, n: int = 4) -> float:
    return round(float(x), n)


def _safe_pct_gap(value: float, baseline: float) -> Optional[float]:
    if baseline is None or baseline == 0:
        return None
    return _round((value - baseline) / abs(baseline), 4)


# ---------- dataset-level scan ----------------------------------------------


def compute_dataset_bias_signals(
    records_json: List[Dict[str, Any]],
    target_column: str = "default payment next month",
) -> Dict[str, Any]:
    """Compute bias signals from the raw, unscaled CSV records."""
    df = pd.DataFrame(records_json)
    if target_column not in df.columns:
        raise KeyError(f"Target {target_column!r} missing from records.")

    overall_default = float(df[target_column].mean())

    # --- Per-group default rate ---------------------------------------------
    per_group_default: Dict[str, Any] = {}
    for col in SENSITIVE_COLS:
        if col not in df.columns:
            continue
        if col == "AGE":
            group_labels = _bucket_age(df[col].tolist())
        else:
            group_labels = df[col].astype(int).astype(str).tolist()
        ser = pd.Series(group_labels, name=col)
        agg = df.groupby(ser)[target_column].agg(["mean", "count"]).reset_index()
        per_group_default[col] = {
            str(row[col]): {
                "default_rate": _round(row["mean"]),
                "count": int(row["count"]),
                "gap_vs_overall_pp": _round((row["mean"] - overall_default) * 100, 2),
            }
            for _, row in agg.iterrows()
        }

    # --- EDUCATION: undocumented codes (0, 5, 6) ----------------------------
    education_undoc: Dict[str, Any] = {}
    if "EDUCATION" in df.columns:
        und_mask = ~df["EDUCATION"].isin(DOCUMENTED_EDUCATION)
        und = df[und_mask]
        documented = df[~und_mask]
        education_undoc = {
            "count": int(und_mask.sum()),
            "fraction": _round(float(und_mask.mean())),
            "default_rate_undocumented": _round(und[target_column].mean()) if len(und) else None,
            "default_rate_documented": _round(documented[target_column].mean()) if len(documented) else None,
            "values_seen": sorted({int(v) for v in df["EDUCATION"].unique()}),
        }

    # --- LIMIT_BAL by group (proxy bias) ------------------------------------
    limit_bal_by_group: Dict[str, Any] = {}
    if "LIMIT_BAL" in df.columns:
        overall_limit = float(df["LIMIT_BAL"].mean())
        limit_bal_by_group["overall_mean"] = _round(overall_limit, 2)
        for col in ["SEX", "EDUCATION", "MARRIAGE"]:
            if col not in df.columns:
                continue
            grouped = df.groupby(df[col].astype(int).astype(str))["LIMIT_BAL"].mean()
            limit_bal_by_group[col] = {
                str(g): {
                    "mean": _round(v, 2),
                    "pct_gap_vs_overall": _safe_pct_gap(v, overall_limit),
                    "below_20_pct_threshold": (v < 0.80 * overall_limit),
                }
                for g, v in grouped.items()
            }

    # --- BILL_AMT median skew ----------------------------------------------
    bill_amt_skew: Dict[str, Any] = {}
    bill_cols_present = [c for c in BILL_AMT_COLS if c in df.columns]
    if bill_cols_present:
        df_avg = df[bill_cols_present].mean(axis=1)
        overall_med = float(df_avg.median())
        bill_amt_skew["overall_median_avg_bill"] = _round(overall_med, 2)
        for col in ["SEX", "EDUCATION", "MARRIAGE"]:
            if col not in df.columns:
                continue
            medians = df_avg.groupby(df[col].astype(int).astype(str)).median()
            bill_amt_skew[col] = {
                str(g): {
                    "median": _round(v, 2),
                    "pct_gap_vs_overall": _safe_pct_gap(v, overall_med),
                    "exceeds_15_pct_flag": (
                        abs(_safe_pct_gap(v, overall_med) or 0) > 0.15
                    ),
                }
                for g, v in medians.items()
            }

    # --- Negative BILL_AMT subgroup ----------------------------------------
    negative_bill_amt: Dict[str, Any] = {}
    if bill_cols_present:
        any_neg = (df[bill_cols_present] < 0).any(axis=1)
        negative_bill_amt = {
            "count": int(any_neg.sum()),
            "fraction": _round(float(any_neg.mean())),
            "default_rate_negative_bill": _round(df[any_neg][target_column].mean()) if any_neg.any() else None,
            "default_rate_non_negative": _round(df[~any_neg][target_column].mean()) if (~any_neg).any() else None,
        }

    return {
        "overall": {
            "rows": int(len(df)),
            "default_rate": _round(overall_default),
        },
        "per_group_default_rate": per_group_default,
        "education_undocumented": education_undoc,
        "limit_bal_by_group": limit_bal_by_group,
        "bill_amt_skew": bill_amt_skew,
        "negative_bill_amt": negative_bill_amt,
    }


# ---------- post-training scan (feature-importance) -------------------------


def prescan_feature_importance(metrics_report: Dict[str, Any], source: str = "post_training") -> Dict[str, Any]:
    """Flag demographic + proxy features in each model's top-10."""
    out: Dict[str, Any] = {
        "demographic_flags": [],
        "proxy_flags": [],
        "models_scanned": [],
        "source": source,
    }
    for model in metrics_report.get("models", []):
        name = model.get("model_name", "unknown")
        out["models_scanned"].append(name)
        importance_records = model.get("feature_importance") or []
        # ml_trainer.py emits a list of {feature, importance, rank} dicts.
        for entry in importance_records[:10]:
            feat = entry.get("feature")
            rank = entry.get("rank")
            score = entry.get("importance")
            if feat in DEMOGRAPHIC_FEATURES:
                out["demographic_flags"].append({
                    "model": name,
                    "feature": feat,
                    "rank": rank,
                    "importance": score,
                })
            elif feat in PROXY_FEATURES:
                out["proxy_flags"].append({
                    "model": name,
                    "feature": feat,
                    "rank": rank,
                    "importance": score,
                })
    return out


# ---------- convenience flatteners for Streamlit ----------------------------


def flatten_per_group_default(per_group: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for attr, groups in per_group.items():
        for group, payload in groups.items():
            rows.append({
                "attribute": attr,
                "group": group,
                "default_rate": payload.get("default_rate"),
                "count": payload.get("count"),
                "gap_vs_overall_pp": payload.get("gap_vs_overall_pp"),
            })
    return rows


def flatten_limit_bal_by_group(limit_bal: Dict[str, Any]) -> List[Dict[str, Any]]:
    overall = limit_bal.get("overall_mean")
    rows: List[Dict[str, Any]] = []
    for attr, groups in limit_bal.items():
        if attr == "overall_mean" or not isinstance(groups, dict):
            continue
        for group, payload in groups.items():
            rows.append({
                "attribute": attr,
                "group": group,
                "mean_LIMIT_BAL": payload.get("mean"),
                "overall_mean": overall,
                "pct_gap_vs_overall": payload.get("pct_gap_vs_overall"),
                "below_20_pct_flag": payload.get("below_20_pct_threshold"),
            })
    return rows
