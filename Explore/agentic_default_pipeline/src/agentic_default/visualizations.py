"""
visualizations.py

Ten Plotly figures for the Fairness & Bias audit dashboard.
All functions return go.Figure objects — no Streamlit calls here.
The app.py Visualizations tab handles st.plotly_chart() rendering.

Priority order (highest impact first):
 1. plot_fpr_comparison          — FPR per group per model
 2. plot_subgroup_radar          — multi-metric polar chart per group
 3. plot_confusion_matrix_grid   — annotated CM per model
 4. plot_feature_importance_bias — colour-coded importance bars
 5. plot_agent_agreement_matrix  — pass/fail criteria matrix
 6. plot_default_rate_heatmap    — dataset default rates by demographic
 7. plot_equalized_odds          — TPR-gap vs FPR-gap scatter
 8. plot_threshold_impact        — FPR/TPR curves at varying thresholds
 9. plot_proxy_bias_network      — Sankey of proxy-feature → group gaps
10. plot_precision_recall_frontier — PR curves per model
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .bias_glossary import decode_group_label


def _validate_fairness_per_model(per_model: dict, context: str = "") -> None:
    """Warn if any per-model value is not a dict (e.g. _tuning_notes leaked in)."""
    import warnings as _w
    for model_name, attrs in per_model.items():
        for k, v in attrs.items():
            if not k.startswith("_") and not isinstance(v, dict):
                _w.warn(f"fairness_metrics[per_model][{model_name!r}][{k!r}] = {type(v).__name__!r}, expected dict. Context: {context}")


# ── Palette ───────────────────────────────────────────────────────────────────

MODEL_COLORS: Dict[str, str] = {
    "random_forest":  "#1565C0",
    "xgboost":        "#2E7D32",
    "neural_network": "#F57C00",
}
_DEFAULT_MODEL_COLOR = "#546E7A"

ATTR_COLORS: Dict[str, str] = {
    "SEX":       "#7B1FA2",
    "AGE":       "#00838F",
    "MARRIAGE":  "#E65100",
    "EDUCATION": "#33691E",
}

DEMOGRAPHIC_FEATURES = {"SEX", "AGE", "EDUCATION", "MARRIAGE"}
PROXY_FEATURES = {
    "LIMIT_BAL",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",
    "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6",
}

PASS_COLOR = "#43A047"
FAIL_COLOR = "#E53935"
WARN_COLOR = "#FB8C00"
NA_COLOR   = "#90A4AE"

_PLOT_BG    = "#0f172a"
_PAPER_BG   = "#0f172a"
_GRID_COLOR = "#1e293b"
_TEXT_COLOR = "#e2e8f0"
_FONT       = dict(family="Inter, sans-serif", color=_TEXT_COLOR, size=12)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _model_color(name: str) -> str:
    return MODEL_COLORS.get(name, _DEFAULT_MODEL_COLOR)


def _base_layout(**kwargs) -> dict:
    """Shared dark-theme layout defaults.

    Uses dict-literal unpacking so caller kwargs silently override defaults
    (dict(key=val, **kwargs) raises TypeError on duplicate keys in Python 3).
    """
    return {
        "paper_bgcolor": _PAPER_BG,
        "plot_bgcolor":  _PLOT_BG,
        "font":          _FONT,
        "margin":        dict(l=60, r=30, t=60, b=60),
        **kwargs,
    }


def _axis_style(**kwargs) -> dict:
    # Use dict-literal unpacking so caller kwargs silently override defaults
    # (dict(key=val, **kwargs) raises TypeError on duplicate keys in Python 3).
    return {
        "gridcolor":   _GRID_COLOR,
        "linecolor":   _GRID_COLOR,
        "tickcolor":   _TEXT_COLOR,
        "tickfont":    dict(color=_TEXT_COLOR),
        "title_font":  dict(color=_TEXT_COLOR),
        **kwargs,
    }


def _all_models(fairness_metrics: dict) -> List[str]:
    return list((fairness_metrics.get("per_model") or {}).keys())


def _all_attrs(fairness_metrics: dict, model: str) -> List[str]:
    return [
        k for k, v in ((fairness_metrics.get("per_model") or {}).get(model) or {}).items()
        if not k.startswith("_") and isinstance(v, dict)
    ]


def _pretty_model(name: str) -> str:
    return {"random_forest": "Random Forest",
            "xgboost": "XGBoost",
            "neural_network": "Neural Network"}.get(name, name.replace("_", " ").title())


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=_TEXT_COLOR),
    )
    fig.update_layout(**_base_layout())
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. FPR Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_fpr_comparison(fairness_metrics: dict) -> go.Figure:
    """Grouped bar chart: False Positive Rate per demographic group per model.

    One subplot per attribute (SEX, AGE, MARRIAGE).  All models shown side-by-side.
    A high FPR for a group means good customers there are incorrectly denied credit.
    """
    per_model = fairness_metrics.get("per_model") or {}
    if not per_model:
        return _empty_fig("No fairness metrics available. Train a model and run Compute + Audit.")

    models   = list(per_model.keys())
    _first_model_data = next(iter(per_model.values()))
    _first_model_data = {k: v for k, v in _first_model_data.items() if not k.startswith("_") and isinstance(v, dict)}
    attrs    = list(_first_model_data.keys())
    n_attrs  = len(attrs)

    fig = make_subplots(
        rows=n_attrs, cols=1,
        subplot_titles=[f"<b>{a}</b>" for a in attrs],
        vertical_spacing=0.14,
    )

    for row_idx, attr in enumerate(attrs, start=1):
        all_groups: dict[str, str] = {}
        for mdl in models:
            groups = (per_model.get(mdl) or {}).get(attr, {}).get("groups") or {}
            for g in groups:
                all_groups[g] = g
        group_names = sorted(all_groups.keys())

        for mdl in models:
            attr_data = (per_model.get(mdl) or {}).get(attr) or {}
            groups    = attr_data.get("groups") or {}
            fprs      = [groups.get(g, {}).get("fpr") for g in group_names]
            fig.add_trace(
                go.Bar(
                    name=_pretty_model(mdl),
                    x=group_names,
                    y=fprs,
                    marker_color=_model_color(mdl),
                    showlegend=(row_idx == 1),
                    text=[f"{v:.3f}" if v is not None else "—" for v in fprs],
                    textposition="outside",
                    textfont=dict(size=10, color=_TEXT_COLOR),
                ),
                row=row_idx, col=1,
            )

        fig.add_shape(
            type="line", line=dict(color=WARN_COLOR, dash="dash", width=1.5),
            x0=-0.5, x1=len(group_names) - 0.5, y0=0.10, y1=0.10,
            row=row_idx, col=1,
        )

    fig.update_layout(
        **_base_layout(
            barmode="group",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_COLOR)),
            height=340 * n_attrs,
        )
    )
    fig.update_layout(margin=dict(l=70, r=30, t=60, b=90))
    for i in range(1, n_attrs + 1):
        fig.update_xaxes(
            **_axis_style(title="Group"),
            tickangle=-30, automargin=True,
            row=i, col=1,
        )
        fig.update_yaxes(
            # Extended upper range so "outside" bar labels are never clipped
            **_axis_style(title="FPR", range=[0, 0.42]),
            row=i, col=1,
        )

    fig.add_annotation(
        text="Dashed line = 0.10 FPR concern threshold",
        xref="paper", yref="paper", x=1.0, y=-0.04,
        showarrow=False, font=dict(size=10, color=WARN_COLOR), xanchor="right",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Subgroup Performance Radar
# ─────────────────────────────────────────────────────────────────────────────

def plot_subgroup_radar(
    fairness_metrics: dict,
    model_name: Optional[str] = None,
) -> go.Figure:
    """One large polar chart per attribute (SEX, AGE, MARRIAGE) shown side-by-side.

    Axes: TPR · Specificity (1−FPR) · Selection Rate · Base Rate.
    A shrunken polygon means that group is being systematically under-served.
    """
    per_model = fairness_metrics.get("per_model") or {}
    if not per_model:
        return _empty_fig("No fairness metrics available.")

    model_name  = model_name or next(iter(per_model))
    attrs_data  = per_model.get(model_name) or {}
    attrs_data  = {k: v for k, v in attrs_data.items() if not k.startswith("_") and isinstance(v, dict)}
    attrs       = list(attrs_data.keys())
    if not attrs:
        return _empty_fig(f"No attribute data for {model_name}.")

    n           = len(attrs)
    axes_labels = ["TPR", "Specificity (1−FPR)", "Selection Rate", "Base Rate", "TPR"]
    specs       = [[{"type": "polar"}] * n]
    polar_keys  = ["polar", "polar2", "polar3", "polar4"]

    fig = make_subplots(
        rows=1, cols=n,
        specs=specs,
        subplot_titles=[f"<b>{a}</b>" for a in attrs],
        horizontal_spacing=0.06,
    )

    group_palette = [
        ("#EF5350", "solid"),
        ("#42A5F5", "solid"),
        ("#66BB6A", "solid"),
        ("#FFA726", "solid"),
        ("#AB47BC", "dash"),
        ("#26C6DA", "dash"),
    ]

    for col_idx, attr in enumerate(attrs, start=1):
        groups = (attrs_data.get(attr) or {}).get("groups") or {}
        already_in_legend = set()

        for grp_idx, (grp_name, grp_data) in enumerate(groups.items()):
            tpr       = grp_data.get("tpr")       or 0.0
            fpr       = grp_data.get("fpr")       or 0.0
            sel_rate  = grp_data.get("selection_rate") or 0.0
            base_rate = grp_data.get("base_rate") or 0.0
            values    = [tpr, 1.0 - fpr, sel_rate, base_rate, tpr]

            hex_c, dash = group_palette[grp_idx % len(group_palette)]
            r_v = int(hex_c[1:3], 16)
            g_v = int(hex_c[3:5], 16)
            b_v = int(hex_c[5:7], 16)
            fill_rgba = f"rgba({r_v},{g_v},{b_v},0.18)"
            show_leg  = grp_name not in already_in_legend
            already_in_legend.add(grp_name)

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=axes_labels,
                    fill="toself",
                    fillcolor=fill_rgba,
                    line=dict(color=hex_c, width=3, dash=dash),
                    name=grp_name,
                    showlegend=show_leg,
                    hovertemplate=(
                        f"<b>{grp_name}</b><br>"
                        "TPR: %{r[0]:.3f}<br>"
                        "Specificity: %{r[1]:.3f}<br>"
                        "Selection Rate: %{r[2]:.3f}<br>"
                        "Base Rate: %{r[3]:.3f}<extra></extra>"
                    ),
                ),
                row=1, col=col_idx,
            )

    # Style each polar subplot independently
    polar_cfg = dict(
        bgcolor=_PLOT_BG,
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            tickvals=[0.2, 0.4, 0.6, 0.8],
            ticktext=["0.2", "0.4", "0.6", "0.8"],
            gridcolor=_GRID_COLOR,
            linecolor=_GRID_COLOR,
            tickfont=dict(color="#94a3b8", size=10),
        ),
        angularaxis=dict(
            tickfont=dict(color=_TEXT_COLOR, size=13),
            linecolor=_GRID_COLOR,
            gridcolor=_GRID_COLOR,
        ),
    )
    polar_layout = {}
    for i in range(n):
        key = polar_keys[i]
        polar_layout[key] = polar_cfg

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"Subgroup Performance Radar — {_pretty_model(model_name)}",
                font=dict(size=16, color=_TEXT_COLOR),
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=_TEXT_COLOR, size=12),
                orientation="h",
                y=-0.08,
            ),
            height=560,
            **polar_layout,
        )
    )
    fig.update_layout(margin=dict(l=40, r=40, t=80, b=80))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Confusion Matrix Grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix_grid(metrics_report: dict) -> go.Figure:
    """Annotated confusion-matrix heatmap, one panel per model.
    Cells show raw counts (large) and row-normalised percentage (small).
    Rows = actual class, columns = predicted class.
    """
    models = [m for m in (metrics_report.get("models") or []) if m.get("confusion_matrix")]
    if not models:
        return _empty_fig("No confusion matrices available. Train at least one model.")

    n = len(models)
    titles = [_pretty_model(m["model_name"]) for m in models]
    fig = make_subplots(rows=1, cols=n, subplot_titles=titles)

    # Short labels prevent axis text from overlapping heatmap cells
    labels = ["No Default", "Default"]
    label_short = ["No Default", "Default"]

    for col_idx, m in enumerate(models, start=1):
        cm = np.array(m["confusion_matrix"])
        total = cm.sum()
        row_totals = cm.sum(axis=1, keepdims=True)
        cm_pct = np.where(row_totals > 0, cm / row_totals * 100, 0.0)

        text_vals = [
            [f"<b>{cm[r][c]:,}</b><br>{cm_pct[r][c]:.1f}%" for c in range(2)]
            for r in range(2)
        ]
        cell_labels = [
            ["TN", "FP"],
            ["FN", "TP"],
        ]
        hover_text = [
            [
                f"{cell_labels[r][c]}<br>Count: {cm[r][c]:,}<br>Row %: {cm_pct[r][c]:.1f}%<br>Total %: {cm[r][c]/total*100:.1f}%"
                for c in range(2)
            ]
            for r in range(2)
        ]

        colorscale = [
            [0.0, "#1a1a2e"],
            [0.5, "#1565C0"],
            [1.0, "#43A047"],
        ]

        fig.add_trace(
            go.Heatmap(
                z=cm_pct,
                x=labels,
                y=label_short,
                text=text_vals,
                texttemplate="%{text}",
                textfont=dict(size=13, color="#ffffff"),
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
                colorscale=colorscale,
                showscale=False,
                zmin=0, zmax=100,
            ),
            row=1, col=col_idx,
        )

    fig.update_layout(
        **_base_layout(
            title=dict(text="Confusion Matrix by Model", font=dict(size=16, color=_TEXT_COLOR)),
            height=480,
        )
    )
    fig.update_layout(margin=dict(l=100, r=30, t=100, b=80))
    for i in range(1, n + 1):
        fig.update_xaxes(
            **_axis_style(title="Predicted", tickfont=dict(size=12, color=_TEXT_COLOR)),
            automargin=True,
            row=1, col=i,
        )
        fig.update_yaxes(
            **_axis_style(title="Actual", tickfont=dict(size=12, color=_TEXT_COLOR)),
            automargin=True,
            row=1, col=i,
        )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature Importance Bias Flag
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance_bias(metrics_report: dict) -> go.Figure:
    """All models shown as stacked horizontal bar subplots — no dropdown required.

    Colour key:
      🔴 Red    = direct demographic feature (SEX, AGE, EDUCATION, MARRIAGE)
      🟠 Orange = proxy feature (LIMIT_BAL, BILL_AMT*, PAY_AMT*)
      🔵 Blue   = standard predictive feature
    Shared x-axis makes cross-model comparison straightforward.
    """
    model_list = [m for m in (metrics_report.get("models") or []) if m.get("feature_importance")]
    if not model_list:
        return _empty_fig("No models trained yet.")

    n_models    = len(model_list)
    n_features  = max(len(m["feature_importance"]) for m in model_list)
    row_height  = max(300, n_features * 22 + 70)

    fig = make_subplots(
        rows=n_models, cols=1,
        subplot_titles=[_pretty_model(m["model_name"]) for m in model_list],
        shared_xaxes=True,
        vertical_spacing=0.10,
    )

    legend_added: set = set()

    for row_idx, m in enumerate(model_list, start=1):
        fi         = sorted(m["feature_importance"], key=lambda x: x.get("importance", 0))
        features   = [f["feature"] for f in fi]
        importances = [f["importance"] for f in fi]

        for feat, imp in zip(features, importances):
            if feat in DEMOGRAPHIC_FEATURES:
                color, cat = FAIL_COLOR, "Direct demographic feature"
            elif feat in PROXY_FEATURES:
                color, cat = WARN_COLOR, "Proxy feature"
            else:
                color, cat = "#42A5F5", "Standard feature"

            show_leg = cat not in legend_added
            legend_added.add(cat)

            fig.add_trace(
                go.Bar(
                    x=[imp],
                    y=[feat],
                    orientation="h",
                    marker_color=color,
                    name=cat,
                    legendgroup=cat,
                    showlegend=show_leg,
                    text=[f"{imp:.4f}"],
                    textposition="outside",
                    textfont=dict(size=9, color=_TEXT_COLOR),
                    hovertemplate=f"<b>{feat}</b> ({cat})<br>Importance: {imp:.5f}<extra></extra>",
                ),
                row=row_idx, col=1,
            )

    fig.update_layout(
        **_base_layout(
            barmode="overlay",
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=_TEXT_COLOR),
                orientation="h",
                y=1.02, x=0,
            ),
            height=row_height * n_models,
        )
    )
    fig.update_layout(margin=dict(l=130, r=60, t=60, b=50))
    for i in range(1, n_models + 1):
        fig.update_xaxes(
            **_axis_style(title="Importance Score" if i == n_models else ""),
            automargin=True,
            row=i, col=1,
        )
        fig.update_yaxes(
            **_axis_style(title=""),
            automargin=True,
            row=i, col=1,
        )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model-Agent Agreement Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_agent_agreement_matrix(
    fairness_metrics: dict,
    metrics_report: dict,
    tpr_gap_threshold: float = 0.05,
    fpr_gap_threshold: float = 0.05,
) -> go.Figure:
    """Heatmap showing pass/fail verdicts for every model × fairness criterion.

    Criteria evaluated:
      • Passes 80% Rule (Disparate Impact ≥ 0.80)
      • TPR Gap ≤ threshold (Equal Opportunity)
      • FPR Gap ≤ threshold (Predictive Equality)
      • No demographic feature in top-10 importances
      • No proxy feature in top-10 importances
    """
    per_model = fairness_metrics.get("per_model") or {}
    model_list = {m["model_name"]: m for m in (metrics_report.get("models") or [])}

    # Combine models from both sources
    all_model_names = sorted(set(list(per_model.keys()) + list(model_list.keys())))
    if not all_model_names:
        return _empty_fig("No model data available.")

    # Build rows: one per (model, attribute) for fairness criteria
    # plus one per model for feature-importance criteria
    row_labels: List[str] = []
    col_labels = [
        "Passes\n80% Rule",
        f"TPR Gap\n≤{tpr_gap_threshold}",
        f"FPR Gap\n≤{fpr_gap_threshold}",
        "No Demo\nFeature (top-10)",
        "No Proxy\nFeature (top-10)",
    ]
    z_vals: List[List[float]] = []   # 1=pass, 0=fail, 0.5=N/A
    hover_vals: List[List[str]] = []

    for model_name in all_model_names:
        pretty = _pretty_model(model_name)
        attrs_data = (per_model.get(model_name) or {})

        # Feature-importance flags (apply once per model, not per attribute)
        fi = (model_list.get(model_name) or {}).get("feature_importance") or []
        top10 = {f["feature"] for f in fi[:10]}
        has_demo  = bool(top10 & DEMOGRAPHIC_FEATURES)
        has_proxy = bool(top10 & PROXY_FEATURES)

        for attr in sorted(k for k, v in attrs_data.items()
                           if not k.startswith("_") and isinstance(v, dict)):
            attr_payload = attrs_data[attr]
            row_labels.append(f"{pretty}\n{attr}")

            passes_80 = attr_payload.get("passes_80_pct_rule")
            eo = attr_payload.get("equalized_odds") or {}
            tpr_gap = eo.get("tpr_gap")
            fpr_gap = eo.get("fpr_gap")

            def _score(cond) -> float:
                return 1.0 if cond is True else (0.0 if cond is False else 0.5)

            def _hv(label, val, cond) -> str:
                status = "✓ PASS" if cond is True else ("✗ FAIL" if cond is False else "— N/A")
                return f"{label}<br>Value: {val}<br>{status}"

            p80   = passes_80
            tpr_p = (tpr_gap <= tpr_gap_threshold) if tpr_gap is not None else None
            fpr_p = (fpr_gap <= fpr_gap_threshold) if fpr_gap is not None else None
            demo_p  = not has_demo
            proxy_p = not has_proxy

            z_row = [
                _score(p80),
                _score(tpr_p),
                _score(fpr_p),
                _score(demo_p),
                _score(proxy_p),
            ]
            h_row = [
                _hv("Passes 80% Rule", f"{attr_payload.get('disparate_impact', '—'):.3f}" if attr_payload.get('disparate_impact') else "—", p80),
                _hv(f"TPR Gap ≤ {tpr_gap_threshold}", f"{tpr_gap:.4f}" if tpr_gap is not None else "—", tpr_p),
                _hv(f"FPR Gap ≤ {fpr_gap_threshold}", f"{fpr_gap:.4f}" if fpr_gap is not None else "—", fpr_p),
                _hv("Demo feature in top-10", ", ".join(top10 & DEMOGRAPHIC_FEATURES) or "none", demo_p),
                _hv("Proxy feature in top-10", ", ".join(top10 & PROXY_FEATURES) or "none", proxy_p),
            ]
            z_vals.append(z_row)
            hover_vals.append(h_row)

    if not row_labels:
        return _empty_fig("No fairness criteria could be evaluated.")

    colorscale = [
        [0.0,  FAIL_COLOR],
        [0.45, FAIL_COLOR],
        [0.50, "#B0BEC5"],
        [0.55, "#B0BEC5"],
        [1.0,  PASS_COLOR],
    ]

    tick_vals = [f"{'✓' if v >= 0.9 else ('—' if 0.4 < v < 0.6 else '✗')}"
                 for row in z_vals for v in row]

    text_matrix = [
        ["✓" if v >= 0.9 else ("—" if 0.4 < v < 0.6 else "✗") for v in row]
        for row in z_vals
    ]

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=col_labels,
        y=row_labels,
        text=text_matrix,
        texttemplate="<b>%{text}</b>",
        hovertext=hover_vals,
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale=colorscale,
        showscale=False,
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text="Model Fairness Criteria — Agreement Matrix", font=dict(size=16, color=_TEXT_COLOR)),
            xaxis=_axis_style(side="top"),
            yaxis=_axis_style(autorange="reversed"),
            height=max(320, len(row_labels) * 48 + 120),
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Demographic Default Rate Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_default_rate_heatmap(bias_signals: dict) -> go.Figure:
    """Colour-intensity bar chart showing ground-truth default rates per
    demographic group across all attributes, with the overall baseline marked.
    """
    per_group = bias_signals.get("per_group_default_rate") or {}
    overall_rate = (bias_signals.get("overall") or {}).get("default_rate", 0.0)
    if not per_group:
        return _empty_fig("No bias signals available. Load the dataset and Compute Signals.")

    attrs = [a for a in ["SEX", "EDUCATION", "MARRIAGE", "AGE"] if a in per_group]
    n = len(attrs)
    fig = make_subplots(rows=n, cols=1, subplot_titles=[f"<b>{a}</b>" for a in attrs],
                        vertical_spacing=0.06)

    for row_idx, attr in enumerate(attrs, start=1):
        groups_data = per_group[attr]
        # Sort groups by default rate descending for readability
        sorted_groups = sorted(groups_data.items(), key=lambda x: x[1]["default_rate"], reverse=True)
        group_labels  = [decode_group_label(attr, g) for g, _ in sorted_groups]
        default_rates = [v["default_rate"] for _, v in sorted_groups]
        counts        = [v["count"] for _, v in sorted_groups]
        gaps          = [v["gap_vs_overall_pp"] for _, v in sorted_groups]

        colors = [
            FAIL_COLOR if r > overall_rate * 1.10 else (
                PASS_COLOR if r < overall_rate * 0.90 else "#42A5F5"
            )
            for r in default_rates
        ]

        fig.add_trace(
            go.Bar(
                x=group_labels,
                y=default_rates,
                marker_color=colors,
                text=[f"{r:.1%}" for r in default_rates],
                textposition="outside",
                textfont=dict(size=10, color=_TEXT_COLOR),
                customdata=list(zip(counts, gaps)),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Default Rate: %{y:.2%}<br>"
                    "Count: %{customdata[0]:,}<br>"
                    "Gap vs Overall: %{customdata[1]:+.2f} pp"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row_idx, col=1,
        )
        # Overall baseline line
        fig.add_shape(
            type="line",
            line=dict(color=WARN_COLOR, dash="dot", width=1.5),
            x0=-0.5, x1=len(group_labels) - 0.5,
            y0=overall_rate, y1=overall_rate,
            row=row_idx, col=1,
        )

    fig.update_layout(
        **_base_layout(
            title=dict(text="Ground-Truth Default Rate by Demographic Group", font=dict(size=16, color=_TEXT_COLOR)),
            height=270 * n + 80,
        )
    )
    fig.update_layout(margin=dict(l=70, r=30, t=50, b=80))
    for i in range(1, n + 1):
        fig.update_xaxes(
            **_axis_style(title=""),
            tickangle=-30, automargin=True,
            row=i, col=1,
        )
        fig.update_yaxes(**_axis_style(title="Default Rate", tickformat=".0%"), row=i, col=1)

    fig.add_annotation(
        text=f"Dotted line = overall default rate ({overall_rate:.1%})",
        xref="paper", yref="paper", x=1.0, y=-0.02,
        showarrow=False, font=dict(size=10, color=WARN_COLOR),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Equalized Odds Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_equalized_odds(fairness_metrics: dict) -> go.Figure:
    """Scatter plot: TPR gap (x) vs FPR gap (y) per model × attribute.

    Points close to (0,0) are the fairest. Coloured by attribute, shaped by model.
    Concentric threshold circles drawn at 0.05 and 0.10.
    """
    per_model = fairness_metrics.get("per_model") or {}
    if not per_model:
        return _empty_fig("No fairness metrics available.")

    fig = go.Figure()

    # Threshold circles
    for r, label in [(0.05, "5% threshold"), (0.10, "10% threshold")]:
        theta = np.linspace(0, 2 * np.pi, 200)
        fig.add_trace(go.Scatter(
            x=r * np.cos(theta), y=r * np.sin(theta),
            mode="lines",
            line=dict(color=WARN_COLOR, dash="dash", width=1),
            name=label, showlegend=True,
            hoverinfo="skip",
        ))

    symbols = ["circle", "square", "diamond", "cross"]
    model_syms = {m: symbols[i % len(symbols)] for i, m in enumerate(per_model)}

    for model_name, attrs_data in per_model.items():
        attrs_data = {k: v for k, v in attrs_data.items() if not k.startswith("_") and isinstance(v, dict)}
        for attr, attr_payload in attrs_data.items():
            if not isinstance(attr_payload, dict):
                continue
            eo = attr_payload.get("equalized_odds") or {}
            tpr_gap = eo.get("tpr_gap")
            fpr_gap = eo.get("fpr_gap")
            if tpr_gap is None or fpr_gap is None:
                continue

            di = attr_payload.get("disparate_impact")
            passes = attr_payload.get("passes_80_pct_rule", False)
            border = PASS_COLOR if passes else FAIL_COLOR

            fig.add_trace(go.Scatter(
                x=[tpr_gap], y=[fpr_gap],
                # Labels removed from chart surface — they overlap when points
                # are close together. Identification via legend + rich hover.
                mode="markers",
                marker=dict(
                    size=18,
                    color=ATTR_COLORS.get(attr, "#90A4AE"),
                    symbol=model_syms[model_name],
                    line=dict(color=border, width=2),
                ),
                name=f"{_pretty_model(model_name)} / {attr}",
                customdata=[[di, passes]],
                hovertemplate=(
                    f"<b>{_pretty_model(model_name)} — {attr}</b><br>"
                    "TPR Gap: %{x:.4f}<br>"
                    "FPR Gap: %{y:.4f}<br>"
                    "Disparate Impact: %{customdata[0]:.4f}<br>"
                    "80% Rule: %{customdata[1]}"
                    "<extra></extra>"
                ),
                showlegend=True,
            ))

    # Perfect-fairness origin
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(size=10, color=PASS_COLOR, symbol="star"),
        name="Perfect fairness (0, 0)",
    ))

    fig.update_layout(
        **_base_layout(
            title=dict(text="Equalized Odds: TPR Gap vs FPR Gap", font=dict(size=16, color=_TEXT_COLOR)),
            xaxis=_axis_style(title="TPR Gap (Equal Opportunity)", zeroline=True, zerolinecolor=_GRID_COLOR),
            yaxis=_axis_style(title="FPR Gap (Predictive Equality)", zeroline=True, zerolinecolor=_GRID_COLOR),
            # Legend placed below chart so it never overlaps data points
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=_TEXT_COLOR, size=11),
                orientation="h",
                x=0, y=-0.22,
                xanchor="left",
            ),
            height=560,
            margin=dict(l=70, r=30, t=60, b=160),
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. Threshold Impact by Group
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_impact(
    metrics_report: dict,
    y_true: List[int],
    sensitive_df: "pd.DataFrame",
    metric: str = "fpr",
) -> go.Figure:
    """All attributes × all models shown at once — no dropdowns required.

    Layout: one subplot per attribute (rows), lines coloured by demographic group,
    line dash distinguishes models (solid / dash / dot).

    Requires prediction probabilities — re-train models if a placeholder appears.
    """
    from .bias_metrics import _bucket_age  # local import to avoid circular

    models_list = [m for m in (metrics_report.get("models") or []) if m.get("probabilities")]
    if not models_list:
        return _empty_fig(
            "Prediction probabilities not stored.\n"
            "Re-train the model(s) to enable threshold analysis."
        )
    if sensitive_df.empty or not y_true:
        return _empty_fig("Dataset not loaded. Load the dataset to enable this chart.")

    avail_attrs  = [c for c in sensitive_df.columns]
    n_attrs      = len(avail_attrs)
    y_true_arr   = np.array(y_true)
    thresholds   = np.linspace(0.05, 0.95, 19)
    metric_label = {
        "fpr": "False Positive Rate",
        "tpr": "True Positive Rate",
        "selection_rate": "Selection Rate",
    }.get(metric, metric.upper())
    model_dashes = ["solid", "dash", "dot", "dashdot"]
    group_colors = ["#EF5350", "#42A5F5", "#66BB6A", "#FFA726", "#AB47BC", "#26C6DA"]

    fig = make_subplots(
        rows=n_attrs, cols=1,
        subplot_titles=[f"<b>{a}</b>" for a in avail_attrs],
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    legend_seen: set = set()

    for row_idx, attr in enumerate(avail_attrs, start=1):
        attr_series = sensitive_df[attr]
        if attr == "AGE":
            group_series = pd.Series(_bucket_age(attr_series.tolist()), index=attr_series.index)
        else:
            group_series = attr_series.astype(int).astype(str).map(
                lambda code: decode_group_label(attr, code)
            )
        unique_groups = sorted(group_series.unique())

        for mdl_idx, m in enumerate(models_list):
            probs_arr = np.array(m["probabilities"])
            if len(probs_arr) != len(y_true_arr):
                continue
            mdl_name  = m["model_name"]
            dash_style = model_dashes[mdl_idx % len(model_dashes)]

            for grp_idx, grp in enumerate(unique_groups):
                mask = (group_series == grp).values
                y_g  = y_true_arr[mask]
                p_g  = probs_arr[mask]
                ys: List[Optional[float]] = []

                for t in thresholds:
                    pred = (p_g >= t).astype(int)
                    tp = int(((pred == 1) & (y_g == 1)).sum())
                    fp = int(((pred == 1) & (y_g == 0)).sum())
                    fn = int(((pred == 0) & (y_g == 1)).sum())
                    tn = int(((pred == 0) & (y_g == 0)).sum())
                    if metric == "fpr":
                        val = fp / (fp + tn) if (fp + tn) > 0 else None
                    elif metric == "tpr":
                        val = tp / (tp + fn) if (tp + fn) > 0 else None
                    else:
                        val = (tp + fp) / len(y_g) if len(y_g) > 0 else None
                    ys.append(val)

                leg_key   = f"{grp} ({_pretty_model(mdl_name)})"
                show_leg  = leg_key not in legend_seen
                legend_seen.add(leg_key)
                color     = group_colors[grp_idx % len(group_colors)]

                fig.add_trace(
                    go.Scatter(
                        x=thresholds, y=ys,
                        mode="lines+markers",
                        name=leg_key,
                        legendgroup=leg_key,
                        showlegend=show_leg,
                        line=dict(color=color, width=2, dash=dash_style),
                        marker=dict(size=4),
                        hovertemplate=(
                            f"<b>{grp}</b> [{_pretty_model(mdl_name)}]<br>"
                            f"Threshold: %{{x:.2f}}<br>{metric_label}: %{{y:.3f}}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row_idx, col=1,
                )

        fig.add_vline(
            x=0.5, line_dash="dash", line_color=WARN_COLOR,
            annotation_text="Default 0.5" if row_idx == 1 else "",
            row=row_idx, col=1,
        )

    fig.update_layout(
        **_base_layout(
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=_TEXT_COLOR, size=10),
                orientation="v",
            ),
            height=280 * n_attrs,
        )
    )
    fig.update_layout(margin=dict(l=70, r=30, t=50, b=60))
    for i in range(1, n_attrs + 1):
        fig.update_xaxes(
            **_axis_style(title="Decision Threshold" if i == n_attrs else ""),
            tickformat=".2f", automargin=True,
            row=i, col=1,
        )
        fig.update_yaxes(
            **_axis_style(title=metric_label, tickformat=".2f"),
            row=i, col=1,
        )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. Proxy Bias Network (Sankey)
# ─────────────────────────────────────────────────────────────────────────────

def plot_proxy_bias_network(bias_signals: dict, gap_threshold: float = 0.10) -> go.Figure:
    """Sankey diagram showing which proxy features have large credit-limit
    or bill-amount gaps for which demographic groups.

    Left nodes  = proxy features (LIMIT_BAL, Avg BILL_AMT).
    Right nodes = demographic groups with |gap| > gap_threshold.
    Link width  = absolute gap magnitude.
    Link colour = red (under-represented) / blue (over-represented).
    """
    limit_data = bias_signals.get("limit_bal_by_group") or {}
    bill_data  = bias_signals.get("bill_amt_skew") or {}

    node_labels: List[str] = []
    node_colors: List[str] = []
    sources: List[int] = []
    targets: List[int] = []
    values: List[float] = []
    link_colors: List[str] = []
    link_labels: List[str] = []

    node_index: Dict[str, int] = {}

    def _get_or_add_node(label: str, color: str) -> int:
        if label not in node_index:
            node_index[label] = len(node_labels)
            node_labels.append(label)
            node_colors.append(color)
        return node_index[label]

    feature_colors = {"LIMIT_BAL": "#1565C0", "Avg BILL_AMT": "#0097A7"}

    # LIMIT_BAL gaps
    for attr in ["SEX", "EDUCATION", "MARRIAGE"]:
        groups = limit_data.get(attr) or {}
        feat_node = _get_or_add_node("LIMIT_BAL", feature_colors["LIMIT_BAL"])
        for code, payload in groups.items():
            gap = payload.get("pct_gap_vs_overall") or 0.0
            if abs(gap) < gap_threshold:
                continue
            label = decode_group_label(attr, code)
            grp_node = _get_or_add_node(f"{attr}: {label}", ATTR_COLORS.get(attr, "#78909C"))
            sources.append(feat_node)
            targets.append(grp_node)
            values.append(abs(gap) * 100)  # % as magnitude
            color = "rgba(229,57,53,0.5)" if gap < 0 else "rgba(67,160,71,0.5)"
            link_colors.append(color)
            direction = "under-represented" if gap < 0 else "over-represented"
            link_labels.append(f"{gap:+.1%} ({direction})")

    # BILL_AMT skew gaps
    for attr in ["SEX", "EDUCATION", "MARRIAGE"]:
        groups = bill_data.get(attr) or {}
        if not isinstance(groups, dict):
            continue
        feat_node = _get_or_add_node("Avg BILL_AMT", feature_colors["Avg BILL_AMT"])
        for code, payload in groups.items():
            if not isinstance(payload, dict):
                continue
            gap = payload.get("pct_gap_vs_overall") or 0.0
            if abs(gap) < gap_threshold:
                continue
            label = decode_group_label(attr, code)
            grp_node = _get_or_add_node(f"{attr}: {label}", ATTR_COLORS.get(attr, "#78909C"))
            sources.append(feat_node)
            targets.append(grp_node)
            values.append(abs(gap) * 100)
            color = "rgba(229,57,53,0.5)" if gap < 0 else "rgba(67,160,71,0.5)"
            link_colors.append(color)
            direction = "under-represented" if gap < 0 else "over-represented"
            link_labels.append(f"{gap:+.1%} ({direction})")

    if not sources:
        return _empty_fig(
            f"No proxy-bias links exceed the {gap_threshold:.0%} gap threshold.\n"
            "Try lowering the threshold."
        )

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="#334155", width=0.5),
            label=node_labels,
            color=node_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            label=link_labels,
            hovertemplate=(
                "%{source.label} → %{target.label}<br>"
                "Gap magnitude: %{value:.1f}%<br>"
                "%{label}<extra></extra>"
            ),
        ),
    ))

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"Proxy Bias Network — gaps > {gap_threshold:.0%} shown<br>"
                     "<sup>Red = under-represented (lower credit/bill), "
                     "Green = over-represented (higher credit/bill)</sup>",
                font=dict(size=15, color=_TEXT_COLOR),
            ),
            height=520,
        )
    )
    fig.update_layout(margin=dict(l=20, r=20, t=90, b=20))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. Precision-Recall Frontier
# ─────────────────────────────────────────────────────────────────────────────

def plot_precision_recall_frontier(
    metrics_report: dict,
    y_true: List[int],
) -> go.Figure:
    """Precision-Recall curves for each model.

    Requires prediction probabilities (stored after training v2+).
    Falls back to a single-point scatter if probabilities are missing.
    """
    models = [m for m in (metrics_report.get("models") or [])]
    if not models:
        return _empty_fig("No models available.")

    y_true_arr = np.array(y_true)
    fig = go.Figure()
    any_curves = False

    for m in models:
        name   = m["model_name"]
        probs  = m.get("probabilities") or []
        color  = _model_color(name)
        pretty = _pretty_model(name)
        auprc  = m.get("metrics", {}).get("average_precision")

        if probs and len(probs) == len(y_true_arr):
            any_curves = True
            from sklearn.metrics import precision_recall_curve
            precision, recall, _ = precision_recall_curve(y_true_arr, np.array(probs))
            legend_label = f"{pretty} (AUPRC={auprc:.3f})" if auprc else pretty
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode="lines",
                name=legend_label,
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{pretty}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>",
            ))
        else:
            # Fallback: single point — no inline text to avoid overlapping when
            # precision/recall values are similar across models. Rely on legend.
            metrics = m.get("metrics", {})
            rc = metrics.get("recall")
            pr = metrics.get("precision")
            if rc is not None and pr is not None:
                legend_label = f"{pretty} (no curve — re-train for full PR)"
                fig.add_trace(go.Scatter(
                    x=[rc], y=[pr],
                    mode="markers",
                    marker=dict(size=16, color=color, symbol="star"),
                    name=legend_label,
                    hovertemplate=f"<b>{pretty}</b><br>Recall: {rc:.3f}<br>Precision: {pr:.3f}<extra></extra>",
                ))

    # No-skill baseline
    pos_rate = float(y_true_arr.mean()) if len(y_true_arr) > 0 else 0.22
    fig.add_hline(
        y=pos_rate, line_dash="dot", line_color=NA_COLOR,
        annotation_text=f"No-skill baseline ({pos_rate:.2%})",
        annotation_font_color=NA_COLOR,
    )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="Precision-Recall Frontier" + ("" if any_curves else " (re-train for full curves)"),
                font=dict(size=16, color=_TEXT_COLOR),
            ),
            xaxis=_axis_style(title="Recall", range=[0, 1]),
            yaxis=_axis_style(title="Precision", range=[0, 1.05]),
            # Legend below chart prevents long model names from overlapping curves
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=_TEXT_COLOR, size=11),
                orientation="h",
                x=0, y=-0.18,
                xanchor="left",
            ),
            height=520,
            margin=dict(l=70, r=30, t=60, b=120),
        )
    )
    return fig
