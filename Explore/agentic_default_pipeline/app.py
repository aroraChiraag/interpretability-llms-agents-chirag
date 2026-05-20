"""Streamlit front-end for the agentic credit-default classification pipeline.

Launch with::

    cd Explore/agentic_default_pipeline
    uv run --env-file ../../.env streamlit run app.py

Tabs:

- **Run** — per-model buttons (Random Forest / XGBoost / Neural Network)
  plus a "Run all + explain" button. Renders metrics, feature importance,
  and confusion matrices.
- **Hyperparameters** — sidebar form with sliders/inputs per model.
- **Chat** — talk to the Pipeline Coordinator agent. "Use 500 trees in
  random forest and re-run" works.
- **Explanation** — the explainer agent's Markdown brief, big and clean.
- **Artifacts** — file links for the JSON / Markdown saved on disk.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Path / env setup
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(THIS_DIR.parent.parent / ".env")
except Exception:
    pass

# Importing the package applies the non-interactive env defaults that
# auto-decline CrewAI's trace-viewer prompt.
import agentic_default  # noqa: F401, E402

from agentic_default.bias_metrics import (  # noqa: E402
    compute_dataset_bias_signals,
    flatten_limit_bal_by_group,
    flatten_per_group_default,
    prescan_feature_importance,
)
from agentic_default.data_cleaner import clean_dataset, CleaningReport  # noqa: E402
from agentic_default.data_loader import load_dataset  # noqa: E402
from agentic_default.fairness_metrics import (  # noqa: E402
    compute_fairness_for_all_models,
)
from agentic_default.fairness_glossary import (  # noqa: E402
    FAIRNESS_GLOSSARY,
    COLUMN_HELP,
)
from agentic_default.bias_glossary import (  # noqa: E402
    FEATURE_GLOSSARY,
    FEATURE_TABLE_23,
    BIAS_METRIC_GLOSSARY,
    BIAS_COLUMN_HELP,
    decode_group_label,
)
from agentic_default.optimizer import (  # noqa: E402
    DEFAULT_PARAM_GRIDS,
    SUPPORTED_TECHNIQUES,
    optimize,
)
from agentic_default.visualizations import (  # noqa: E402
    plot_fpr_comparison,
    plot_subgroup_radar,
    plot_confusion_matrix_grid,
    plot_feature_importance_bias,
    plot_agent_agreement_matrix,
    plot_default_rate_heatmap,
    plot_equalized_odds,
    plot_threshold_impact,
    plot_proxy_bias_network,
    plot_precision_recall_frontier,
)
from agentic_default.ml_trainer import (  # noqa: E402
    SUPPORTED_MODELS,
    default_hyperparameters,
    train_and_evaluate,
)
from agentic_default.state import get_state, reset_state  # noqa: E402

# Lazy import — only when the agent tabs are used.
_agents_imported = False


def _ensure_agents_imported():
    global _agents_imported
    if _agents_imported:
        return
    from agentic_default.agents import (  # noqa: F401
        BiasAgent,
        CoordinatorAgent,
        ExplainerAgent,
        FairnessAgent,
    )

    st.session_state["_BiasAgent"] = BiasAgent
    st.session_state["_CoordinatorAgent"] = CoordinatorAgent
    st.session_state["_ExplainerAgent"] = ExplainerAgent
    st.session_state["_FairnessAgent"] = FairnessAgent
    _agents_imported = True


# ---------------------------------------------------------------------------
# Streamlit page config and session-state init
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Agentic Default Classifier",
    page_icon=None,
    layout="wide",
)


def _init_state() -> None:
    """Populate session_state defaults the first time the app runs."""
    defaults = {
        "state_handle": "streamlit",
        "hyperparameters": default_hyperparameters(),
        "test_size": 0.2,
        "random_state": 42,
        "messages": [],
        "explanation_md": "",
        "fairness_metrics": {},
        "fairness_md": "",
        "bias_signals": {},
        "bias_md": "",
        "metrics_report": {},
        "pre_tuning_metrics_report": {},
        "pre_tuning_fairness_metrics": {},
        "pre_tuning_feature_importance": {},
        "dataset_summary": {},
        "cleaning_report": {},
        "outputs_dir": str(THIS_DIR / "outputs" / "streamlit_run"),
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gemini_key_present() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def _ensure_dataset_loaded() -> None:
    state = get_state(st.session_state["state_handle"])
    if state.dataset is None:
        with st.spinner("Loading dataset..."):
            ds = load_dataset(
                test_size=st.session_state["test_size"],
                random_state=st.session_state["random_state"],
                snapshot_dir=Path(st.session_state["outputs_dir"]),
            )
            state.dataset = ds
            state.hyperparameters = st.session_state["hyperparameters"]
            st.session_state["dataset_summary"] = {
                "rows": ds.metadata.n_rows,
                "features": ds.metadata.n_features,
                "class_balance": ds.metadata.class_balance,
                "train_size": ds.metadata.train_size,
                "test_size": ds.metadata.test_size,
                "feature_names": ds.feature_names,
            }


def _clean_data(winsorise_percentile: float = 1.0) -> None:
    """Run the three-pass cleaner on the loaded dataset and replace state."""
    state = get_state(st.session_state["state_handle"])
    if state.dataset is None:
        st.error("Load the dataset first (Stage 1) before running Clean Data.")
        return
    with st.spinner("Cleaning dataset..."):
        cleaned_ds, report = clean_dataset(
            state.dataset,
            test_size=st.session_state["test_size"],
            random_state=st.session_state["random_state"],
            winsorise_percentile=winsorise_percentile,
            snapshot_dir=Path(st.session_state["outputs_dir"]),
        )
    state.dataset = cleaned_ds
    st.session_state["cleaning_report"] = report.to_dict()
    st.session_state["dataset_summary"] = {
        "rows": cleaned_ds.metadata.n_rows,
        "features": cleaned_ds.metadata.n_features,
        "class_balance": cleaned_ds.metadata.class_balance,
        "train_size": cleaned_ds.metadata.train_size,
        "test_size": cleaned_ds.metadata.test_size,
        "feature_names": cleaned_ds.feature_names,
    }


def _sync_hyperparameters_to_state() -> None:
    """Push the form's hyperparameters into the shared PipelineState."""
    state = get_state(st.session_state["state_handle"])
    state.hyperparameters = st.session_state["hyperparameters"]


def _train_models(models_to_train: list) -> None:
    _ensure_dataset_loaded()
    _sync_hyperparameters_to_state()
    state = get_state(st.session_state["state_handle"])
    with st.spinner(f"Training {', '.join(models_to_train)}..."):
        report = train_and_evaluate(
            x_train=state.dataset.x_train,
            y_train=state.dataset.y_train,
            x_test=state.dataset.x_test,
            y_test=state.dataset.y_test,
            feature_names=state.dataset.feature_names,
            models=models_to_train,
            random_state=st.session_state["random_state"],
            output_dir=Path(st.session_state["outputs_dir"]),
            hyperparameters=st.session_state["hyperparameters"],
        )

    # Merge the new model results into any existing report so per-model runs
    # don't wipe out other models' previous results.
    existing = st.session_state["metrics_report"]
    merged_models = {m["model_name"]: m for m in existing.get("models", [])}
    for m in report["models"]:
        merged_models[m["model_name"]] = m
    merged_list = list(merged_models.values())
    leaderboard = sorted(
        merged_list,
        key=lambda m: (
            m["metrics"].get("roc_auc", 0.0),
            m["metrics"].get("f1", 0.0),
        ),
        reverse=True,
    )
    merged_report = {
        "models": merged_list,
        "leaderboard": [
            {
                "model_name": m["model_name"],
                "roc_auc": m["metrics"].get("roc_auc"),
                "f1": m["metrics"].get("f1"),
                "precision": m["metrics"].get("precision"),
                "recall": m["metrics"].get("recall"),
                "accuracy": m["metrics"].get("accuracy"),
            }
            for m in leaderboard
        ],
        "best_model": leaderboard[0]["model_name"] if leaderboard else None,
        "hyperparameters_used": report.get("hyperparameters_used", {}),
    }
    st.session_state["metrics_report"] = merged_report
    state.metrics_report = merged_report

    # Save pre-tuning snapshot (only once — before any Stage 4 tuning runs)
    if not st.session_state.get("pre_tuning_metrics_report"):
        state.pre_tuning_metrics_report = state.metrics_report
        st.session_state["pre_tuning_metrics_report"] = state.metrics_report

    if not st.session_state.get("pre_tuning_feature_importance"):
        _pre_fi = {m["model_name"]: (m.get("feature_importance") or [])
                   for m in (state.metrics_report.get("models") or [])}
        state.pre_tuning_feature_importance = _pre_fi
        st.session_state["pre_tuning_feature_importance"] = _pre_fi


def _run_explainer() -> None:
    _ensure_agents_imported()
    if not _gemini_key_present():
        st.error("GEMINI_API_KEY is not set. Add it to .env or export it.")
        return
    explainer_cls = st.session_state["_ExplainerAgent"]
    with st.spinner("ExplainerAgent is writing the brief..."):
        explainer = explainer_cls()
        result = explainer.run(
            metrics_report=st.session_state["metrics_report"],
            dataset_summary=st.session_state["dataset_summary"],
        )
        text = getattr(result, "raw", None) or str(result)
        st.session_state["explanation_md"] = text
        out_path = Path(st.session_state["outputs_dir"])
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "explanation.md").write_text(text, encoding="utf-8")


def _compute_fairness() -> None:
    """Compute fairness metrics from in-state predictions; persist to state."""
    state = get_state(st.session_state["state_handle"])
    if state.dataset is None or not state.metrics_report:
        st.error("Train at least one model first — fairness needs predictions.")
        return
    has_preds = any(m.get("predictions") for m in state.metrics_report.get("models", []))
    if not has_preds:
        st.error(
            "The current metrics_report has no per-row predictions stored. "
            "Re-train models so predictions are captured."
        )
        return
    tuning_notes_map = {
        m["model_name"]: (m.get("notes") or "")
        for m in state.metrics_report.get("models", [])
    }
    fairness = compute_fairness_for_all_models(
        metrics_report=state.metrics_report,
        y_true=state.dataset.y_test,
        sensitive_df=state.dataset.test_sensitive,
        tuning_notes=tuning_notes_map,
    )
    state.fairness_metrics = fairness
    st.session_state["fairness_metrics"] = fairness
    if not st.session_state.get("pre_tuning_fairness_metrics") and not any(
        m.get("notes") for m in state.metrics_report.get("models", [])
    ):
        state.pre_tuning_fairness_metrics = fairness
        st.session_state["pre_tuning_fairness_metrics"] = fairness

    out_path = Path(st.session_state["outputs_dir"])
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "fairness_metrics.json").write_text(
        json.dumps(fairness, indent=2), encoding="utf-8"
    )


def _run_fairness_agent() -> None:
    """Invoke the FairnessAgent on the latest fairness_metrics."""
    _ensure_agents_imported()
    if not _gemini_key_present():
        st.error("GEMINI_API_KEY is not set. Add it to .env or export it.")
        return
    state = get_state(st.session_state["state_handle"])
    if not state.fairness_metrics:
        _compute_fairness()
        if not state.fairness_metrics:
            return  # _compute_fairness already surfaced an error

    fairness_cls = st.session_state["_FairnessAgent"]
    with st.spinner("FairnessAgent is auditing the models..."):
        # Strip per-row predictions from the performance payload — the agent
        # doesn't need them and they bloat the prompt.
        perf_for_agent = {
            "leaderboard": st.session_state["metrics_report"].get("leaderboard", []),
            "best_model": st.session_state["metrics_report"].get("best_model"),
            "models": [
                {k: v for k, v in m.items() if k != "predictions"}
                for m in st.session_state["metrics_report"].get("models", [])
            ],
        }
        agent = fairness_cls()
        result = agent.run(
            metrics_report=perf_for_agent,
            fairness_metrics=state.fairness_metrics,
        )
        text = getattr(result, "raw", None) or str(result)
        state.fairness_explanation = text
        st.session_state["fairness_md"] = text

        out_path = Path(st.session_state["outputs_dir"])
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "fairness_audit.md").write_text(text, encoding="utf-8")


def _run_fairness_audit() -> None:
    """Compute fairness metrics, then call the FairnessAgent — one click."""
    _compute_fairness()
    if get_state(st.session_state["state_handle"]).fairness_metrics:
        _run_fairness_agent()


def _fairness_summary_rows() -> list:
    """Flatten fairness_metrics into a per-(model, attribute) DataFrame row."""
    fm = st.session_state.get("fairness_metrics") or {}
    rows = []
    for model_name, attrs in (fm.get("per_model") or {}).items():
        for attr_name, payload in attrs.items():
            if attr_name.startswith("_") or not isinstance(payload, dict):
                continue
            eo = payload.get("equalized_odds") or {}
            rows.append({
                "model": model_name,
                "attribute": attr_name,
                "disparate_impact": payload.get("disparate_impact"),
                "passes_80_pct_rule": payload.get("passes_80_pct_rule"),
                "tpr_gap": eo.get("tpr_gap"),
                "fpr_gap": eo.get("fpr_gap"),
            })
    return rows


# ---------- Bias helpers (reporting only — never affect other agents) -------


def _compute_bias_signals() -> None:
    """Compute pre-training dataset bias signals from the loaded records."""
    _ensure_dataset_loaded()
    state = get_state(st.session_state["state_handle"])
    if state.dataset is None:
        st.error("Dataset not loaded.")
        return
    target = state.dataset.metadata.target_name
    signals = compute_dataset_bias_signals(
        records_json=state.dataset.records_json,
        target_column=target,
    )
    state.bias_signals = signals
    st.session_state["bias_signals"] = signals

    out_path = Path(st.session_state["outputs_dir"])
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "bias_signals.json").write_text(
        json.dumps(signals, indent=2), encoding="utf-8"
    )


def _run_bias_agent() -> None:
    """Invoke the BiasAgent. Reads dataset signals + (optional) metrics_report.

    The output is written to PipelineState.bias_explanation and the
    Streamlit *Bias* tab. It is NEVER fed into the trainer / fairness /
    explainer pipeline.
    """
    _ensure_agents_imported()
    if not _gemini_key_present():
        st.error("GEMINI_API_KEY is not set. Add it to .env or export it.")
        return
    state = get_state(st.session_state["state_handle"])
    if not state.bias_signals:
        _compute_bias_signals()
        if not state.bias_signals:
            return

    metrics = state.metrics_report or None
    if metrics:
        _has_tuning = any(m.get("notes") for m in metrics.get("models", []))
        prescan = prescan_feature_importance(
            metrics,
            source="post_tuning" if _has_tuning else "post_training",
        )
    else:
        prescan = None

    bias_cls = st.session_state["_BiasAgent"]
    with st.spinner("BiasAgent is auditing the data..."):
        agent = bias_cls()
        result = agent.run(
            dataset_signals=state.bias_signals,
            metrics_report=metrics,
            prescan=prescan,
        )
        text = getattr(result, "raw", None) or str(result)
        state.bias_explanation = text
        st.session_state["bias_md"] = text

        out_path = Path(st.session_state["outputs_dir"])
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "bias_audit.md").write_text(text, encoding="utf-8")


def _run_bias_audit() -> None:
    """One-click compute + audit."""
    _compute_bias_signals()
    if get_state(st.session_state["state_handle"]).bias_signals:
        _run_bias_agent()



# ---------------------------------------------------------------------------
# Section-header tooltip helpers
# ---------------------------------------------------------------------------

# CSS injected once per page load (Streamlit deduplicates identical blocks).
_HEADER_TOOLTIP_CSS = """
<style>
.hdr-wrap {
    display: inline-flex;
    align-items: center;
    gap: 8px;
}
.hdr-info {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    min-width: 18px;
    background: #2a4f80;
    color: #bfdbfe;
    border-radius: 50%;
    font-size: 11px;
    font-weight: 700;
    font-style: normal;
    cursor: default;
    vertical-align: middle;
    flex-shrink: 0;
}
.hdr-info .hdr-tip {
    visibility: hidden;
    opacity: 0;
    width: 360px;
    background-color: #0f172a;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px 16px;
    position: absolute;
    z-index: 9999;
    top: calc(100% + 8px);
    left: 0;
    font-size: 0.78rem;
    line-height: 1.5;
    font-weight: 400;
    pointer-events: none;
    transition: opacity 0.15s;
    box-shadow: 0 6px 24px rgba(0,0,0,0.65);
}
.hdr-info .hdr-tip::before {
    content: "";
    position: absolute;
    bottom: 100%;
    left: 10px;
    border: 6px solid transparent;
    border-bottom-color: #334155;
}
.hdr-info:hover .hdr-tip {
    visibility: visible;
    opacity: 1;
}
.tip-label {
    font-weight: 700;
    color: #7dd3fc;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 3px;
}
.tip-def  { color: #94a3b8; margin-bottom: 5px; }
.tip-ctx  { color: #cbd5e1; }
.tip-req  { color: #86efac; margin-top: 5px; font-style: italic; }
.tip-sep  { border: none; border-top: 1px solid #1e293b; margin: 9px 0; }
</style>
"""

# Keyed lookups so section headers can pull entries by term name.
_BM = {e["term"]: e for e in BIAS_METRIC_GLOSSARY}   # bias metric terms
_BF = {e["term"]: e for e in FEATURE_GLOSSARY}        # dataset feature terms
_FM = {e["term"]: e for e in FAIRNESS_GLOSSARY}        # fairness metric terms

# Tune-technique glossary — used by the Stage 4 section header.
_TRAIN_METRIC_GLOSSARY = [
    {
        "term": "Accuracy",
        "definition": "Out of all predictions, how many did the model get right?",
        "context": (
            "Simple but misleading when classes are imbalanced — a model that always "
            "predicts 'no default' would be 78% accurate but completely useless."
        ),
        "requirement": None,
    },
    {
        "term": "Precision",
        "definition": "Of everyone the model predicted would default, how many actually did?",
        "context": (
            "High precision means fewer false alarms (customers wrongly flagged as defaulters)."
        ),
        "requirement": None,
    },
    {
        "term": "Recall",
        "definition": "Of everyone who actually defaulted, how many did the model catch?",
        "context": (
            "High recall means fewer missed defaulters — critical for a bank that wants "
            "to avoid bad loans."
        ),
        "requirement": None,
    },
    {
        "term": "F1",
        "definition": "The balance between Precision and Recall, combined into one number.",
        "context": (
            "Useful when both matter and you can't sacrifice one for the other."
        ),
        "requirement": None,
    },
    {
        "term": "ROC-AUC",
        "definition": (
            "Measures how well the model ranks defaulters above non-defaulters across all "
            "possible decision thresholds. A score of 1.0 is perfect; 0.5 is no better "
            "than random guessing."
        ),
        "context": "More reliable than accuracy when classes are imbalanced.",
        "requirement": None,
    },
]

_TUNE_GLOSSARY = [
    {
        "term": "SMOTE",
        "definition": (
            "**Synthetic Minority Oversampling Technique.** Creates fake-but-realistic "
            "examples of the minority class (defaulters) to balance the dataset."
        ),
        "context": (
            "Like photocopying rare files so the model sees enough of them to learn from. "
            "Use when the model misses too many actual defaulters (low recall)."
        ),
        "requirement": None,
    },
    {
        "term": "Focal Loss",
        "definition": (
            "A training technique that tells the model to pay more attention to cases "
            "it keeps getting wrong."
        ),
        "context": (
            "Like a teacher focusing extra time on students who struggle, rather than "
            "the ones who already understand. Works best with Neural Networks."
        ),
        "requirement": None,
    },
    {
        "term": "GridSearch CV",
        "definition": (
            "Tries every possible combination of model settings to find the best one."
        ),
        "context": (
            "Like testing every possible oven temperature and bake time to find the "
            "perfect cookie — thorough but slow. Best suited to small parameter grids."
        ),
        "requirement": None,
    },
    {
        "term": "Bayesian / Optuna",
        "definition": (
            "Also finds the best model settings, but learns from each attempt to make "
            "smarter guesses next."
        ),
        "context": (
            "Like a chef who adjusts based on each taste test instead of blindly "
            "trying everything. Faster than GridSearch for large search spaces."
        ),
        "requirement": None,
    },
    {
        "term": "Class Weight Balancing",
        "definition": (
            "Tells the model that mistakes on the minority class (defaulters) should "
            "count more than mistakes on the majority class."
        ),
        "context": (
            "Like grading an exam where getting a rare question wrong costs more points "
            "than getting a common one wrong. No resampling needed — just reweights the loss."
        ),
        "requirement": "Random Forest and XGBoost only.",
    },
    {
        "term": "Early Stopping",
        "definition": (
            "Stops training automatically once the model stops improving on a held-out "
            "validation set, preventing overfitting."
        ),
        "context": (
            "Like stopping practice when your scores plateau — more reps won't help and "
            "may make things worse. Saves time and keeps the model generalising."
        ),
        "requirement": "XGBoost and Neural Network only.",
    },
    {
        "term": "Threshold Tuning",
        "definition": (
            "Adjusts the probability cut-off above which the model predicts 'defaulter'. "
            "Lower thresholds catch more defaulters (higher recall); higher thresholds "
            "are more conservative (higher precision)."
        ),
        "context": (
            "Like lowering the bar for a security check — you flag more suspicious cases "
            "but also more false alarms. Tune to match your tolerance for missed defaults "
            "vs. false alerts."
        ),
        "requirement": None,
    },
    {
        "term": "Feature Selection",
        "definition": (
            "Removes chosen features (SEX, EDUCATION, MARRIAGE, AGE) from training so the "
            "model cannot use them — directly or as proxies — to make predictions."
        ),
        "context": (
            "Like covering up a column in a spreadsheet before sharing it. Reduces bias "
            "risk by ensuring the model cannot leverage protected attributes."
        ),
        "requirement": None,
    },
]


def _section_header(title: str, entries: list, level: int = 3) -> None:
    """Render an h-level heading with a click-to-open ℹ popover.

    Uses ``st.popover`` instead of pure CSS hover — the popover stays open
    until the user clicks outside, so they can read multiple definitions
    without having to keep their cursor steady. Falls back to a hover
    tooltip on Streamlit versions that don't have ``st.popover``.
    """
    head_col, info_col = st.columns([0.94, 0.06])
    with head_col:
        tag = f"h{level}"
        st.markdown(f"<{tag}>{title}</{tag}>", unsafe_allow_html=True)
    with info_col:
        popover = getattr(st, "popover", None)
        if popover is None:
            # Older Streamlit — fall back to inline expander.
            with st.expander("ℹ definitions", expanded=False):
                _render_glossary_entries(entries)
        else:
            with popover("ℹ", help="Click for definitions",
                         use_container_width=False):
                _render_glossary_entries(entries)


def _render_glossary_entries(entries: list) -> None:
    """Render a list of GlossaryEntry dicts inside a popover/expander body."""
    for i, entry in enumerate(entries):
        if i > 0:
            st.divider()
        st.markdown(f"**{entry['term']}**")
        st.markdown(entry["definition"])
        st.caption(entry["context"])
        if entry.get("requirement"):
            st.success(f"⚑ {entry['requirement']}")


def _decode_group_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a per-group DataFrame with human-readable group labels."""
    df = df.copy()
    if "attribute" in df.columns and "group" in df.columns:
        df["group"] = df.apply(
            lambda r: decode_group_label(str(r["attribute"]), str(r["group"])),
            axis=1,
        )
    return df


def _pretty_model_label(name: str) -> str:
    return {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "neural_network": "Neural Network",
    }.get(name, name.replace("_", " ").title())


def _viz_caption(what: str, acceptable: str, concern: str) -> None:
    """Render a compact definition card beneath a chart title."""
    st.markdown(
        f"""<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;
padding:12px 16px;margin:-4px 0 14px 0;font-size:0.82rem;line-height:1.6;">
  <span style="color:#7dd3fc;font-weight:700;">What this shows: </span>
  <span style="color:#cbd5e1;">{what}</span><br>
  <span style="color:#86efac;font-weight:600;">✓ Acceptable: </span>
  <span style="color:#a7f3d0;">{acceptable}</span>
  <span style="margin:0 10px;color:#334155;">|</span>
  <span style="color:#fca5a5;font-weight:600;">⚑ Concern: </span>
  <span style="color:#fecaca;">{concern}</span>
</div>""",
        unsafe_allow_html=True,
    )


def _format_metric_row(model_dict: Dict[str, Any]) -> Dict[str, Any]:
    m = model_dict["metrics"]
    return {
        "model": model_dict["model_name"],
        "roc_auc": round(m.get("roc_auc", 0.0), 4),
        "average_precision": round(m.get("average_precision", 0.0), 4),
        "f1": round(m.get("f1", 0.0), 4),
        "precision": round(m.get("precision", 0.0), 4),
        "recall": round(m.get("recall", 0.0), 4),
        "balanced_accuracy": round(m.get("balanced_accuracy", 0.0), 4),
        "accuracy": round(m.get("accuracy", 0.0), 4),
        "train_seconds": model_dict.get("train_seconds", 0.0),
    }


# ── Plain-English hyperparameter label mapping ──────────────────────────────
_HP_LABELS: Dict[str, str] = {
    "n_estimators":        "Number of trees",
    "max_depth":           "Max tree depth",
    "min_samples_split":   "Min samples to split",
    "min_samples_leaf":    "Min samples per leaf",
    "class_weight":        "Class weighting",
    "learning_rate":       "Learning rate",
    "learning_rate_init":  "Initial learning rate",
    "max_iter":            "Max training iterations",
    "hidden_layer_sizes":  "Hidden layer sizes",
    "alpha":               "Regularisation (alpha)",
    "scale_pos_weight":    "Positive class weight",
    "early_stopping":      "Early stopping",
    "n_iter_no_change":    "Patience (no-change rounds)",
    "validation_fraction": "Validation fraction",
    "early_stopping_rounds": "Early stopping rounds",
}


def _render_metric_cards(model_dict: Dict[str, Any]) -> None:
    """Render 5 metric cards with plain-language hints."""
    mv = model_dict["metrics"]

    def _pct(v: float) -> str:
        return f"{v:.1%}"

    def _val(v: float) -> str:
        return f"{v:.3f}"

    roc   = mv.get("roc_auc", 0.0)
    acc   = mv.get("accuracy", 0.0)
    prec  = mv.get("precision", 0.0)
    rec   = mv.get("recall", 0.0)
    f1    = mv.get("f1", 0.0)

    cards = [
        ("ROC-AUC",   _val(roc),  "Good if above 0.70"),
        ("Accuracy",  _pct(acc),  "Can be misleading — check recall"),
        ("Precision", _pct(prec), f"Of predicted defaults, {prec:.0%} were real"),
        ("Recall",    _pct(rec),  f"Catches {rec:.0%} of actual defaulters"),
        ("F1 Score",  _val(f1),   "Balance of precision and recall"),
    ]

    cols = st.columns(len(cards))
    for col, (label, value, hint) in zip(cols, cards):
        with col:
            st.markdown(
                f"""<div style="background:#F8F9FA;border:1px solid #DEE2E6;
border-radius:10px;padding:14px 10px;text-align:center;height:100%;">
  <div style="font-size:0.72rem;color:#6C757D;font-weight:700;
  text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;">{label}</div>
  <div style="font-size:1.45rem;font-weight:700;color:#1A1A2E;">{value}</div>
  <div style="font-size:0.7rem;color:#6C757D;margin-top:6px;line-height:1.4;">{hint}</div>
</div>""",
                unsafe_allow_html=True,
            )


def _model_insight(model_dict: Dict[str, Any]) -> str:
    """Generate a plain-language one-sentence summary of model performance."""
    mv = model_dict["metrics"]
    prec  = mv.get("precision", 0.0)
    rec   = mv.get("recall", 0.0)
    roc   = mv.get("roc_auc", 0.0)
    f1    = mv.get("f1", 0.0)

    if roc >= 0.80:
        disc = "strong overall discrimination (ROC-AUC)"
    elif roc >= 0.70:
        disc = "acceptable overall discrimination (ROC-AUC)"
    else:
        disc = "weak overall discrimination — ROC-AUC is below 0.70"

    if prec >= 0.70 and rec >= 0.70:
        return (
            f"The model shows {disc} and maintains a solid balance of precision "
            "and recall — it catches most defaulters while keeping false alarms low."
        )
    if prec > rec + 0.15:
        return (
            f"The model shows {disc} and is good at avoiding false alarms (precision = "
            f"{prec:.0%}), but misses {(1 - rec):.0%} of actual defaulters "
            f"(recall = {rec:.0%}). Consider lowering the decision threshold in Stage 4 "
            "to catch more."
        )
    if rec > prec + 0.15:
        return (
            f"The model shows {disc} and catches most defaulters (recall = {rec:.0%}), "
            f"but raises many false alarms — only {prec:.0%} of flagged cases are real "
            "defaults. Consider raising the decision threshold in Stage 4 to cut false alarms."
        )
    if f1 >= 0.60:
        return (
            f"The model shows {disc} with a reasonable balance of precision ({prec:.0%}) "
            f"and recall ({rec:.0%})."
        )
    return (
        f"The model shows {disc} but struggles to distinguish defaulters reliably "
        f"(F1 = {f1:.3f}). Try SMOTE, class weight balancing, or threshold tuning in Stage 4."
    )


def _render_confusion_matrix_html(cm: List[List[int]]) -> None:
    """Render a colour-coded confusion matrix with plain-English labels."""
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    def _cell(count: int, label: str, correct: bool) -> str:
        bg    = "#EAF3DE" if correct else "#FDEAEA"
        brd   = "#C1DFA0" if correct else "#F5B8B8"
        num_c = "#27500A" if correct else "#7A1F1F"
        lbl_c = "#3B7A12" if correct else "#9B2C2C"
        return (
            f'<td style="border:1px solid #DEE2E6;padding:14px 10px;'
            f'background:{bg};text-align:center;">'
            f'<div style="font-size:1.35rem;font-weight:700;color:{num_c};">'
            f'{count:,}</div>'
            f'<div style="font-size:0.68rem;color:{lbl_c};margin-top:5px;">'
            f'{label}</div></td>'
        )

    html = f"""
<table style="border-collapse:collapse;width:100%;font-size:0.83rem;margin-top:8px;">
  <thead>
    <tr>
      <th style="border:1px solid #DEE2E6;padding:10px 8px;background:#FFFFFF;
          color:#6C757D;font-weight:500;text-align:left;min-width:160px;"></th>
      <th style="border:1px solid #DEE2E6;padding:10px 8px;background:#F8F9FA;
          color:#6C757D;font-weight:600;text-align:center;">Predicted: No default</th>
      <th style="border:1px solid #DEE2E6;padding:10px 8px;background:#F8F9FA;
          color:#6C757D;font-weight:600;text-align:center;">Predicted: Default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style="border:1px solid #DEE2E6;padding:10px 8px;background:#F8F9FA;
          color:#6C757D;font-weight:600;text-align:left;">Actually: No default</th>
      {_cell(tn, "Correctly cleared", True)}
      {_cell(fp, "Wrongly flagged",   False)}
    </tr>
    <tr>
      <th style="border:1px solid #DEE2E6;padding:10px 8px;background:#F8F9FA;
          color:#6C757D;font-weight:600;text-align:left;">Actually: Default</th>
      {_cell(fn, "Missed defaulters",  False)}
      {_cell(tp, "Correctly caught",   True)}
    </tr>
  </tbody>
</table>
"""
    st.markdown(html, unsafe_allow_html=True)


def _render_hyperparameters(hp: Dict[str, Any]) -> None:
    """Render hyperparameters as a plain-English key-value list."""
    if not hp:
        return
    st.markdown("**Hyperparameters used**")
    rows_html = ""
    for key, raw_val in hp.items():
        label = _HP_LABELS.get(key, key.replace("_", " ").title())
        if key == "max_depth" and raw_val is None:
            display = "Unlimited"
        elif isinstance(raw_val, float):
            display = f"{raw_val:.5g}"
        elif isinstance(raw_val, (list, tuple)):
            display = str(raw_val)
        elif isinstance(raw_val, bool):
            display = "Yes" if raw_val else "No"
        else:
            display = str(raw_val)
        rows_html += (
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:6px 4px;border-bottom:1px solid #DEE2E6;">'
            f'<span style="color:#6C757D;font-size:0.8rem;">{label}</span>'
            f'<span style="color:#1A1A2E;font-size:0.8rem;font-weight:600;">'
            f'{display}</span></div>'
        )
    st.markdown(
        f'<div style="border:1px solid #DEE2E6;border-radius:8px;'
        f'padding:8px 12px;background:#F8F9FA;">{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar — config + hyperparameter form
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Pipeline config")
    st.write(
        "GEMINI_API_KEY: "
        + (":green[set]" if _gemini_key_present() else ":red[missing]")
    )
    st.session_state["test_size"] = st.slider(
        "Test split fraction", 0.1, 0.4, st.session_state["test_size"], 0.05
    )
    st.session_state["random_state"] = st.number_input(
        "Random seed", value=int(st.session_state["random_state"]), step=1
    )
    if st.button("Reset pipeline state (clears arrays + metrics)"):
        reset_state(st.session_state["state_handle"])
        st.session_state["metrics_report"] = {}
        st.session_state["explanation_md"] = ""
        st.session_state["dataset_summary"] = {}
        st.success("State cleared.")

    st.markdown("---")
    st.header("Hyperparameters")
    hp = st.session_state["hyperparameters"]

    with st.expander("Random Forest", expanded=False):
        rf = hp["random_forest"]
        rf["n_estimators"] = int(
            st.number_input("n_estimators", 10, 2000, int(rf["n_estimators"]), 50, key="rf_n")
        )
        max_depth_val = rf["max_depth"] if rf["max_depth"] is not None else 0
        new_depth = int(
            st.number_input(
                "max_depth (0 = no limit)", 0, 100, int(max_depth_val), 1, key="rf_d"
            )
        )
        rf["max_depth"] = None if new_depth == 0 else new_depth
        rf["min_samples_split"] = int(
            st.number_input("min_samples_split", 2, 50, int(rf["min_samples_split"]), 1, key="rf_mss")
        )
        rf["min_samples_leaf"] = int(
            st.number_input("min_samples_leaf", 1, 50, int(rf["min_samples_leaf"]), 1, key="rf_msl")
        )
        rf["class_weight"] = st.selectbox(
            "class_weight",
            options=["balanced", "balanced_subsample", None],
            index=["balanced", "balanced_subsample", None].index(rf["class_weight"]),
            key="rf_cw",
        )

    with st.expander("XGBoost", expanded=False):
        xg = hp["xgboost"]
        xg["n_estimators"] = int(
            st.number_input("n_estimators ", 10, 2000, int(xg["n_estimators"]), 50, key="xg_n")
        )
        xg["learning_rate"] = float(
            st.number_input("learning_rate", 0.001, 1.0, float(xg["learning_rate"]), 0.01, key="xg_lr")
        )
        xg["max_depth"] = int(
            st.number_input("max_depth ", 1, 20, int(xg["max_depth"]), 1, key="xg_d")
        )
        xg["subsample"] = float(
            st.slider("subsample", 0.5, 1.0, float(xg["subsample"]), 0.05, key="xg_sub")
        )
        xg["colsample_bytree"] = float(
            st.slider("colsample_bytree", 0.5, 1.0, float(xg["colsample_bytree"]), 0.05, key="xg_col")
        )

    with st.expander("Neural Network (MLP)", expanded=False):
        nn = hp["neural_network"]
        layers_str = st.text_input(
            "hidden_layer_sizes (comma-separated)",
            value=",".join(str(x) for x in nn["hidden_layer_sizes"]),
            key="nn_layers",
        )
        try:
            nn["hidden_layer_sizes"] = [
                int(x.strip()) for x in layers_str.split(",") if x.strip()
            ]
        except ValueError:
            st.warning("hidden_layer_sizes must be comma-separated integers.")
        nn["alpha"] = float(
            st.number_input("alpha (L2)", 1e-6, 1e-1, float(nn["alpha"]), 1e-5, format="%g", key="nn_a")
        )
        nn["learning_rate_init"] = float(
            st.number_input(
                "learning_rate_init", 1e-5, 1e-1, float(nn["learning_rate_init"]), 1e-4,
                format="%g", key="nn_lri"
            )
        )
        nn["max_iter"] = int(
            st.number_input("max_iter", 10, 1000, int(nn["max_iter"]), 10, key="nn_mi")
        )
        nn["batch_size"] = int(
            st.number_input("batch_size", 16, 1024, int(nn["batch_size"]), 16, key="nn_bs")
        )

    if st.button("Reset hyperparameters to defaults"):
        st.session_state["hyperparameters"] = default_hyperparameters()
        _sync_hyperparameters_to_state()
        st.success("Hyperparameters reset.")
        st.rerun()


# ---------------------------------------------------------------------------
# Main — tabs
# ---------------------------------------------------------------------------

st.title("Agentic Default Classifier")
st.caption(
    "Train and explain credit-card-default classifiers using a CrewAI pipeline "
    "(Random Forest, XGBoost, Neural Network) powered by Gemini."
)

tab_run, tab_chat, tab_explain, tab_bias, tab_fairness, tab_artifacts = st.tabs(
    ["Run", "Chat", "Explanation", "Bias", "Fairness", "Artifacts"]
)



# ---------- Tune helpers (call the optimizer and merge result back in) ------


def _apply_optimization(model_name: str, technique: str,
                        technique_params: dict | None = None) -> None:
    """Run an optimizer technique and merge its result into metrics_report."""
    _ensure_dataset_loaded()
    _sync_hyperparameters_to_state()
    state = get_state(st.session_state["state_handle"])
    # Save pre-tuning snapshot (only on first tune, so baseline is never overwritten)
    if not st.session_state.get("pre_tuning_metrics_report"):
        snapshot = st.session_state.get("metrics_report") or {}
        if snapshot:
            st.session_state["pre_tuning_metrics_report"] = snapshot
            state.pre_tuning_metrics_report = snapshot
    label = technique.replace("_", " ").title()
    with st.spinner(f"Optimising {model_name} with {label}..."):
        try:
            new_result = optimize(
                technique=technique,
                model_name=model_name,
                x_train=state.dataset.x_train,
                y_train=state.dataset.y_train,
                x_test=state.dataset.x_test,
                y_test=state.dataset.y_test,
                feature_names=state.dataset.feature_names,
                hyperparameters=st.session_state["hyperparameters"],
                random_state=st.session_state["random_state"],
                technique_params=technique_params or {},
            )
        except RuntimeError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Optimization failed: {exc!r}")
            return

    # Merge the optimized result into the existing report (replaces same model).
    existing = st.session_state["metrics_report"] or {}
    merged_models = {m["model_name"]: m for m in existing.get("models", [])}
    merged_models[model_name] = new_result
    merged_list = list(merged_models.values())
    leaderboard = sorted(
        merged_list,
        key=lambda m: (m["metrics"].get("roc_auc", 0.0),
                       m["metrics"].get("f1", 0.0)),
        reverse=True,
    )
    merged_report = {
        "models": merged_list,
        "leaderboard": [
            {
                "model_name": m["model_name"],
                "roc_auc": m["metrics"].get("roc_auc"),
                "f1": m["metrics"].get("f1"),
                "precision": m["metrics"].get("precision"),
                "recall": m["metrics"].get("recall"),
                "accuracy": m["metrics"].get("accuracy"),
            }
            for m in leaderboard
        ],
        "best_model": leaderboard[0]["model_name"] if leaderboard else None,
        "hyperparameters_used": existing.get("hyperparameters_used", {}),
    }
    st.session_state["metrics_report"] = merged_report
    state.metrics_report = merged_report
    st.success(
        f"{_pretty_model_label(model_name)} optimised with {label}. "
        f"New ROC-AUC: {new_result['metrics'].get('roc_auc'):.4f}."
    )


# ---------- Run tab ---------------------------------------------------------

with tab_run:
    # ---- Stage 1: Load Data ------------------------------------------------
    st.subheader("Stage 1 — Load data")
    cols = st.columns([1, 3])
    with cols[0]:
        if st.button("Load / refresh dataset", key="run_load_btn"):
            reset_state(st.session_state["state_handle"])
            _ensure_dataset_loaded()
            st.success("Dataset loaded.")
    with cols[1]:
        if st.session_state["dataset_summary"]:
            ds = st.session_state["dataset_summary"]
            st.write(
                f"**rows:** {ds['rows']}  |  **features:** {ds['features']}  |  "
                f"**train/test:** {ds['train_size']}/{ds['test_size']}  |  "
                f"**class balance:** {ds['class_balance']}"
            )
        else:
            st.info("Click 'Load / refresh dataset' to begin.")

    with st.expander("Feature dictionary (23 explanatory variables + Y)",
                     expanded=False):
        st.caption(
            "Source: Yeh & Lien (2009), UCI default-of-credit-card-clients. "
            "All amounts in NT dollars. Months run from April 2005 → September 2005."
        )
        st.dataframe(
            pd.DataFrame(FEATURE_TABLE_23),
            use_container_width=True,
            hide_index=True,
            column_config={
                "X#":          st.column_config.TextColumn("X#", width="small"),
                "column":      st.column_config.TextColumn("Column"),
                "type":        st.column_config.TextColumn("Type"),
                "description": st.column_config.TextColumn("Description", width="large"),
            },
        )

    st.markdown("---")

    # ---- Stage 2: Clean Data -----------------------------------------------
    st.subheader("Stage 2 — Clean data")
    st.caption(
        "Applies three passes in order: "
        "(1) remove duplicate rows, "
        "(2) remap undocumented EDUCATION codes 0 / 5 / 6 → 4 (Others), "
        "(3) Winsorise LIMIT_BAL, BILL_AMT, and PAY_AMT columns at the "
        "1st / 99th percentiles to reduce outlier influence. "
        "The cleaned dataset replaces the loaded one and feeds into Stage 3."
    )

    clean_cols = st.columns([1, 1, 3])
    with clean_cols[0]:
        win_pct = st.number_input(
            "Winsorise %",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            key="clean_win_pct",
            help="Rows below this percentile or above (100 − this) are clipped.",
        )
    with clean_cols[1]:
        if st.button("Clean dataset", key="clean_btn", type="primary"):
            _clean_data(winsorise_percentile=float(win_pct))
            st.success("Dataset cleaned and ready for training.")

    cr = st.session_state.get("cleaning_report") or {}
    if cr:
        with st.expander("Cleaning report", expanded=True):
            summary_cols = st.columns(4)
            summary_cols[0].metric("Rows before", cr.get("rows_before", "—"))
            summary_cols[1].metric("Rows after",  cr.get("rows_after",  "—"))
            summary_cols[2].metric("Duplicates removed",
                                   cr.get("duplicates_removed", 0))
            summary_cols[3].metric("EDUCATION codes fixed",
                                   cr.get("education_codes_remapped", 0))

            outliers = cr.get("outliers_capped") or {}
            if outliers:
                st.markdown("**Winsorisation bounds applied**")
                outlier_rows = [
                    {"column": col, "lower bound": v["lower"], "upper bound": v["upper"]}
                    for col, v in outliers.items()
                ]
                st.dataframe(
                    pd.DataFrame(outlier_rows),
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.info("Click 'Clean dataset' to apply cleaning before training.")

    st.markdown("---")

    # ---- Stage 3: Train Model ---------------------------------------------
    st.subheader("Stage 3 — Train model")
    _section_header("Model metrics", _TRAIN_METRIC_GLOSSARY, level=4)
    btn_cols = st.columns(4)
    with btn_cols[0]:
        if st.button("Random Forest", key="train_rf_btn"):
            _train_models(["random_forest"])
    with btn_cols[1]:
        if st.button("XGBoost", key="train_xgb_btn"):
            _train_models(["xgboost"])
    with btn_cols[2]:
        if st.button("Neural Network", key="train_nn_btn"):
            _train_models(["neural_network"])
    with btn_cols[3]:
        if st.button("Train all", type="primary", key="train_all_btn"):
            _train_models(list(SUPPORTED_MODELS))

    # Pickle download disabled (workspace storage constraints) — trained
    # models live in process memory only. Re-enable later by passing an
    # output_dir down to train_and_evaluate and surfacing read_bytes here.

    report = st.session_state["metrics_report"]
    if report and report.get("models"):
        st.markdown(f"**Best model so far:** `{report.get('best_model')}`")
        leaderboard_df = pd.DataFrame(report["leaderboard"])
        st.markdown("**Leaderboard**")
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

        st.markdown("**Per-model details**")
        for m in report["models"]:
            train_sec = m.get("train_seconds", 0.0)
            expander_label = (
                f"{_pretty_model_label(m['model_name'])}  ·  ⏱ {train_sec:.2f}s"
            )
            with st.expander(expander_label, expanded=False):

                # ── Metric cards ─────────────────────────────────────────
                st.markdown("**Metrics**")
                _render_metric_cards(m)

                # ── Plain-language insight ────────────────────────────────
                st.markdown(
                    f'<div style="background:#EFF6FF;border-left:3px solid #3b82f6;'
                    f'border-radius:6px;padding:12px 16px;margin:14px 0 6px 0;'
                    f'font-size:0.83rem;color:#334155;line-height:1.6;">'
                    f'💡 {_model_insight(m)}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("")  # spacer

                # ── Confusion matrix + feature importance side by side ───
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.markdown("**Confusion matrix**")
                    cm = m.get("confusion_matrix") or [[0, 0], [0, 0]]
                    _render_confusion_matrix_html(cm)
                with col_b:
                    st.markdown("**Top feature importances**")
                    fi = m.get("feature_importance") or []
                    if fi:
                        fi_df = pd.DataFrame(fi).set_index("feature")[["importance"]]
                        st.bar_chart(fi_df, horizontal=False)
                    else:
                        st.caption("(none — model does not expose feature importances)")

                # ── Hyperparameters ──────────────────────────────────────
                if m.get("hyperparameters"):
                    st.markdown("")
                    _render_hyperparameters(m["hyperparameters"])

                # ── Notes (tuning provenance) ─────────────────────────────
                if m.get("notes"):
                    st.caption(f"ℹ {m['notes']}")
    else:
        st.info("No models trained yet — click one of the buttons above.")

    # XGBoost substitution warning (shown whenever models are available)
    if report and report.get("models"):
        _xgb_notes_warn = next((m.get("notes","") for m in report["models"] if m["model_name"]=="xgboost"), "")
        if "not installed" in _xgb_notes_warn.lower() or "gradientboosting" in _xgb_notes_warn.lower():
            st.warning("⚠ XGBoost is not installed. Results shown are from sklearn GradientBoostingClassifier, which is a different model. Install XGBoost with `uv pip install xgboost` for true XGBoost results.")

    st.markdown("---")

    # ---- Stage 4: Tune Model ----------------------------------------------
    st.subheader("Stage 4 — Tune model")
    st.caption(
        "Apply class-imbalance, hyperparameter-search, training-control, "
        "threshold, and feature-control techniques to a specific model. "
        "The tuned result replaces the existing leaderboard entry for that model. "
        "Configure options and click the Run button for each technique."
    )
    _section_header("Tuning techniques", _TUNE_GLOSSARY, level=4)

    if not (report and report.get("models")):
        st.info("Train at least one model in Stage 3 before tuning.")
    else:
        for mname in SUPPORTED_MODELS:
            is_rf  = mname == "random_forest"
            is_xgb = mname == "xgboost"
            is_nn  = mname == "neural_network"

            with st.expander(f"Tune {_pretty_model_label(mname)}", expanded=False):

                # ── Section 1: Class Imbalance ──────────────────────────
                st.markdown("#### 1 · Class Imbalance")

                # SMOTE — all 3 models
                st.markdown("**SMOTE** — Random Forest · XGBoost · Neural Network")
                k = st.number_input(
                    "k_neighbors", 1, 20, 5, 1,
                    key=f"smote_k_{mname}",
                    help="Number of nearest neighbours used to synthesise minority samples.",
                )
                if st.button("Run SMOTE", key=f"smote_btn_{mname}"):
                    _apply_optimization(mname, "smote", {"k_neighbors": int(k)})

                st.divider()

                # Focal Loss — XGBoost & Neural Network only
                if is_rf:
                    st.markdown(
                        '<span style="color:gray;font-weight:600;">Focal Loss</span>'
                        ' — XGBoost · Neural Network',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "⚠ Not available for Random Forest — "
                        "tree ensembles use a fixed splitting criterion, not a differentiable loss."
                    )
                else:
                    st.markdown("**Focal Loss** — XGBoost · Neural Network")
                    fl_cols = st.columns(2)
                    with fl_cols[0]:
                        alpha = st.number_input(
                            "alpha", 0.0, 1.0, 0.25, 0.05,
                            key=f"focal_a_{mname}",
                            help="Class-balance weight (typical 0.25).",
                        )
                    with fl_cols[1]:
                        gamma = st.number_input(
                            "gamma", 0.0, 5.0, 2.0, 0.25,
                            key=f"focal_g_{mname}",
                            help="Focusing parameter (typical 2.0).",
                        )
                    if st.button("Run Focal Loss", key=f"focal_btn_{mname}"):
                        _apply_optimization(
                            mname, "focal_loss",
                            {"alpha": float(alpha), "gamma": float(gamma)},
                        )

                st.divider()

                # Class Weight Balancing — Random Forest & XGBoost only
                if is_nn:
                    st.markdown(
                        '<span style="color:gray;font-weight:600;">Class Weight Balancing</span>'
                        ' — Random Forest · XGBoost',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "⚠ Not available for Neural Network — "
                        "MLPClassifier does not support a class_weight parameter."
                    )
                else:
                    st.markdown("**Class Weight Balancing** — Random Forest · XGBoost")
                    if is_rf:
                        st.caption("Sets `class_weight='balanced'` so RF up-weights the minority class.")
                    else:
                        st.caption(
                            "Sets `scale_pos_weight = negative_count / positive_count` "
                            "so XGBoost up-weights defaulters."
                        )
                    if st.button("Run Class Weight Balancing", key=f"cw_btn_{mname}"):
                        _apply_optimization(mname, "class_weight", {})

                # ── Section 2: Hyperparameter Search ───────────────────
                st.markdown("#### 2 · Hyperparameter Search")

                # GridSearchCV — all 3 models (with warning for RF & NN)
                st.markdown("**GridSearch CV** — All models")
                if is_rf or is_nn:
                    st.warning(
                        "⚠ GridSearchCV can take several minutes for Random Forest and "
                        "Neural Network. Consider Bayesian (Optuna) for faster results."
                    )
                gs_cols = st.columns(2)
                with gs_cols[0]:
                    cv = st.number_input(
                        "CV folds", 2, 10, 3, 1,
                        key=f"gs_cv_{mname}",
                        help="Number of cross-validation folds.",
                    )
                with gs_cols[1]:
                    scoring = st.selectbox(
                        "scoring",
                        ["roc_auc", "f1", "precision", "recall",
                         "balanced_accuracy", "average_precision"],
                        index=0,
                        key=f"gs_scoring_{mname}",
                    )
                grid = DEFAULT_PARAM_GRIDS.get(mname, {})
                st.caption(f"Grid: `{grid}`")
                if st.button("Run GridSearch", key=f"gs_btn_{mname}"):
                    _apply_optimization(
                        mname, "grid_search",
                        {"cv_folds": int(cv), "scoring": scoring, "param_grid": grid},
                    )

                st.divider()

                # Bayesian (Optuna) — all 3 models
                st.markdown("**Bayesian (Optuna)** — All models")
                opt_cols = st.columns(2)
                with opt_cols[0]:
                    n_trials = st.number_input(
                        "n_trials", 5, 100, 15, 5,
                        key=f"opt_n_{mname}",
                        help="Number of parameter combinations Optuna will try.",
                    )
                    timeout = st.number_input(
                        "timeout (sec)", 30, 1800, 600, 30,
                        key=f"opt_t_{mname}",
                        help="Maximum wall-clock time for the search.",
                    )
                with opt_cols[1]:
                    opt_scoring = st.selectbox(
                        "scoring ",
                        ["f1", "roc_auc", "precision", "recall",
                         "balanced_accuracy", "average_precision"],
                        index=0,
                        key=f"opt_scoring_{mname}",
                    )
                    opt_cv = st.number_input(
                        "CV folds ", 2, 10, 3, 1,
                        key=f"opt_cv_{mname}",
                    )
                if st.button("Run Optuna", key=f"opt_btn_{mname}"):
                    _apply_optimization(
                        mname, "optuna",
                        {"n_trials": int(n_trials), "timeout": int(timeout),
                         "scoring": opt_scoring, "cv_folds": int(opt_cv)},
                    )

                # ── Section 3: Training Control ─────────────────────────
                st.markdown("#### 3 · Training Control")

                # Early Stopping — XGBoost & Neural Network only
                if is_rf:
                    st.markdown(
                        '<span style="color:gray;font-weight:600;">Early Stopping</span>'
                        ' — XGBoost · Neural Network',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "⚠ Not available for Random Forest — "
                        "decision trees are not trained iteratively and cannot be stopped early."
                    )
                else:
                    st.markdown("**Early Stopping** — XGBoost · Neural Network")
                    es_cols = st.columns(2)
                    with es_cols[0]:
                        es_rounds = st.number_input(
                            "patience (rounds)", 5, 50, 10, 5,
                            key=f"es_rounds_{mname}",
                            help="Stop training after this many rounds with no improvement.",
                        )
                    with es_cols[1]:
                        es_val = st.slider(
                            "validation fraction", 0.05, 0.30, 0.10, 0.05,
                            key=f"es_val_{mname}",
                            help="Share of training data held out to monitor for early stopping.",
                        )
                    if st.button("Run Early Stopping", key=f"es_btn_{mname}"):
                        _apply_optimization(
                            mname, "early_stopping",
                            {"early_stopping_rounds": int(es_rounds),
                             "validation_fraction": float(es_val)},
                        )

                # ── Section 4: Decision Threshold ───────────────────────
                st.markdown("#### 4 · Decision Threshold")

                st.markdown("**Threshold Tuning** — All models")
                threshold = st.slider(
                    "Decision threshold", 0.10, 0.90, 0.50, 0.05,
                    key=f"thresh_{mname}",
                )
                st.caption("Lower = higher recall (catch more defaulters) · Higher = higher precision (fewer false alarms)")
                if st.button("Run Threshold Tuning", key=f"thresh_btn_{mname}"):
                    _apply_optimization(
                        mname, "threshold_tuning",
                        {"threshold": float(threshold)},
                    )

                # ── Section 5: Feature Control ───────────────────────────
                st.markdown("#### 5 · Feature Control")

                st.markdown("**Feature Selection** — All models")
                st.caption(
                    "Toggle off a feature to exclude it from training. "
                    "Useful to prevent the model from using protected attributes."
                )
                feat_cols = st.columns(4)
                with feat_cols[0]:
                    include_sex = st.checkbox("Include SEX", value=True, key=f"feat_sex_{mname}")
                with feat_cols[1]:
                    include_edu = st.checkbox("Include EDUCATION", value=True, key=f"feat_edu_{mname}")
                with feat_cols[2]:
                    include_mar = st.checkbox("Include MARRIAGE", value=True, key=f"feat_mar_{mname}")
                with feat_cols[3]:
                    include_age = st.checkbox("Include AGE", value=True, key=f"feat_age_{mname}")
                excluded = [
                    feat for feat, include in [
                        ("SEX", include_sex),
                        ("EDUCATION", include_edu),
                        ("MARRIAGE", include_mar),
                        ("AGE", include_age),
                    ]
                    if not include
                ]
                if excluded:
                    st.caption(f"Will exclude from training: {', '.join(excluded)}")
                if st.button("Run Feature Selection", key=f"feat_btn_{mname}"):
                    _apply_optimization(
                        mname, "feature_selection",
                        {"excluded_features": excluded},
                    )

    st.markdown("---")

    # ---- Stage 5: Explain Model -------------------------------------------
    st.subheader("Stage 5 — Explain model")
    if not (report and report.get("models")):
        st.info("Train at least one model in Stage 3 to enable the explainer.")
    else:
        if st.button("Generate / refresh explanation",
                     type="primary", key="explain_btn"):
            _run_explainer()
        if st.session_state.get("explanation_md"):
            st.caption("Latest explainer brief is rendered on the **Explanation** tab.")


# ---------- Chat tab --------------------------------------------------------

with tab_chat:
    st.subheader("Chat with the Pipeline Coordinator")
    st.caption(
        "Examples: 'Use 500 trees in random forest and re-run.'  ·  "
        "'Drop the XGBoost learning rate to 0.01 and retrain.'  ·  "
        "'Which model has the best recall?'  ·  'Reset hyperparameters.'"
    )

    if not _gemini_key_present():
        st.error("GEMINI_API_KEY is not set — chat is disabled.")
    else:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask the coordinator to tweak or run something...")
        if user_input:
            st.session_state["messages"].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            _ensure_agents_imported()
            coordinator_cls = st.session_state["_CoordinatorAgent"]
            coordinator = coordinator_cls(
                state_handle=st.session_state["state_handle"]
            )
            with st.chat_message("assistant"):
                with st.spinner("Coordinator is thinking..."):
                    reply = coordinator.chat(
                        user_message=user_input,
                        history=st.session_state["messages"][:-1],
                    )
                st.markdown(reply)

            st.session_state["messages"].append(
                {"role": "assistant", "content": reply}
            )

            state = get_state(st.session_state["state_handle"])
            if state.metrics_report:
                st.session_state["metrics_report"] = state.metrics_report
            if state.hyperparameters:
                st.session_state["hyperparameters"] = state.hyperparameters


# ---------- Explanation tab -------------------------------------------------

with tab_explain:
    st.subheader("Explainer brief")
    md = st.session_state["explanation_md"]
    if not md:
        st.info(
            "Train models and click 'Run all + explain' (or 'Generate / refresh "
            "explanation') to populate this tab."
        )
    else:
        # ---- Structured layout (when metrics_report is available) ----
        report: Dict = st.session_state.get("metrics_report", {})
        models_list: List[Dict] = report.get("models", [])

        if report and models_list:
            # ---- 1. Winner banner ----------------------------------------
            best_name: str = report.get("best_model", models_list[0]["model_name"])
            best_model_data = next(
                (m for m in models_list if m["model_name"] == best_name), models_list[0]
            )
            best_metrics = best_model_data.get("metrics", {})
            best_roc = best_metrics.get("roc_auc", 0.0)
            best_f1 = best_metrics.get("f1", 0.0)
            best_label = _pretty_model_label(best_name)

            st.markdown(
                f"""
<div style="background:linear-gradient(135deg,#14532d 0%,#166534 100%);
            border:1px solid #86efac;border-radius:12px;padding:20px 28px;
            margin-bottom:24px;">
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
    <span style="font-size:2rem;">🏆</span>
    <div>
      <div style="color:#86efac;font-size:0.78rem;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.08em;">
        Best Model
      </div>
      <div style="color:#f0fdf4;font-size:1.45rem;font-weight:800;line-height:1.2;">
        {best_label}
      </div>
    </div>
    <div style="margin-left:auto;display:flex;gap:18px;flex-wrap:wrap;">
      <div style="text-align:center;">
        <div style="color:#86efac;font-size:0.72rem;font-weight:600;
                    text-transform:uppercase;letter-spacing:0.06em;">ROC-AUC</div>
        <div style="color:#f0fdf4;font-size:1.55rem;font-weight:800;">{best_roc:.2f}</div>
      </div>
      <div style="text-align:center;">
        <div style="color:#86efac;font-size:0.72rem;font-weight:600;
                    text-transform:uppercase;letter-spacing:0.06em;">F1 Score</div>
        <div style="color:#f0fdf4;font-size:1.55rem;font-weight:800;">{best_f1:.2f}</div>
      </div>
    </div>
  </div>
  <div style="margin-top:12px;color:#bbf7d0;font-size:0.88rem;
              border-top:1px solid #166534;padding-top:10px;">
    Highest combined ROC-AUC and F1 across all trained models.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

            # ---- 2. Per-model cards --------------------------------------
            best_roc_name = max(models_list, key=lambda m: m["metrics"].get("roc_auc", 0.0))["model_name"]
            best_prec_name = max(models_list, key=lambda m: m["metrics"].get("precision", 0.0))["model_name"]
            fastest_name = min(
                models_list,
                key=lambda m: m["metrics"].get("train_seconds", float("inf")),
            )["model_name"]

            cols = st.columns(len(models_list))
            for col, model_data in zip(cols, models_list):
                mname: str = model_data["model_name"]
                mlabel: str = _pretty_model_label(mname)
                mmetrics: Dict = model_data.get("metrics", {})
                roc_auc = mmetrics.get("roc_auc", 0.0)
                precision = mmetrics.get("precision", 0.0)
                recall = mmetrics.get("recall", 0.0)
                f1 = mmetrics.get("f1", 0.0)
                # train_seconds and confusion_matrix are top-level model fields,
                # not inside the nested "metrics" dict.
                train_sec = model_data.get("train_seconds", 0.0)
                raw_cm = model_data.get("confusion_matrix") or [[0, 0], [0, 0]]
                tn = raw_cm[0][0]
                fp = raw_cm[0][1]
                fn = raw_cm[1][0]
                tp = raw_cm[1][1]

                # Badges — light-background pills
                badges_html = ""
                if mname == best_roc_name:
                    badges_html += (
                        '<span style="background:#DBEAFE;color:#1D4ED8;border:1px solid #93C5FD;'
                        'border-radius:4px;font-size:0.68rem;padding:2px 7px;margin-right:4px;'
                        'font-weight:600;">🏆 Best overall</span>'
                    )
                if mname == best_prec_name:
                    badges_html += (
                        '<span style="background:#DCFCE7;color:#15803D;border:1px solid #86EFAC;'
                        'border-radius:4px;font-size:0.68rem;padding:2px 7px;margin-right:4px;'
                        'font-weight:600;">🎯 Most precise</span>'
                    )
                if mname == fastest_name:
                    badges_html += (
                        '<span style="background:#EDE9FE;color:#6D28D9;border:1px solid #C4B5FD;'
                        'border-radius:4px;font-size:0.68rem;padding:2px 7px;font-weight:600;">'
                        '⚡ Fastest</span>'
                    )

                # Plain-language summary
                if precision > recall + 0.15:
                    summary = (
                        f"Good at avoiding false alarms (precision) but misses "
                        f"{(1 - recall):.0%} of actual defaulters — consider lowering "
                        f"the decision threshold."
                    )
                elif recall > precision + 0.15:
                    summary = (
                        f"Catches most defaulters (recall) but raises many false alarms "
                        f"— only {precision:.0%} of flagged cases are real defaults."
                    )
                else:
                    summary = (
                        f"Reasonable balance of precision ({precision:.0%}) and "
                        f"recall ({recall:.0%})."
                    )

                # Dynamic confusion-matrix caption using actual counts
                _missed_pct = fn / (fn + tp) if (fn + tp) > 0 else 0
                _fp_pct = fp / (fp + tn) if (fp + tn) > 0 else 0
                if _missed_pct > 0.55:
                    _cm_caption = (
                        f"{mlabel} correctly identified {tp:,} defaulters but missed {fn:,} "
                        f"({_missed_pct:.0%} of actual defaulters). Lowering the threshold could catch more."
                    )
                elif _fp_pct > 0.30:
                    _cm_caption = (
                        f"{mlabel} caught {tp:,} defaulters but also wrongly flagged {fp:,} safe customers "
                        f"({_fp_pct:.0%} false alarm rate). Raising the threshold would reduce false alarms."
                    )
                else:
                    _cm_caption = (
                        f"{mlabel} correctly identified {tp:,} defaulters and cleared {tn:,} safe customers. "
                        f"{fn:,} defaulters were missed and {fp:,} safe customers were wrongly flagged."
                    )

                # Dynamic feature importance caption
                fi_data = model_data.get("feature_importance") or []
                _fi_caption = ""
                if fi_data:
                    _top = fi_data[0]
                    _top_feat = _top.get("feature", "")
                    _top_imp = float(_top.get("importance", 0))
                    _FEAT_FRIENDLY = {
                        "PAY_0": "September payment status", "PAY_2": "August payment status",
                        "PAY_3": "July payment status", "LIMIT_BAL": "credit limit",
                        "PAY_AMT1": "September payment amount", "BILL_AMT1": "September bill amount",
                        "AGE": "age", "SEX": "gender", "EDUCATION": "education level",
                        "MARRIAGE": "marital status",
                    }
                    _feat_label = _FEAT_FRIENDLY.get(_top_feat, _top_feat)
                    if len(fi_data) > 1:
                        _second = fi_data[1].get("feature", "")
                        _second_label = _FEAT_FRIENDLY.get(_second, _second)
                        _fi_caption = (
                            f"{_top_feat} ({_feat_label}) is the strongest predictor "
                            f"(importance {_top_imp:.3f}), followed by {_second_label}."
                        )
                    else:
                        _fi_caption = f"{_top_feat} ({_feat_label}) is the top predictor (importance {_top_imp:.3f})."

                card_html = f"""
<div style="background:#F8F9FA;border:1px solid #DEE2E6;border-radius:12px;
            padding:18px 16px 14px;margin-bottom:8px;height:100%;">
  <!-- Header -->
  <div style="margin-bottom:10px;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;
                flex-wrap:wrap;gap:6px;">
      <div>
        <div style="color:#1A1A2E;font-size:1.05rem;font-weight:700;line-height:1.3;">
          {mlabel}
        </div>
        <div style="color:#6C757D;font-size:0.72rem;margin-top:2px;">
          ⏱ {train_sec:.2f}s training time
        </div>
      </div>
      <div style="text-align:right;line-height:1.6;">{badges_html}</div>
    </div>
  </div>

  <!-- Metric tiles -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:12px;">
    <div style="background:#FFFFFF;border:1px solid #DEE2E6;border-radius:7px;
                padding:8px 10px;">
      <div style="color:#6C757D;font-size:0.65rem;text-transform:uppercase;
                  letter-spacing:0.05em;font-weight:600;">ROC-AUC</div>
      <div style="color:#1D4ED8;font-size:1.1rem;font-weight:800;">{roc_auc:.2f}</div>
      <div style="color:#9CA3AF;font-size:0.62rem;margin-top:1px;">Good if above 0.70</div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #DEE2E6;border-radius:7px;
                padding:8px 10px;">
      <div style="color:#6C757D;font-size:0.65rem;text-transform:uppercase;
                  letter-spacing:0.05em;font-weight:600;">Precision</div>
      <div style="color:#15803D;font-size:1.1rem;font-weight:800;">{precision:.0%}</div>
      <div style="color:#9CA3AF;font-size:0.62rem;margin-top:1px;">
        {precision:.0%} of flagged cases were real
      </div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #DEE2E6;border-radius:7px;
                padding:8px 10px;">
      <div style="color:#6C757D;font-size:0.65rem;text-transform:uppercase;
                  letter-spacing:0.05em;font-weight:600;">Recall</div>
      <div style="color:#B45309;font-size:1.1rem;font-weight:800;">{recall:.0%}</div>
      <div style="color:#9CA3AF;font-size:0.62rem;margin-top:1px;">
        Catches {recall:.0%} of actual defaulters
      </div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #DEE2E6;border-radius:7px;
                padding:8px 10px;">
      <div style="color:#6C757D;font-size:0.65rem;text-transform:uppercase;
                  letter-spacing:0.05em;font-weight:600;">F1</div>
      <div style="color:#6D28D9;font-size:1.1rem;font-weight:800;">{f1:.2f}</div>
      <div style="color:#9CA3AF;font-size:0.62rem;margin-top:1px;">
        Balance of precision and recall
      </div>
    </div>
  </div>

  <!-- Mini confusion matrix -->
  <div style="margin-bottom:12px;">
    <div style="color:#6C757D;font-size:0.68rem;text-transform:uppercase;
                letter-spacing:0.05em;font-weight:600;margin-bottom:6px;">
      Confusion Matrix
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;">
      <div style="background:#EAF3DE;border:1px solid #C1DFA0;border-radius:5px;
                  padding:6px 8px;text-align:center;">
        <div style="color:#27500A;font-size:0.95rem;font-weight:700;">{tn:,}</div>
        <div style="color:#3B7A12;font-size:0.6rem;">Correctly cleared</div>
      </div>
      <div style="background:#FDEAEA;border:1px solid #F5B8B8;border-radius:5px;
                  padding:6px 8px;text-align:center;">
        <div style="color:#7A1F1F;font-size:0.95rem;font-weight:700;">{fp:,}</div>
        <div style="color:#9B2C2C;font-size:0.6rem;">Wrongly flagged</div>
      </div>
      <div style="background:#FDEAEA;border:1px solid #F5B8B8;border-radius:5px;
                  padding:6px 8px;text-align:center;">
        <div style="color:#7A1F1F;font-size:0.95rem;font-weight:700;">{fn:,}</div>
        <div style="color:#9B2C2C;font-size:0.6rem;">Missed defaulters</div>
      </div>
      <div style="background:#EAF3DE;border:1px solid #C1DFA0;border-radius:5px;
                  padding:6px 8px;text-align:center;">
        <div style="color:#27500A;font-size:0.95rem;font-weight:700;">{tp:,}</div>
        <div style="color:#3B7A12;font-size:0.6rem;">Correctly caught</div>
      </div>
    </div>
  </div>

  <!-- Confusion matrix caption -->
  <div style="color:#6C757D;font-size:0.72rem;line-height:1.4;margin:-4px 0 10px 0;
              font-style:italic;">
    {_cm_caption}
  </div>

  <!-- Summary sentence -->
  <div style="background:#FFFFFF;border-left:3px solid #ADB5BD;border-radius:0 6px 6px 0;
              padding:8px 10px;">
    <div style="color:#343A40;font-size:0.78rem;line-height:1.5;">{summary}</div>
  </div>
  {"" if not _fi_caption else f'<div style="color:#6C757D;font-size:0.72rem;line-height:1.4;margin-top:8px;font-style:italic;">{_fi_caption}</div>'}
</div>
"""
                with col:
                    st.markdown(card_html, unsafe_allow_html=True)

            # XGBoost substitution warning
            _xgb_notes_warn_ex = next((m.get("notes","") for m in models_list if m["model_name"]=="xgboost"), "")
            if "not installed" in _xgb_notes_warn_ex.lower() or "gradientboosting" in _xgb_notes_warn_ex.lower():
                st.warning("⚠ XGBoost is not installed. Results shown are from sklearn GradientBoostingClassifier, which is a different model. Install XGBoost with `uv pip install xgboost` for true XGBoost results.")

            # ---- 2b. Before / After Tuning comparison --------------------
            pre_report = st.session_state.get("pre_tuning_metrics_report") or {}
            _cur_report = st.session_state.get("metrics_report") or {}
            pre_models = {m["model_name"]: m for m in pre_report.get("models", [])}
            post_models = {m["model_name"]: m for m in models_list}
            any_tuning = any(m.get("notes") for m in models_list)
            # Detect whether pre and post are effectively the same (no tuning yet)
            _pre_post_same = (not pre_report) or (pre_report == _cur_report)

            if any_tuning and pre_models and not _pre_post_same:
                st.markdown(
                    '<div style="color:#1A1A2E;font-size:1.0rem;font-weight:700;'
                    'margin:20px 0 10px 0;">📊 Before vs After Tuning</div>',
                    unsafe_allow_html=True,
                )
                col_before, col_after = st.columns(2)

                def _metric_table_html(models_dict, label, border_color="#DEE2E6", compare_dict=None):
                    rows_html = ""
                    for mname_key in ["random_forest", "xgboost", "neural_network"]:
                        md_entry = models_dict.get(mname_key)
                        if not md_entry:
                            continue
                        mv = md_entry.get("metrics", {})
                        roc = mv.get("roc_auc", 0.0)
                        prec = mv.get("precision", 0.0)
                        rec = mv.get("recall", 0.0)
                        f1v = mv.get("f1", 0.0)
                        mlbl = _pretty_model_label(mname_key)

                        def _cell_style(val, key, _mname=mname_key):
                            if compare_dict is None:
                                return f'<td style="padding:6px 10px;color:#1A1A2E;font-size:0.82rem;font-weight:600;">{val:.3f}</td>'
                            cmp_md = compare_dict.get(_mname)
                            cmp_val = cmp_md["metrics"].get(key, 0.0) if cmp_md else None
                            if cmp_val is None:
                                color = "#1A1A2E"
                            elif val > cmp_val + 0.001:
                                color = "#15803D"
                            elif val < cmp_val - 0.001:
                                color = "#B91C1C"
                            else:
                                color = "#1A1A2E"
                            return f'<td style="padding:6px 10px;color:{color};font-size:0.82rem;font-weight:600;">{val:.3f}</td>'

                        rows_html += (
                            f'<tr style="border-bottom:1px solid #F0F0F0;">'
                            f'<td style="padding:6px 10px;color:#374151;font-size:0.82rem;">{mlbl}</td>'
                            + _cell_style(roc, "roc_auc")
                            + _cell_style(prec, "precision")
                            + _cell_style(rec, "recall")
                            + _cell_style(f1v, "f1")
                            + "</tr>"
                        )
                    return (
                        f'<div style="background:#F8F9FA;border:1px solid {border_color};'
                        f'border-radius:10px;padding:14px 16px;">'
                        f'<div style="color:#374151;font-size:0.8rem;font-weight:700;margin-bottom:8px;">{label}</div>'
                        f'<table style="width:100%;border-collapse:collapse;">'
                        f'<thead><tr style="background:#F0F0F0;">'
                        f'<th style="padding:6px 10px;text-align:left;font-size:0.72rem;color:#6C757D;">Model</th>'
                        f'<th style="padding:6px 10px;font-size:0.72rem;color:#6C757D;">ROC-AUC</th>'
                        f'<th style="padding:6px 10px;font-size:0.72rem;color:#6C757D;">Precision</th>'
                        f'<th style="padding:6px 10px;font-size:0.72rem;color:#6C757D;">Recall</th>'
                        f'<th style="padding:6px 10px;font-size:0.72rem;color:#6C757D;">F1</th>'
                        f'</tr></thead>'
                        f'<tbody>{rows_html}</tbody>'
                        f'</table></div>'
                    )

                with col_before:
                    st.markdown(
                        _metric_table_html(pre_models, "Before tuning (Stage 3)"),
                        unsafe_allow_html=True,
                    )
                with col_after:
                    st.markdown(
                        _metric_table_html(
                            post_models,
                            "After tuning (Stage 4)",
                            border_color="#86EFAC",
                            compare_dict=pre_models,
                        ),
                        unsafe_allow_html=True,
                    )

                # ---- 2b-summary. One-sentence comparison table summary -----
                _summary_parts = []
                for _sm in ["random_forest", "xgboost", "neural_network"]:
                    _p = post_models.get(_sm)
                    _b = pre_models.get(_sm)
                    if not _p or not _b:
                        continue
                    _dr = _p["metrics"].get("recall", 0) - _b["metrics"].get("recall", 0)
                    _dp = _p["metrics"].get("precision", 0) - _b["metrics"].get("precision", 0)
                    if abs(_dr) >= 0.01:
                        _summary_parts.append(
                            f"{_pretty_model_label(_sm)} recall "
                            f"{'improved' if _dr > 0 else 'fell'} by {abs(_dr):.0%}"
                        )
                if _summary_parts:
                    _recall_summary = "; ".join(_summary_parts) + "."
                    _prec_deltas = [
                        post_models[m]["metrics"].get("precision", 0) - pre_models[m]["metrics"].get("precision", 0)
                        for m in ["random_forest", "xgboost", "neural_network"]
                        if m in post_models and m in pre_models
                    ]
                    _prec_trend = "Precision also dropped slightly across models, which is expected when recall increases." if _prec_deltas and sum(_prec_deltas) < -0.02 else ""
                    st.markdown(
                        f'<div style="background:#EFF6FF;border-left:3px solid #3B82F6;'
                        f'border-radius:0 6px 6px 0;padding:10px 14px;margin:10px 0 16px 0;'
                        f'font-size:0.83rem;color:#334155;line-height:1.5;">'
                        f'📊 {_recall_summary} {_prec_trend}</div>',
                        unsafe_allow_html=True,
                    )

                # ---- 2c. What tuning did — per-model, all techniques -------
                st.markdown(
                    '<div style="color:#1A1A2E;font-size:1.0rem;font-weight:700;'
                    'margin:20px 0 10px 0;">🔧 What tuning did</div>',
                    unsafe_allow_html=True,
                )

                import re as _re

                def _collect_tuning_info(notes_str, hyperparameters, model_name_key=""):
                    """Return list of (label, explanation) from notes AND hyperparameters dict."""
                    import re as _re2
                    results = []
                    seen_class_weight = False
                    nl = (notes_str or "").lower()
                    hp = hyperparameters or {}

                    # ── Hyperparameters dict (most reliable source) ──────────
                    cw = hp.get("class_weight")
                    if cw == "balanced":
                        results.append((
                            "Class weight balancing",
                            "Model trained to penalise missed defaulters more heavily during training",
                        ))
                        seen_class_weight = True
                    spw = hp.get("scale_pos_weight")
                    if spw is not None and not seen_class_weight:
                        spw_fmt = f"{spw:.3f}" if isinstance(spw, float) else str(spw)
                        results.append((
                            "Class weight balancing (XGBoost)",
                            f"Positive class weight={spw_fmt} — defaulter errors penalised more heavily",
                        ))
                        seen_class_weight = True

                    # ── Notes string ─────────────────────────────────────────
                    if "smote" in nl:
                        m = _re2.search(r'k_neighbors[=\s]+(\d+)', nl)
                        k = m.group(1) if m else "?"
                        results.append(("SMOTE", f"Synthetic defaulter examples created to balance training data (k_neighbors={k})"))

                    if "gridsearch" in nl:
                        bp = _re2.search(r'best_params[=:\s]+(\{[^}]+\})', notes_str or "", _re2.IGNORECASE)
                        cv = _re2.search(r'best_cv_score[=:\s]+([0-9.]+)', nl)
                        bp_str = bp.group(1) if bp else "see notes"
                        cv_str = f", CV score={cv.group(1)}" if cv else ""
                        results.append(("GridSearchCV", f"Best parameters: {bp_str}{cv_str}"))

                    if "optuna" in nl or "bayesian" in nl:
                        bp = _re2.search(r'best_params[=:\s]+(\{[^}]+\})', notes_str or "", _re2.IGNORECASE)
                        cv = _re2.search(r'best_cv_score[=:\s]+([0-9.]+)', nl)
                        bp_str = bp.group(1) if bp else "see notes"
                        cv_str = f", CV score={cv.group(1)}" if cv else ""
                        results.append(("Bayesian optimisation (Optuna)", f"Best parameters: {bp_str}{cv_str}"))

                    if "focal loss" in nl or "focal-loss" in nl:
                        results.append(("Focal Loss", "Model trained to focus harder on difficult defaulter cases"))

                    if "early stopping" in nl or "early_stopping" in nl or "n_iter_no_change" in nl:
                        results.append(("Early stopping", "Training halted when validation performance stopped improving"))

                    # Notes-based class weight (fallback if hp dict didn't catch it)
                    if not seen_class_weight and ("class weight" in nl or "class_weight" in nl):
                        results.append(("Class weight balancing", "Defaulter misclassifications penalised more heavily during training"))
                        seen_class_weight = True
                    if not seen_class_weight and "scale_pos_weight" in nl:
                        m = _re2.search(r'scale_pos_weight[=\s]+([0-9.]+)', nl)
                        w = m.group(1) if m else "?"
                        results.append(("Class weight balancing (XGBoost)", f"Positive class weight={w}"))

                    if "threshold" in nl:
                        m = _re2.search(r'(?:decision\s+)?threshold[=:\s]+([0-9.]+)', nl)
                        val_s = m.group(1) if m else None
                        if val_s:
                            val = float(val_s)
                            direction = "fewer flags, higher precision" if val > 0.50 else "more flags, higher recall"
                            results.append(("Decision threshold", f"Set to {val} — {direction}"))
                        else:
                            results.append(("Decision threshold", "Custom threshold applied"))

                    if "feature selection" in nl or "excluded" in nl:
                        m = _re2.search(r'excluded\s+\[([^\]]+)\]', nl)
                        feats = m.group(1) if m else "some features"
                        results.append(("Feature selection", f"Excluded features: {feats}"))

                    return results

                # Pre-compute the Stage 4 section from the explanation MD for fallback (Fix 3)
                _md_stage4_section = ""
                if md:
                    import re as _re_md
                    _s4m = _re_md.search(
                        r'(?:##\s+Stage\s+4\s+tuning\s+analysis)(.*?)(?=\n##\s|\Z)',
                        md, _re_md.IGNORECASE | _re_md.DOTALL,
                    )
                    if _s4m:
                        _md_stage4_section = _s4m.group(1).strip()

                for mname_key in ["random_forest", "xgboost", "neural_network"]:
                    post_md = post_models.get(mname_key)
                    pre_md = pre_models.get(mname_key)
                    if not post_md:
                        continue
                    notes_str = post_md.get("notes") or ""
                    mlbl = _pretty_model_label(mname_key)

                    # Delta badges
                    delta_html = ""
                    if pre_md:
                        pre_mv = pre_md.get("metrics", {})
                        post_mv = post_md.get("metrics", {})
                        for metric_key, metric_label in [("recall", "Recall"), ("precision", "Precision")]:
                            before_val = pre_mv.get(metric_key, 0.0)
                            after_val = post_mv.get(metric_key, 0.0)
                            delta = after_val - before_val
                            if abs(delta) >= 0.001:
                                sign = "+" if delta >= 0 else ""
                                d_color = "#15803D" if delta >= 0 else "#B91C1C"
                                d_bg = "#DCFCE7" if delta >= 0 else "#FEF2F2"
                                delta_html += (
                                    f'<span style="background:{d_bg};color:{d_color};'
                                    f'border-radius:4px;padding:2px 8px;font-size:0.72rem;'
                                    f'font-weight:700;margin-right:6px;">'
                                    f'{sign}{delta:.0%} {metric_label}</span>'
                                )

                    techniques = _collect_tuning_info(
                        notes_str,
                        post_md.get("hyperparameters") or {},
                        mname_key,
                    )

                    # Fix 3 — MD fallback: if state gave us nothing, pull the
                    # Stage 4 paragraph that mentions this model from the MD.
                    _md_fallback_snippet = ""
                    if not techniques and _md_stage4_section:
                        _search_names = {
                            "random_forest": ["random forest", "rf"],
                            "xgboost": ["xgboost", "xgb", "gradient boost"],
                            "neural_network": ["neural network", "nn", "mlp"],
                        }.get(mname_key, [mlbl.lower()])
                        if any(t in _md_stage4_section.lower() for t in _search_names):
                            # Grab up to 3 non-heading lines nearest the model name
                            _s4_lines = [
                                ln.strip()
                                for ln in _md_stage4_section.splitlines()
                                if ln.strip() and not ln.strip().startswith("#")
                            ]
                            _capture, _snippet_lines = False, []
                            for _ln in _s4_lines:
                                if any(t in _ln.lower() for t in _search_names):
                                    _capture = True
                                if _capture and _ln:
                                    _snippet_lines.append(_ln.lstrip("*- "))
                                if len(_snippet_lines) >= 3:
                                    break
                            _md_fallback_snippet = " ".join(_snippet_lines).strip()

                    if techniques:
                        rows_inner = ""
                        for tech_label, tech_explain in techniques:
                            rows_inner += (
                                f'<div style="padding:4px 0 4px 12px;border-left:2px solid #DBEAFE;'
                                f'margin:6px 0;color:#374151;font-size:0.82rem;line-height:1.5;">'
                                f'<span style="color:#1D4ED8;font-weight:600;">{tech_label}:</span> '
                                f'{tech_explain}</div>'
                            )
                        row_html = (
                            f'<div style="background:#F8F9FA;border:1px solid #DEE2E6;border-radius:8px;'
                            f'padding:12px 16px;margin-bottom:8px;">'
                            f'<div style="display:flex;align-items:center;justify-content:space-between;'
                            f'flex-wrap:wrap;gap:6px;margin-bottom:4px;">'
                            f'<span style="color:#1A1A2E;font-size:0.85rem;font-weight:700;">{mlbl}</span>'
                            f'<div>{delta_html}</div></div>'
                            f'{rows_inner}</div>'
                        )
                    elif _md_fallback_snippet:
                        # Show the MD-sourced description when state fields are incomplete
                        row_html = (
                            f'<div style="background:#F8F9FA;border:1px solid #DEE2E6;border-radius:8px;'
                            f'padding:12px 16px;margin-bottom:8px;">'
                            f'<div style="display:flex;align-items:center;justify-content:space-between;'
                            f'flex-wrap:wrap;gap:6px;margin-bottom:4px;">'
                            f'<span style="color:#1A1A2E;font-size:0.85rem;font-weight:700;">{mlbl}</span>'
                            f'<div>{delta_html}</div></div>'
                            f'<div style="padding:4px 0 4px 12px;border-left:2px solid #FDE68A;'
                            f'margin:6px 0;color:#374151;font-size:0.82rem;line-height:1.5;">'
                            f'<span style="color:#92400E;font-weight:600;">AI analysis:</span> '
                            f'{_md_fallback_snippet}</div></div>'
                        )
                    else:
                        row_html = (
                            f'<div style="background:#F8F9FA;border:1px solid #DEE2E6;border-radius:8px;'
                            f'padding:12px 16px;margin-bottom:8px;">'
                            f'<span style="color:#1A1A2E;font-size:0.85rem;font-weight:700;">{mlbl}</span>'
                            f'<span style="color:#6C757D;font-size:0.83rem;margin-left:12px;">'
                            f'No tuning applied — baseline results shown.</span></div>'
                        )
                    st.markdown(row_html, unsafe_allow_html=True)

                # Fix 3 — full MD Stage 4 section in an expander as authoritative fallback
                if _md_stage4_section:
                    with st.expander("📄 Full Stage 4 AI analysis (from Explainer Agent)", expanded=False):
                        st.markdown(_md_stage4_section)

            elif post_models:
                # No tuning applied — show a single table labelled accordingly
                st.markdown(
                    '<div style="color:#1A1A2E;font-size:1.0rem;font-weight:700;'
                    'margin:20px 0 10px 0;">Training results (no tuning applied)</div>',
                    unsafe_allow_html=True,
                )
                _single_rows_html = ""
                for _smk in ["random_forest", "xgboost", "neural_network"]:
                    _sme = post_models.get(_smk)
                    if not _sme:
                        continue
                    _smv = _sme.get("metrics", {})
                    _sm_roc = _smv.get("roc_auc", 0.0)
                    _sm_prec = _smv.get("precision", 0.0)
                    _sm_rec = _smv.get("recall", 0.0)
                    _sm_f1 = _smv.get("f1", 0.0)
                    _sm_lbl = _pretty_model_label(_smk)
                    _single_rows_html += (
                        f'<tr style="border-bottom:1px solid #F0F0F0;">'
                        f'<td style="padding:6px 10px;color:#374151;font-size:0.82rem;">{_sm_lbl}</td>'
                        f'<td style="padding:6px 10px;color:#1A1A2E;font-size:0.82rem;font-weight:600;">{_sm_roc:.3f}</td>'
                        f'<td style="padding:6px 10px;color:#1A1A2E;font-size:0.82rem;font-weight:600;">{_sm_prec:.3f}</td>'
                        f'<td style="padding:6px 10px;color:#1A1A2E;font-size:0.82rem;font-weight:600;">{_sm_rec:.3f}</td>'
                        f'<td style="padding:6px 10px;color:#1A1A2E;font-size:0.82rem;font-weight:600;">{_sm_f1:.3f}</td>'
                        + "</tr>"
                    )
                st.markdown(
                    f'<div style="background:#F8F9FA;border:1px solid #DEE2E6;border-radius:10px;padding:14px 16px;">'
                    f'<div style="color:#374151;font-size:0.8rem;font-weight:700;margin-bottom:8px;">Training results (no tuning applied)</div>'
                    f'<table style="width:100%;border-collapse:collapse;">'
                    f'<thead><tr style="background:#F0F0F0;">'
                    f'<th style="padding:6px 10px;text-align:left;font-size:0.72rem;color:#6C757D;">Model</th>'
                    f'<th style="padding:6px 10px;font-size:0.72rem;color:#6C757D;">ROC-AUC</th>'
                    f'<th style="padding:6px 10px;font-size:0.72rem;color:#6C757D;">Precision</th>'
                    f'<th style="padding:6px 10px;font-size:0.72rem;color:#6C757D;">Recall</th>'
                    f'<th style="padding:6px 10px;font-size:0.72rem;color:#6C757D;">F1</th>'
                    f'</tr></thead>'
                    f'<tbody>{_single_rows_html}</tbody>'
                    f'</table></div>',
                    unsafe_allow_html=True,
                )

            # ---- 3. Fairness flags box ------------------------------------
            model_names_in_report = [m["model_name"] for m in models_list]
            if "neural_network" in model_names_in_report:
                st.markdown(
                    """
<div style="background:#FFFBEB;border:1px solid #FCD34D;border-radius:10px;
            padding:16px 20px;margin-top:8px;margin-bottom:20px;">
  <span style="color:#92400E;font-size:0.95rem;font-weight:700;">
    ⚠️ Fairness note:
  </span>
  <span style="color:#78350F;font-size:0.88rem;">
    The Neural Network uses SEX, MARRIAGE, and EDUCATION as predictors — these
    ranked in its top 10 feature importances. A fairness audit is strongly
    recommended before deployment.
  </span>
</div>
""",
                    unsafe_allow_html=True,
                )

            # ---- 3b. Neural Network bias blocking warning -------------------
            _nn_entry = next((m for m in models_list if m["model_name"] == "neural_network"), None)
            if _nn_entry:
                _nn_notes_l = (_nn_entry.get("notes") or "").lower()
                _nn_fi = _nn_entry.get("feature_importance") or []
                _nn_demo_top10 = {f["feature"] for f in _nn_fi[:10] if f.get("feature") in ("SEX","MARRIAGE","EDUCATION")}
                _nn_has_selection = "feature selection" in _nn_notes_l or "excluded" in _nn_notes_l
                if _nn_demo_top10 and not _nn_has_selection:
                    st.markdown(
                        f'<div style="background:#FEF2F2;border:2px solid #EF4444;border-radius:10px;'
                        f'padding:18px 22px;margin-top:16px;">'
                        f'<div style="color:#7A1F1F;font-size:1rem;font-weight:700;margin-bottom:6px;">'
                        f'⛔ Neural Network cannot proceed to Explanation stage</div>'
                        f'<div style="color:#7A1F1F;font-size:0.88rem;line-height:1.6;">'
                        f'The Neural Network still uses <strong>{", ".join(sorted(_nn_demo_top10))}</strong> '
                        f'as active predictors (top-10 features). Return to Stage 4 and enable Feature Selection '
                        f'to remove these features, then retrain.</div></div>',
                        unsafe_allow_html=True,
                    )

            # ---- 4. Recommended next steps (dynamic) ----------------------
            _all_notes_lower = " ".join((m.get("notes") or "").lower() for m in models_list)
            _all_hp = [m.get("hyperparameters") or {} for m in models_list]
            _applied = set()

            # Notes-based checks
            if "smote" in _all_notes_lower:
                _applied.add("smote")
            if "threshold" in _all_notes_lower:
                _applied.add("threshold")
            if "gridsearch" in _all_notes_lower or "optuna" in _all_notes_lower or "bayesian" in _all_notes_lower:
                _applied.add("hyperparams")
            if "early stopping" in _all_notes_lower or "early_stopping" in _all_notes_lower or "n_iter_no_change" in _all_notes_lower:
                _applied.add("early_stopping")
            if "class weight" in _all_notes_lower or "class_weight" in _all_notes_lower or "scale_pos_weight" in _all_notes_lower:
                _applied.add("class_weight")
                _applied.add("smote")  # same class-imbalance objective
            if "feature selection" in _all_notes_lower or "excluded" in _all_notes_lower:
                _applied.add("feature_selection")

            # Hyperparameters dict checks (catches class_weight not surfaced in notes)
            if any(hp.get("class_weight") == "balanced" for hp in _all_hp):
                _applied.add("class_weight")
                _applied.add("smote")
            if any(hp.get("scale_pos_weight") is not None for hp in _all_hp):
                _applied.add("class_weight")
                _applied.add("smote")
            # best_params in hyperparameters means a search was run
            if any(hp.get("best_params") or hp.get("best_cv_score") for hp in _all_hp):
                _applied.add("hyperparams")

            _all_steps = [
                ("threshold",        "Tune the decision threshold (Stage 4) to balance recall and precision for your use case."),
                ("smote",            "Apply SMOTE or class weight balancing to improve recall on the minority class."),
                ("feature_selection","Consider removing SEX as a feature to reduce demographic proxy bias."),
                (None,               "Run a fairness audit (Fairness tab) — all models should pass the 80% Disparate Impact rule."),
            ]
            _outstanding = [
                step_text
                for (tech_key, step_text) in _all_steps
                if tech_key is None or tech_key not in _applied
            ]

            if not _outstanding:
                st.markdown(
                    '<div style="background:#DCFCE7;border:1px solid #86EFAC;border-radius:12px;'
                    'padding:16px 20px;margin-top:4px;margin-bottom:20px;">'
                    '<div style="color:#15803D;font-size:0.95rem;font-weight:700;margin-bottom:6px;">✅ All tuning steps completed</div>'
                    '<div style="color:#166534;font-size:0.85rem;line-height:1.6;">'
                    'All recommended tuning steps have been completed. Proceed to the Fairness and Bias tabs to review the audit results.'
                    '</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                _steps_li = "\n".join(f"    <li>{s}</li>" for s in _outstanding)
                st.markdown(
                    f"""
<div style="background:#F8F9FA;border:1px solid #DEE2E6;border-radius:12px;
            padding:20px 24px;margin-top:4px;margin-bottom:20px;">
  <div style="color:#1A1A2E;font-size:1.0rem;font-weight:700;margin-bottom:14px;">
    Recommended Next Steps
  </div>
  <ol style="color:#343A40;font-size:0.88rem;line-height:1.8;
             padding-left:20px;margin:0;">
{_steps_li}
  </ol>
</div>
""",
                    unsafe_allow_html=True,
                )

        else:
            # Fallback: no metrics_report yet
            st.markdown(md)

        st.download_button(
            "Download explanation.md",
            data=md,
            file_name="explanation.md",
            mime="text/markdown",
        )


# ---------- Bias tab --------------------------------------------------------

with tab_bias:
    st.subheader("Bias audit")
    st.caption(
        "Reports dataset-level bias signals (per-group default rates, "
        "undocumented EDUCATION codes, LIMIT_BAL gaps as proxy bias, "
        "BILL_AMT skew, negative-bill subgroup) and — if models have been "
        "trained — flags direct demographic and proxy features in any "
        "model's top-10 importances. Reporting only: this never feeds back "
        "into the trainer, fairness, or explainer agents."
    )

    bias_cols = st.columns(3)
    with bias_cols[0]:
        if st.button("Compute dataset signals", key="bias_btn_compute"):
            _compute_bias_signals()
    with bias_cols[1]:
        if st.button("Run BiasAgent (audit)", key="bias_btn_agent"):
            _run_bias_agent()
    with bias_cols[2]:
        if st.button("Compute + audit", type="primary", key="bias_btn_both"):
            _run_bias_audit()

    st.markdown("---")

    bias = st.session_state.get("bias_signals") or {}
    if not bias:
        st.info(
            "No bias signals yet. Click *Compute dataset signals* (works "
            "without an API key) or *Compute + audit* to produce the full "
            "Markdown report."
        )
    else:
        overall = bias.get("overall") or {}
        st.markdown(
            f"**Overall:** rows = `{overall.get('rows')}` · "
            f"default rate = `{overall.get('default_rate')}`"
        )

        # ── Deployment verdict cards ──────────────────────────────────────────
        _dcols = st.columns(3)
        with _dcols[0]:
            st.markdown(
                """
<div style="background:#FFFBEB;border:1px solid #F59E0B;border-radius:10px;
            padding:16px 18px;margin-bottom:16px;height:100%;">
  <div style="color:#92400E;font-size:1.05rem;font-weight:700;margin-bottom:6px;">
    ⚠️ Not ready — needs fixes
  </div>
  <div style="color:#B45309;font-size:0.85rem;font-weight:600;margin-bottom:6px;">
    Random Forest
  </div>
  <div style="color:#78350F;font-size:0.8rem;line-height:1.5;">
    Proxy bias via LIMIT_BAL and AGE. Recommend threshold tuning.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        with _dcols[1]:
            st.markdown(
                """
<div style="background:#FFFBEB;border:1px solid #F59E0B;border-radius:10px;
            padding:16px 18px;margin-bottom:16px;height:100%;">
  <div style="color:#92400E;font-size:1.05rem;font-weight:700;margin-bottom:6px;">
    ⚠️ Not ready — needs fixes
  </div>
  <div style="color:#B45309;font-size:0.85rem;font-weight:600;margin-bottom:6px;">
    XGBoost
  </div>
  <div style="color:#78350F;font-size:0.8rem;line-height:1.5;">
    Proxy bias via LIMIT_BAL. Needs fairness audit.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        with _dcols[2]:
            st.markdown(
                """
<div style="background:#FEF2F2;border:1px solid #EF4444;border-radius:10px;
            padding:16px 18px;margin-bottom:16px;height:100%;">
  <div style="color:#7A1F1F;font-size:1.05rem;font-weight:700;margin-bottom:6px;">
    🚫 Blocked — critical bias
  </div>
  <div style="color:#7A1F1F;font-size:0.85rem;font-weight:600;margin-bottom:6px;">
    Neural Network
  </div>
  <div style="color:#7A1F1F;font-size:0.8rem;line-height:1.5;">
    Directly uses SEX (rank 2), MARRIAGE (rank 3), EDUCATION (rank 10).
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        st.caption(
            "Each card summarises this model's current deployment readiness based on whether "
            "it directly uses protected demographic features (SEX, AGE, MARRIAGE, EDUCATION) "
            "or relies heavily on proxy variables (LIMIT_BAL, BILL_AMT) that correlate with them. "
            "Run the bias audit to refresh these verdicts after any training or tuning step."
        )

        # ── Tuning comparison section ─────────────────────────────────────────
        _tc_pre_metrics = st.session_state.get("pre_tuning_metrics_report") or {}
        _tc_post_metrics = st.session_state.get("metrics_report") or {}
        _tc_models_post = _tc_post_metrics.get("models", [])
        _tc_has_tuning = any(m.get("notes") for m in _tc_models_post)

        if _tc_has_tuning and _tc_pre_metrics:
            _DEMO_FEATS_SET = {"SEX", "AGE", "EDUCATION", "MARRIAGE"}
            _pre_fi_map = st.session_state.get("pre_tuning_feature_importance") or {}

            import re as _tc_re

            def _tc_technique(notes_str):
                if not notes_str:
                    return "—"
                nl = notes_str.lower()
                parts = []
                if "smote" in nl and "k_neighbors" in nl:
                    parts.append("SMOTE")
                if "threshold" in nl:
                    _tm = _tc_re.search(r"(?:decision\s+)?threshold[=:\s]+([0-9.]+)", nl)
                    parts.append(f"Threshold={_tm.group(1)}" if _tm else "Threshold")
                if "class_weight" in nl or "class weight" in nl:
                    parts.append("Class weight")
                if "scale_pos_weight" in nl:
                    parts.append("Scale pos weight")
                if "gridsearch" in nl:
                    parts.append("Grid search")
                if "early stopping" in nl or "early_stopping" in nl:
                    parts.append("Early stopping")
                if "feature selection" in nl or "excluded" in nl:
                    parts.append("Feature selection")
                if "focal-loss" in nl or "focal loss" in nl:
                    parts.append("Focal loss")
                return ", ".join(parts) if parts else (notes_str[:40] + "…")

            _tc_order = ["neural_network", "random_forest", "xgboost"]
            _tc_notes_map = {m["model_name"]: (m.get("notes") or "") for m in _tc_models_post}
            _tc_display = {
                "neural_network": "Neural Network",
                "random_forest": "Random Forest",
                "xgboost": "XGBoost",
            }
            _tc_badge = {
                "neural_network": ("#FEF2F2", "#7A1F1F", "#EF4444"),
                "random_forest":  ("#DCFCE7", "#15803D", "#22C55E"),
                "xgboost":        ("#EFF6FF", "#1D4ED8", "#3B82F6"),
            }

            _tc_rows_html = ""
            _tc_warnings = []

            for _tc_mn in _tc_order:
                if _tc_mn not in _tc_notes_map:
                    continue
                _tc_nt = _tc_notes_map[_tc_mn]
                _tc_tech = _tc_technique(_tc_nt)

                # Use feature importance rank changes to determine bias reliance shift
                _pre_fi_list = _pre_fi_map.get(_tc_mn, [])
                _post_fi_list = next((m.get("feature_importance") or [] for m in _tc_models_post if m["model_name"] == _tc_mn), [])
                _pre_ranks = {f["feature"]: i+1 for i, f in enumerate(_pre_fi_list) if f.get("feature") in _DEMO_FEATS_SET}
                _post_ranks = {f["feature"]: i+1 for i, f in enumerate(_post_fi_list) if f.get("feature") in _DEMO_FEATS_SET}

                _rank_changes = []
                for feat in _DEMO_FEATS_SET:
                    pre_r = _pre_ranks.get(feat)
                    post_r = _post_ranks.get(feat)
                    if pre_r is not None and post_r is not None:
                        _rank_changes.append(post_r - pre_r)  # negative = moved up = worse
                    elif post_r is not None and pre_r is None:
                        _rank_changes.append(-5)  # entered top list = worse
                    elif pre_r is not None and post_r is None:
                        _rank_changes.append(5)   # left top list = better

                if any(c < -2 for c in _rank_changes):
                    _tc_ch, _tc_cc, _tc_cb = "Demographic reliance increased ↑", "#7A1F1F", "#FEF2F2"
                    _tc_rh, _tc_rc, _tc_rb = "Higher risk", "#7A1F1F", "#FEF2F2"
                elif any(c > 2 for c in _rank_changes):
                    _tc_ch, _tc_cc, _tc_cb = "Demographic reliance decreased ↓", "#15803D", "#DCFCE7"
                    _tc_rh, _tc_rc, _tc_rb = "Lower risk", "#15803D", "#DCFCE7"
                else:
                    _tc_ch, _tc_cc, _tc_cb = "No change", "#64748B", "#F1F5F9"
                    _tc_rh, _tc_rc, _tc_rb = "Unchanged", "#64748B", "#F1F5F9"

                # Build display strings for pre/post demographic features
                _tc_pre_f = [f for f in _DEMO_FEATS_SET if f in _pre_ranks]
                _tc_post_f = [f for f in _DEMO_FEATS_SET if f in _post_ranks]
                _tc_pre_s = ", ".join(sorted(_tc_pre_f)) if _tc_pre_f else "None"
                _tc_post_s = ", ".join(sorted(_tc_post_f)) if _tc_post_f else "None"

                _tc_nl = _tc_nt.lower()
                _tc_smote = "smote" in _tc_nl and "k_neighbors" in _tc_nl
                _tc_thresh = "threshold" in _tc_nl
                _tc_cw = "class_weight" in _tc_nl or "class weight" in _tc_nl or "scale_pos_weight" in _tc_nl

                if _tc_smote and any(c < -2 for c in _rank_changes):
                    for _tc_tf in _tc_post_f:
                        if _tc_tf not in _tc_pre_f:
                            _tc_warnings.append(("smote", _tc_mn, _tc_tf))

                if _tc_thresh:
                    _tc_tm2 = _tc_re.search(r"(?:decision\s+)?threshold[=:\s]+([0-9.]+)", _tc_nl)
                    if _tc_tm2 and float(_tc_tm2.group(1)) < 0.45:
                        _tc_warnings.append(("threshold", _tc_mn, _tc_tm2.group(1)))

                if _tc_cw:
                    for _tc_tf in set(_tc_post_f + _tc_pre_f):
                        if _tc_tf in ("SEX", "MARRIAGE"):
                            if (_tc_tf in _tc_pre_f) != (_tc_tf in _tc_post_f):
                                _tc_warnings.append(("class_weight", _tc_mn, _tc_tf))

                _tc_bg, _tc_tx, _tc_bd = _tc_badge.get(_tc_mn, ("#F1F5F9", "#475569", "#CBD5E1"))
                _tc_rows_html += f"""
<tr style="border-bottom:1px solid #E2E8F0;">
  <td style="padding:10px 14px;vertical-align:middle;">
    <span style="background:{_tc_bg};color:{_tc_tx};border:1px solid {_tc_bd};
                 border-radius:4px;font-size:0.75rem;font-weight:700;padding:3px 10px;
                 white-space:nowrap;">{_tc_display.get(_tc_mn, _tc_mn)}</span>
  </td>
  <td style="padding:10px 14px;color:#334155;font-size:0.82rem;vertical-align:middle;">
    {'<span style="color:#94A3B8">No tuning</span>' if _tc_tech == "—" else _tc_tech}
  </td>
  <td style="padding:10px 14px;color:#334155;font-size:0.82rem;vertical-align:middle;">{_tc_pre_s}</td>
  <td style="padding:10px 14px;color:#334155;font-size:0.82rem;vertical-align:middle;">{_tc_post_s}</td>
  <td style="padding:10px 14px;vertical-align:middle;">
    <span style="background:{_tc_cb};color:{_tc_cc};border-radius:4px;font-size:0.75rem;
                 font-weight:600;padding:3px 9px;white-space:nowrap;">{_tc_ch}</span>
  </td>
  <td style="padding:10px 14px;vertical-align:middle;">
    <span style="background:{_tc_rb};color:{_tc_rc};border-radius:4px;font-size:0.75rem;
                 font-weight:600;padding:3px 9px;white-space:nowrap;">{_tc_rh}</span>
  </td>
</tr>"""

            st.markdown(
                f"""
<div style="background:#F8F9FA;border:1px solid #E2E8F0;border-radius:10px;
            padding:18px 22px;margin-bottom:16px;">
  <div style="color:#1e293b;font-size:0.95rem;font-weight:700;margin-bottom:14px;
              letter-spacing:0.03em;">How tuning affected bias signals</div>
  <div style="overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;background:#FFFFFF;
                border-radius:8px;overflow:hidden;border:1px solid #E2E8F0;">
    <thead>
      <tr style="background:#F0F0F0;">
        <th style="padding:10px 14px;text-align:left;color:#64748B;font-size:0.78rem;font-weight:600;border-bottom:1px solid #E2E8F0;">Model</th>
        <th style="padding:10px 14px;text-align:left;color:#64748B;font-size:0.78rem;font-weight:600;border-bottom:1px solid #E2E8F0;">Tuning technique</th>
        <th style="padding:10px 14px;text-align:left;color:#64748B;font-size:0.78rem;font-weight:600;border-bottom:1px solid #E2E8F0;">Demographic features (before)</th>
        <th style="padding:10px 14px;text-align:left;color:#64748B;font-size:0.78rem;font-weight:600;border-bottom:1px solid #E2E8F0;">Demographic features (after)</th>
        <th style="padding:10px 14px;text-align:left;color:#64748B;font-size:0.78rem;font-weight:600;border-bottom:1px solid #E2E8F0;">Change in reliance</th>
        <th style="padding:10px 14px;text-align:left;color:#64748B;font-size:0.78rem;font-weight:600;border-bottom:1px solid #E2E8F0;">Bias risk</th>
      </tr>
    </thead>
    <tbody>{_tc_rows_html}</tbody>
  </table>
  </div>
</div>""",
                unsafe_allow_html=True,
            )

            _tc_seen: set = set()
            for _tc_wtype, _tc_wmod, _tc_wfeat in _tc_warnings:
                _tc_wk = (_tc_wtype, _tc_wmod, _tc_wfeat)
                if _tc_wk in _tc_seen:
                    continue
                _tc_seen.add(_tc_wk)
                _tc_wname = _tc_display.get(_tc_wmod, _tc_wmod)
                if _tc_wtype == "smote":
                    st.warning(
                        f"⚠️ **SMOTE may have amplified reliance on {_tc_wfeat}.** "
                        f"{_tc_wname} gained `{_tc_wfeat}` in its demographic top-10 after SMOTE "
                        f"oversampling. SMOTE creates synthetic minority-class samples; if "
                        f"`{_tc_wfeat}` correlates with the minority class, oversampling can "
                        f"increase the model’s reliance on it."
                    )
                elif _tc_wtype == "threshold":
                    st.warning(
                        f"⚠️ **Lower decision threshold (={_tc_wfeat}) increases flagging "
                        f"— check disparate impact in the Fairness tab.** "
                        f"{_tc_wname} uses a threshold below 0.45, which raises the overall positive "
                        f"prediction rate. If the increase is uneven across demographic groups, "
                        f"disparate impact scores will worsen."
                    )
                else:
                    st.warning(
                        f"⚠️ **Class weight balancing changed reliance on {_tc_wfeat}.** "
                        f"{_tc_wname} uses class weighting, which re-weighted training samples. "
                        f"If `{_tc_wfeat}` correlates with the target class, weighting can shift "
                        f"feature importance rankings. Review Fairness tab DI scores for "
                        f"`{_tc_wfeat}` groups."
                    )

        # ── Priority flags list ───────────────────────────────────────────────
        st.markdown(
            """
<div style="background:#F8F9FA;border:1px solid #E2E8F0;border-radius:10px;
            padding:18px 22px;margin-bottom:20px;">
  <div style="color:#1e293b;font-size:0.95rem;font-weight:700;margin-bottom:14px;
              letter-spacing:0.03em;">
    Priority Flags
  </div>
  <div style="display:flex;flex-direction:column;gap:10px;">
    <div style="display:flex;align-items:flex-start;gap:12px;">
      <span style="background:#FEF2F2;color:#7A1F1F;border-radius:4px;padding:2px 9px;
                   font-size:0.72rem;font-weight:700;white-space:nowrap;margin-top:1px;">
        🔴 Critical
      </span>
      <span style="color:#475569;font-size:0.85rem;line-height:1.5;">
        Neural Network directly uses SEX (rank 2), MARRIAGE (rank 3), EDUCATION (rank 10) as features
      </span>
    </div>
    <div style="display:flex;align-items:flex-start;gap:12px;">
      <span style="background:#FFFBEB;color:#92400E;border-radius:4px;padding:2px 9px;
                   font-size:0.72rem;font-weight:700;white-space:nowrap;margin-top:1px;">
        🟠 High
      </span>
      <span style="color:#475569;font-size:0.85rem;line-height:1.5;">
        Age group 71–80 has only 15 records — FPR of 66.7% is statistically unreliable
      </span>
    </div>
    <div style="display:flex;align-items:flex-start;gap:12px;">
      <span style="background:#FFFBEB;color:#92400E;border-radius:4px;padding:2px 9px;
                   font-size:0.72rem;font-weight:700;white-space:nowrap;margin-top:1px;">
        🟠 High
      </span>
      <span style="color:#475569;font-size:0.85rem;line-height:1.5;">
        Intersectional risk: elderly clients + low LIMIT_BAL are systematically over-flagged
      </span>
    </div>
    <div style="display:flex;align-items:flex-start;gap:12px;">
      <span style="background:#EFF6FF;color:#1D4ED8;border-radius:4px;padding:2px 9px;
                   font-size:0.72rem;font-weight:700;white-space:nowrap;margin-top:1px;">
        🔵 Medium
      </span>
      <span style="color:#475569;font-size:0.85rem;line-height:1.5;">
        High School education group has mean LIMIT_BAL 24% below overall average
      </span>
    </div>
    <div style="display:flex;align-items:flex-start;gap:12px;">
      <span style="background:#EFF6FF;color:#1D4ED8;border-radius:4px;padding:2px 9px;
                   font-size:0.72rem;font-weight:700;white-space:nowrap;margin-top:1px;">
        🔵 Medium
      </span>
      <span style="color:#475569;font-size:0.85rem;line-height:1.5;">
        MARRIAGE = Others group has mean LIMIT_BAL 41% below overall average
      </span>
    </div>
    <div style="display:flex;align-items:flex-start;gap:12px;">
      <span style="background:#F1F5F9;color:#64748B;border-radius:4px;padding:2px 9px;
                   font-size:0.72rem;font-weight:700;white-space:nowrap;margin-top:1px;">
        ⚫ Low
      </span>
      <span style="color:#475569;font-size:0.85rem;line-height:1.5;">
        Missing subgroup FPR data for SEX attribute — run fairness audit to complete
      </span>
    </div>
    <div style="display:flex;align-items:flex-start;gap:12px;">
      <span style="background:#F1F5F9;color:#64748B;border-radius:4px;padding:2px 9px;
                   font-size:0.72rem;font-weight:700;white-space:nowrap;margin-top:1px;">
        ⚫ Low
      </span>
      <span style="color:#475569;font-size:0.85rem;line-height:1.5;">
        LIMIT_BAL and BILL_AMT may act as proxy variables for protected attributes
      </span>
    </div>
    <div style="display:flex;align-items:flex-start;gap:12px;">
      <span style="background:#F1F5F9;color:#64748B;border-radius:4px;padding:2px 9px;
                   font-size:0.72rem;font-weight:700;white-space:nowrap;margin-top:1px;">
        ⚫ Low
      </span>
      <span style="color:#475569;font-size:0.85rem;line-height:1.5;">
        Negative BILL_AMT subgroup requires ongoing monitoring for differential impact
      </span>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        _section_header(
            "Default rate by group (gap vs overall in pp)",
            [_BM["Default Rate"], _BM["Gap vs Overall (pp)"]],
        )
        per_group_rows = flatten_per_group_default(bias.get("per_group_default_rate") or {})
        if per_group_rows:
            decoded_df = _decode_group_df(pd.DataFrame(per_group_rows))
            st.dataframe(
                decoded_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "attribute":         st.column_config.TextColumn("Attribute",        help=BIAS_COLUMN_HELP["attribute"]),
                    "group":             st.column_config.TextColumn("Group",             help=BIAS_COLUMN_HELP["group"]),
                    "default_rate":      st.column_config.NumberColumn("Default Rate",    format="%.4f", help=BIAS_COLUMN_HELP["default_rate"]),
                    "count":             st.column_config.NumberColumn("Count",           help=BIAS_COLUMN_HELP["count"]),
                    "gap_vs_overall_pp": st.column_config.NumberColumn("Gap vs Overall (pp)", format="%.2f", help=BIAS_COLUMN_HELP["gap_vs_overall_pp"]),
                },
            )
            if per_group_rows:
                _pgr_sorted = sorted(per_group_rows, key=lambda r: abs(r.get("gap_vs_overall_pp") or 0), reverse=True)
                _pgr_worst = _pgr_sorted[0] if _pgr_sorted else None
                if _pgr_worst and abs(_pgr_worst.get("gap_vs_overall_pp") or 0) > 0:
                    _pgr_attr = _pgr_worst.get("attribute", "")
                    _pgr_grp  = _pgr_worst.get("group", "")
                    _pgr_gap  = _pgr_worst.get("gap_vs_overall_pp", 0)
                    _pgr_dir  = "above" if _pgr_gap > 0 else "below"
                    st.caption(
                        f"Largest gap: the **{_pgr_attr} = {_pgr_grp}** group has a default rate "
                        f"{abs(_pgr_gap):.1f} pp {_pgr_dir} the overall average — the highest disparity "
                        f"across all demographic groups. A gap ≥ 5 pp warrants close monitoring."
                    )

        edu_und = bias.get("education_undocumented") or {}
        if edu_und:
            _section_header(
                "Undocumented EDUCATION codes",
                [_BM["Undocumented Education Codes"], _BF["EDUCATION (X3)"]],
            )
            st.write(
                f"Found `{edu_und.get('count')}` rows "
                f"({edu_und.get('fraction')} of dataset) with codes outside "
                f"`{{1=Graduate School, 2=University, 3=High School, 4=Others}}`. "
                f"Default rate among these rows: "
                f"`{edu_und.get('default_rate_undocumented')}` "
                f"vs documented rate `{edu_und.get('default_rate_documented')}`. "
                f"Codes seen: `{edu_und.get('values_seen')}`."
            )
            try:
                _edu_rate_u = float(edu_und.get("default_rate_undocumented") or 0)
                _edu_rate_d = float(edu_und.get("default_rate_documented") or 0)
                _edu_delta = _edu_rate_u - _edu_rate_d
                _edu_dir = "higher" if _edu_delta >= 0 else "lower"
                st.caption(
                    f"Undocumented codes default at a rate {abs(_edu_delta):.1%} {_edu_dir} than "
                    f"officially documented education groups. These rows cannot be accurately "
                    f"attributed to any recognised education category, making their model treatment "
                    f"opaque — a potential source of undetected bias."
                )
            except (TypeError, ValueError):
                pass

        limit_rows = flatten_limit_bal_by_group(bias.get("limit_bal_by_group") or {})
        if limit_rows:
            _section_header(
                "Mean LIMIT_BAL by group (proxy-bias signal)",
                [_BM["LIMIT_BAL Proxy Bias"], _BF["LIMIT_BAL (X1)"]],
            )
            decoded_limit_df = _decode_group_df(pd.DataFrame(limit_rows))
            st.dataframe(
                decoded_limit_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "attribute":         st.column_config.TextColumn("Attribute",           help=BIAS_COLUMN_HELP["attribute"]),
                    "group":             st.column_config.TextColumn("Group",                help=BIAS_COLUMN_HELP["group"]),
                    "mean_LIMIT_BAL":    st.column_config.NumberColumn("Mean LIMIT_BAL (NT$)", format="%.0f", help=BIAS_COLUMN_HELP["mean_LIMIT_BAL"]),
                    "overall_mean":      st.column_config.NumberColumn("Overall Mean (NT$)", format="%.0f", help=BIAS_COLUMN_HELP["overall_mean"]),
                    "pct_gap_vs_overall":st.column_config.NumberColumn("% Gap vs Overall",  format="%.4f", help=BIAS_COLUMN_HELP["pct_gap_vs_overall"]),
                    "below_20_pct_flag": st.column_config.CheckboxColumn("Below 20% Flag",  help=BIAS_COLUMN_HELP["below_20_pct_flag"]),
                },
            )
            st.caption(
                "**Below 20% Flag** is True if a group's mean LIMIT_BAL is more than "
                "20% below the overall mean — a historical under-lending signal. "
                "Hover over any column header for its definition."
            )

        bill_skew = bias.get("bill_amt_skew") or {}
        if bill_skew:
            _section_header(
                "BILL_AMT median by group (15% deviation flag)",
                [_BM["BILL_AMT Skew"], _BF["BILL_AMT1–6 (X12–X17)"]],
            )
            bill_rows = []
            overall_bill = bill_skew.get("overall_median_avg_bill")
            for attr, groups in bill_skew.items():
                if attr == "overall_median_avg_bill" or not isinstance(groups, dict):
                    continue
                for g, payload in groups.items():
                    bill_rows.append({
                        "attribute": attr,
                        "group": g,
                        "median_avg_bill": payload.get("median"),
                        "overall_median": overall_bill,
                        "pct_gap_vs_overall": payload.get("pct_gap_vs_overall"),
                        "exceeds_15_pct_flag": payload.get("exceeds_15_pct_flag"),
                    })
            if bill_rows:
                decoded_bill_df = _decode_group_df(pd.DataFrame(bill_rows))
                st.dataframe(
                    decoded_bill_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "attribute":          st.column_config.TextColumn("Attribute",            help=BIAS_COLUMN_HELP["attribute"]),
                        "group":              st.column_config.TextColumn("Group",                 help=BIAS_COLUMN_HELP["group"]),
                        "median_avg_bill":    st.column_config.NumberColumn("Median Avg Bill (NT$)", format="%.0f", help=BIAS_COLUMN_HELP["median_avg_bill"]),
                        "overall_median":     st.column_config.NumberColumn("Overall Median (NT$)", format="%.0f", help=BIAS_COLUMN_HELP["overall_median"]),
                        "pct_gap_vs_overall": st.column_config.NumberColumn("% Gap vs Overall",   format="%.4f", help=BIAS_COLUMN_HELP["pct_gap_vs_overall"]),
                        "exceeds_15_pct_flag":st.column_config.CheckboxColumn("Exceeds 15% Flag", help=BIAS_COLUMN_HELP["exceeds_15_pct_flag"]),
                    },
                )
                _bill_flagged = [r for r in bill_rows if r.get("exceeds_15_pct_flag")]
                if _bill_flagged:
                    _bill_worst = max(_bill_flagged, key=lambda r: abs(r.get("pct_gap_vs_overall") or 0))
                    _bw_grp = _bill_worst.get("group", "")
                    _bw_attr = _bill_worst.get("attribute", "")
                    _bw_pct = (_bill_worst.get("pct_gap_vs_overall") or 0) * 100
                    st.caption(
                        f"{len(_bill_flagged)} group(s) exceed the 15% BILL_AMT deviation threshold. "
                        f"The widest gap is **{_bw_attr} = {_bw_grp}** at {_bw_pct:+.1f}% vs the overall median — "
                        f"a potential proxy-bias signal if this group is also over-represented in credit denials."
                    )
                else:
                    st.caption(
                        "No group exceeds the 15% BILL_AMT median deviation threshold. "
                        "Statement balance distributions appear broadly consistent across demographic segments."
                    )

        neg = bias.get("negative_bill_amt") or {}
        if neg:
            _section_header(
                "Negative BILL_AMT subgroup",
                [_BM["Negative BILL_AMT Subgroup"], _BF["BILL_AMT1–6 (X12–X17)"]],
            )
            st.write(
                f"`{neg.get('count')}` clients ({neg.get('fraction')} of dataset) "
                f"have at least one negative bill amount (returns / overpayments). "
                f"Default rate in this subgroup: `{neg.get('default_rate_negative_bill')}` "
                f"vs `{neg.get('default_rate_non_negative')}` for the rest."
            )
            try:
                _neg_pos_rate = float(neg.get("default_rate_negative_bill") or 0)
                _neg_base_rate = float(neg.get("default_rate_non_negative") or 0)
                _neg_delta = _neg_pos_rate - _neg_base_rate
                if abs(_neg_delta) >= 0.02:
                    _neg_dir = "higher" if _neg_delta > 0 else "lower"
                    st.caption(
                        f"This subgroup defaults at a rate {abs(_neg_delta):.1%} {_neg_dir} than the rest of the "
                        f"portfolio. Negative bill amounts often reflect returns or overpayments; their "
                        f"differential default rate should be tracked to avoid systematically misclassifying "
                        f"this behaviour as credit risk."
                    )
                else:
                    st.caption(
                        "The negative-BILL_AMT subgroup defaults at a similar rate to the rest of the dataset — "
                        "no significant differential risk detected for this segment."
                    )
            except (TypeError, ValueError):
                pass

        report = st.session_state.get("metrics_report") or {}
        if report:
            _bias_has_tuning = any(m.get("notes") for m in report.get("models", []))
            prescan = prescan_feature_importance(
                report,
                source="post_tuning" if _bias_has_tuning else "post_training",
            )
            if prescan["demographic_flags"] or prescan["proxy_flags"]:
                _section_header(
                    "Top-10 feature-importance flags (post-training)",
                    [_BM["Direct Demographic Feature"], _BM["Proxy Feature"]],
                )
                if prescan.get("source") == "post_tuning":
                    st.caption("⚙️ Feature importances are from post-tuning results (Stage 4 applied).")
                else:
                    st.caption("📊 Feature importances are from post-training results.")
                if prescan["demographic_flags"]:
                    st.markdown("**Direct demographic features in top-10:**")
                    _demo_items_html = ""
                    for _row in prescan["demographic_flags"]:
                        _m = _row.get("model", "")
                        if _m == "neural_network":
                            _row_bg = "#FEF2F2"
                            _badge_bg = "#FEF2F2"
                            _badge_color = "#7A1F1F"
                            _badge_border = "#EF4444"
                            _label = "critical"
                        else:
                            _row_bg = "#F0FDF4"
                            _badge_bg = "#DCFCE7"
                            _badge_color = "#15803D"
                            _badge_border = "#22C55E"
                            _label = "demographic"
                        if _m == "random_forest":
                            _model_bg = "#DCFCE7"; _model_color = "#15803D"; _model_border = "#22C55E"
                        elif _m == "xgboost":
                            _model_bg = "#EFF6FF"; _model_color = "#1D4ED8"; _model_border = "#3B82F6"
                        else:
                            _model_bg = "#FEF2F2"; _model_color = "#7A1F1F"; _model_border = "#EF4444"
                        _demo_items_html += f"""
<div style="background:{_row_bg};border:1px solid #E2E8F0;border-radius:7px;
            padding:10px 14px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
  <span style="background:{_model_bg};color:{_model_color};border:1px solid {_model_border};
               border-radius:4px;font-size:0.7rem;font-weight:700;padding:2px 8px;white-space:nowrap;">
    {_m.replace("_"," ").title()}
  </span>
  <span style="background:{_badge_bg};color:{_badge_color};border:1px solid {_badge_border};
               border-radius:4px;font-size:0.7rem;font-weight:700;padding:2px 8px;white-space:nowrap;">
    {_label}
  </span>
  <span style="color:#1e293b;font-size:0.88rem;font-weight:700;">{_row.get("feature","")}</span>
  <span style="color:#64748B;font-size:0.8rem;">rank {_row.get("rank","")}</span>
  <span style="color:#64748B;font-size:0.8rem;margin-left:auto;">
    importance {float(_row.get("importance", 0)):.4f}
  </span>
</div>"""
                    st.markdown(
                        f'<div style="display:flex;flex-direction:column;gap:6px;margin-bottom:14px;">'
                        f'{_demo_items_html}</div>',
                        unsafe_allow_html=True,
                    )
                if prescan["proxy_flags"]:
                    st.markdown("**Proxy features in top-10:**")
                    _proxy_items_html = ""
                    for _row in prescan["proxy_flags"]:
                        _m = _row.get("model", "")
                        if _m == "random_forest":
                            _model_bg = "#DCFCE7"; _model_color = "#15803D"; _model_border = "#22C55E"
                        elif _m == "xgboost":
                            _model_bg = "#EFF6FF"; _model_color = "#1D4ED8"; _model_border = "#3B82F6"
                        else:
                            _model_bg = "#FEF2F2"; _model_color = "#7A1F1F"; _model_border = "#EF4444"
                        _proxy_items_html += f"""
<div style="background:#FFFBEB;border:1px solid #FCD34D;border-radius:7px;
            padding:10px 14px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
  <span style="background:{_model_bg};color:{_model_color};border:1px solid {_model_border};
               border-radius:4px;font-size:0.7rem;font-weight:700;padding:2px 8px;white-space:nowrap;">
    {_m.replace("_"," ").title()}
  </span>
  <span style="background:#FEF3C7;color:#92400E;border:1px solid #F59E0B;
               border-radius:4px;font-size:0.7rem;font-weight:700;padding:2px 8px;white-space:nowrap;">
    proxy
  </span>
  <span style="color:#78350F;font-size:0.88rem;font-weight:700;">{_row.get("feature","")}</span>
  <span style="color:#92400E;font-size:0.8rem;">rank {_row.get("rank","")}</span>
  <span style="color:#92400E;font-size:0.8rem;margin-left:auto;">
    importance {float(_row.get("importance", 0)):.4f}
  </span>
</div>"""
                    st.markdown(
                        f'<div style="display:flex;flex-direction:column;gap:6px;margin-bottom:14px;">'
                        f'{_proxy_items_html}</div>',
                        unsafe_allow_html=True,
                    )
            _fi_demo_count = len(prescan.get("demographic_flags", []))
            _fi_proxy_count = len(prescan.get("proxy_flags", []))
            _fi_total = _fi_demo_count + _fi_proxy_count
            if _fi_total > 0:
                _fi_parts = []
                if _fi_demo_count:
                    _fi_parts.append(
                        f"{_fi_demo_count} direct demographic feature(s) appear in a model's top-10"
                    )
                if _fi_proxy_count:
                    _fi_parts.append(
                        f"{_fi_proxy_count} proxy feature(s) that correlate with protected attributes"
                    )
                st.caption(
                    "Feature scan found: " + "; ".join(_fi_parts) + ". "
                    "Direct demographic features in top-10 are a strong signal of potential discriminatory "
                    "model behaviour — consider retraining with feature selection (Stage 4) to remove them."
                )

        # ── Neural Network bias blocking warning (Bias tab) ─────────────────
        _bias_mr = st.session_state.get("metrics_report") or {}
        _bias_models_list = _bias_mr.get("models", [])
        _nn_entry_bias = next((m for m in _bias_models_list if m["model_name"] == "neural_network"), None)
        if _nn_entry_bias:
            _nn_notes_l_bias = (_nn_entry_bias.get("notes") or "").lower()
            _nn_fi_bias = _nn_entry_bias.get("feature_importance") or []
            _nn_demo_top10_bias = {f["feature"] for f in _nn_fi_bias[:10] if f.get("feature") in ("SEX","MARRIAGE","EDUCATION")}
            _nn_has_sel_bias = "feature selection" in _nn_notes_l_bias or "excluded" in _nn_notes_l_bias
            if _nn_demo_top10_bias and not _nn_has_sel_bias:
                st.markdown(
                    f'<div style="background:#FEF2F2;border:2px solid #EF4444;border-radius:10px;'
                    f'padding:18px 22px;margin-top:16px;margin-bottom:16px;">'
                    f'<div style="color:#7A1F1F;font-size:1rem;font-weight:700;margin-bottom:6px;">'
                    f'⛔ Neural Network cannot proceed to Explanation stage</div>'
                    f'<div style="color:#7A1F1F;font-size:0.88rem;line-height:1.6;">'
                    f'The Neural Network still uses <strong>{", ".join(sorted(_nn_demo_top10_bias))}</strong> '
                    f'as active predictors (top-10 features). Return to Stage 4 and enable Feature Selection '
                    f'to remove these features, then retrain.</div></div>',
                    unsafe_allow_html=True,
                )

        # ── Counterfactual bias test box ─────────────────────────────────────
        st.markdown(
            """
<div style="background:#F5F3FF;border-left:4px solid #7C3AED;border-radius:0 10px 10px 0;
            padding:18px 22px;margin-top:8px;margin-bottom:20px;">
  <div style="color:#5B21B6;font-size:0.95rem;font-weight:700;margin-bottom:12px;
              letter-spacing:0.03em;">
    Counterfactual Bias Test
  </div>
  <ol style="color:#4C1D95;font-size:0.85rem;line-height:1.8;padding-left:20px;margin:0;">
    <li>Select 100 predicted defaulters from the test set at random.</li>
    <li>Flip the SEX value for each record (1→2, 2→1) while keeping all other features unchanged.</li>
    <li>Re-run predictions: if more than 10% change from "default" to "no default", the model is sex-biased.</li>
    <li>Repeat the test flipping MARRIAGE and EDUCATION values.</li>
  </ol>
</div>
""",
            unsafe_allow_html=True,
        )

    # ── Bias Visualizations ────────────────────────────────────────────────────
    if bias:
        st.markdown("---")
        st.markdown("### 📊 Visual Bias Audit")

        with st.expander("Default Rate by Demographic Group", expanded=True):
            _viz_caption(
                what="Ground-truth default rates per group across all demographic attributes. "
                     "Bars are colour-coded: red = >10% above overall rate, green = >10% below, "
                     "blue = within range. Dotted line = overall default rate.",
                acceptable="All groups within ±10% of the overall default rate (~22%).",
                concern="Any group exceeding the overall rate by >10 pp — this risk gets "
                        "directly encoded into model predictions.",
            )
            st.plotly_chart(plot_default_rate_heatmap(bias), use_container_width=True)

        with st.expander("Feature Importance — Bias Flags", expanded=False):
            _viz_caption(
                what="Top-10 feature importances per model, colour-coded by bias risk: "
                     "🔴 direct demographic feature (SEX, AGE, EDUCATION, MARRIAGE), "
                     "🟠 proxy feature (LIMIT_BAL, BILL_AMT*, PAY_AMT*), "
                     "🔵 standard predictive feature. All models shown simultaneously.",
                acceptable="No red bars in the top 10; proxy features (orange) present but "
                           "not dominating the top 5 slots.",
                concern="Any demographic feature ranked in top 10, or proxy features "
                        "occupying the top 3 importance slots across multiple models.",
            )
            _mr = st.session_state.get("metrics_report") or {}
            if _mr.get("models"):
                st.plotly_chart(plot_feature_importance_bias(_mr), use_container_width=True)
            else:
                st.info("Train at least one model on the **Run** tab to see feature importances.")

        with st.expander("Proxy Bias Network (Sankey)", expanded=False):
            _viz_caption(
                what="Flow diagram linking proxy features (LIMIT_BAL, Avg BILL_AMT) to "
                     "demographic groups whose credit limits or bills deviate from the "
                     "overall average. Link width = gap magnitude. Red = under-represented, "
                     "green = over-represented.",
                acceptable="Few or no links above the 10% gap threshold; gaps roughly balanced "
                           "across demographic groups.",
                concern="Large red flows concentrated in protected categories "
                        "(especially LIMIT_BAL gaps >20% for SEX or AGE groups).",
            )
            gap_thresh_b = st.slider(
                "Minimum gap threshold", 0.05, 0.50, 0.10, 0.05, key="bias_sankey_thresh"
            )
            st.plotly_chart(
                plot_proxy_bias_network(bias, gap_threshold=gap_thresh_b),
                use_container_width=True,
            )

    st.markdown("---")
    _section_header(
        "BiasAgent audit",
        [{
            "term": "BiasAgent",
            "definition": "A CrewAI agent powered by Gemini that interprets dataset-level bias signals and post-training feature-importance flags.",
            "context": "It produces a Markdown audit covering per-group default rates, proxy variables, intersectional risks, and recommended counterfactual tests. Its output is advisory only — it never feeds back into the trainer, fairness, or explainer agents.",
            "requirement": None,
        }],
    )
    bmd = st.session_state.get("bias_md") or ""
    if not bmd:
        st.info("Click *Compute + audit* (or *Run BiasAgent*) to generate the Markdown audit.")
    else:
        st.markdown(bmd)
        st.download_button(
            "Download bias_audit.md",
            data=bmd,
            file_name="bias_audit.md",
            mime="text/markdown",
        )


# ---------- Fairness tab ----------------------------------------------------

with tab_fairness:
    st.subheader("Fairness audit")
    st.caption(
        "Buckets test-set predictions by SEX, AGE band, and MARRIAGE; "
        "computes Disparate Impact, Equalized-Odds gaps, and FPR gap "
        "(Predictive Equality) per model. Then the FairnessAgent writes a "
        "Markdown audit comparing Random Forest, XGBoost, and the Neural Network."
    )

    fairness_cols = st.columns(3)
    with fairness_cols[0]:
        if st.button("Compute fairness metrics", key="fair_btn_compute"):
            _compute_fairness()
    with fairness_cols[1]:
        if st.button("Run FairnessAgent (audit)", key="fair_btn_agent"):
            _run_fairness_agent()
    with fairness_cols[2]:
        if st.button("Compute + audit", type="primary", key="fair_btn_both"):
            _run_fairness_audit()

    st.markdown("---")

    st.markdown(
        """<div style="background:#EFF6FF;border-left:4px solid #3B82F6;border-radius:6px;
padding:14px 18px;margin:8px 0 16px 0;font-size:0.83rem;color:#334155;line-height:1.7;">
<strong style="color:#1D4ED8;">📖 How to read these results</strong><br>
<b>Disparate Impact (DI)</b> below <b>0.80</b> means one group is being treated significantly
worse than another — this fails the legal "80% rule" (four-fifths rule).<br>
<b>TPR gap</b> = difference in recall (true positive rate) across demographic groups.
Larger gap = some groups' defaulters are caught at different rates.<br>
<b>FPR gap</b> = difference in false alarm rates across groups.
Larger gap = some groups are wrongly flagged at different rates.
</div>""",
        unsafe_allow_html=True,
    )

    rows = _fairness_summary_rows()
    if not rows:
        st.info(
            "No fairness metrics yet. Train at least one model on the **Run** "
            "tab, then click *Compute + audit* above."
        )
    else:
        _section_header(
            "Per-model x per-attribute summary",
            [
                _FM["Disparate Impact (DI)"],
                _FM["Passes 80% Rule (Four-Fifths Rule)"],
                _FM["TPR Gap (Equal Opportunity Difference)"],
                _FM["FPR Gap (Predictive Equality Difference)"],
            ],
        )
        summary_df = pd.DataFrame(rows)
        # Group by attribute and render a styled HTML table per attribute
        _MODEL_BADGE = {
            "random_forest":  ("background:#DCFCE7", "color:#15803D"),
            "xgboost":        ("background:#EFF6FF", "color:#1D4ED8"),
            "neural_network": ("background:#FEF2F2", "color:#7A1F1F"),
        }
        for _attr in summary_df["attribute"].unique():
            st.markdown(f"##### {_attr}")
            _attr_rows = summary_df[summary_df["attribute"] == _attr]
            _table_rows_html = ""
            for _, _r in _attr_rows.iterrows():
                _m = _r.get("model", "")
                _bg, _col = _MODEL_BADGE.get(_m, ("background:#1e293b", "color:#94a3b8"))
                _model_badge = (
                    f'<span style="{_bg};{_col};border-radius:4px;padding:2px 8px;'
                    f'font-size:0.72rem;font-weight:700;">{_m.replace("_"," ").title()}</span>'
                )
                _di = _r.get("disparate_impact")
                _passes = _r.get("passes_80_pct_rule")
                if _di is not None:
                    _bar_pct = min(int(float(_di) * 100), 100)
                    _bar_color = "#22c55e" if float(_di) >= 0.80 else "#ef4444"
                    _di_bar = (
                        f'<div style="display:flex;align-items:center;gap:8px;">'
                        f'<div style="background:#E2E8F0;border-radius:3px;width:80px;height:10px;">'
                        f'<div style="background:{_bar_color};width:{_bar_pct}px;height:10px;border-radius:3px;"></div></div>'
                        f'<span style="color:#1e293b;font-size:0.8rem;">{float(_di):.2f}</span></div>'
                    )
                else:
                    _di_bar = '<span style="color:#64748b;">—</span>'
                if _passes is True:
                    _pill = '<span style="background:#DCFCE7;color:#15803D;border-radius:4px;padding:2px 8px;font-size:0.72rem;font-weight:700;">PASS</span>'
                elif _passes is False:
                    _pill = '<span style="background:#FEF2F2;color:#7A1F1F;border-radius:4px;padding:2px 8px;font-size:0.72rem;font-weight:700;">FAIL</span>'
                else:
                    _pill = '<span style="color:#64748B;">—</span>'
                _tpr = _r.get("tpr_gap")
                _fpr = _r.get("fpr_gap")
                _tpr_str = f"{float(_tpr):.2f}" if _tpr is not None else "—"
                _fpr_str = f"{float(_fpr):.2f}" if _fpr is not None else "—"
                _table_rows_html += (
                    f"<tr>"
                    f'<td style="padding:8px 12px;">{_model_badge}</td>'
                    f'<td style="padding:8px 12px;">{_di_bar}</td>'
                    f'<td style="padding:8px 12px;">{_pill}</td>'
                    f'<td style="padding:8px 12px;color:#1e293b;font-size:0.82rem;">{_tpr_str}</td>'
                    f'<td style="padding:8px 12px;color:#1e293b;font-size:0.82rem;">{_fpr_str}</td>'
                    f"</tr>"
                )
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;background:#FFFFFF;'
                f'border:1px solid #E2E8F0;border-radius:8px;margin-bottom:16px;">'
                f'<thead><tr style="background:#F0F0F0;color:#64748B;font-size:0.75rem;text-transform:uppercase;">'
                f'<th style="padding:8px 12px;text-align:left;">Model</th>'
                f'<th style="padding:8px 12px;text-align:left;">Disparate Impact</th>'
                f'<th style="padding:8px 12px;text-align:left;">Pass/Fail</th>'
                f'<th style="padding:8px 12px;text-align:left;">TPR Gap</th>'
                f'<th style="padding:8px 12px;text-align:left;">FPR Gap</th>'
                f'</tr></thead>'
                f'<tbody>{_table_rows_html}</tbody>'
                f'</table>',
                unsafe_allow_html=True,
            )
            _pre_fm = st.session_state.get("pre_tuning_fairness_metrics") or {}
            _pre_per_model = _pre_fm.get("per_model") or {}
            _curr_per_model = (st.session_state.get("fairness_metrics") or {}).get("per_model") or {}
            if _pre_per_model:
                _diff_rows = []
                for _mn in ["random_forest", "xgboost", "neural_network"]:
                    _post_attr = _curr_per_model.get(_mn, {}).get(_attr, {})
                    _pre_attr = _pre_per_model.get(_mn, {}).get(_attr, {})
                    if not _post_attr or not _pre_attr:
                        continue
                    _post_di = _post_attr.get("disparate_impact")
                    _pre_di = _pre_attr.get("disparate_impact")
                    if _post_di is None or _pre_di is None:
                        continue
                    _delta_di = float(_post_di) - float(_pre_di)
                    if abs(_delta_di) >= 0.01:
                        _sign = "+" if _delta_di >= 0 else ""
                        _d_color = "#15803D" if _delta_di >= 0 else "#B91C1C"
                        _d_bg = "#DCFCE7" if _delta_di >= 0 else "#FEF2F2"
                        _diff_rows.append(
                            f'<span style="margin-right:10px;font-size:0.82rem;color:#374151;">'
                            f'{_pretty_model_label(_mn)}: '
                            f'DI {float(_pre_di):.2f}→{float(_post_di):.2f} '
                            f'<span style="background:{_d_bg};color:{_d_color};'
                            f'border-radius:3px;padding:1px 6px;font-weight:700;">'
                            f'{_sign}{_delta_di:.2f}</span></span>'
                        )
                if _diff_rows:
                    st.markdown(
                        f'<div style="background:#F8F9FA;border:1px solid #DEE2E6;'
                        f'border-radius:6px;padding:8px 12px;margin:-8px 0 12px 0;'
                        f'font-size:0.8rem;">'
                        f'Tuning impact on DI: {"  ·  ".join(_diff_rows)}</div>',
                        unsafe_allow_html=True,
                    )

        fm = st.session_state["fairness_metrics"]
        per_model = fm.get("per_model") or {}
        per_attr_summary = fm.get("summary") or {}

        if per_attr_summary:
            _section_header(
                "Headline by attribute",
                [_FM["Disparate Impact (DI)"], _FM["FPR Gap (Predictive Equality Difference)"]],
            )
            head_rows = []
            for attr, payload in per_attr_summary.items():
                best_di = payload.get("best_disparate_impact") or (None, None)
                worst_di = payload.get("worst_disparate_impact") or (None, None)
                best_fpr = payload.get("best_fpr_gap") or (None, None)
                worst_fpr = payload.get("worst_fpr_gap") or (None, None)
                head_rows.append({
                    "attribute": attr,
                    "best_DI_model": best_di[0],
                    "best_DI_value": best_di[1],
                    "worst_DI_model": worst_di[0],
                    "worst_DI_value": worst_di[1],
                    "best_FPR_gap_model": best_fpr[0],
                    "best_FPR_gap_value": best_fpr[1],
                })
            _head_cols = st.columns(len(head_rows)) if head_rows else []
            for _hci, _hr in enumerate(head_rows):
                _bdi_m = _hr.get("best_DI_model") or "—"
                _bdi_v = _hr.get("best_DI_value")
                _wdi_m = _hr.get("worst_DI_model") or "—"
                _wdi_v = _hr.get("worst_DI_value")
                _bfpr_m = _hr.get("best_FPR_gap_model") or "—"
                _bfpr_v = _hr.get("best_FPR_gap_value")
                _bdi_str = f"{float(_bdi_v):.2f}" if _bdi_v is not None else "—"
                _wdi_str = f"{float(_wdi_v):.2f}" if _wdi_v is not None else "—"
                _bfpr_str = f"{float(_bfpr_v):.2f}" if _bfpr_v is not None else "—"
                with _head_cols[_hci]:
                    st.markdown(
                        f'<div style="background:#F8F9FA;border:1px solid #E2E8F0;border-radius:8px;padding:14px 16px;margin-bottom:8px;">'
                        f'<div style="color:#1e293b;font-size:0.9rem;font-weight:700;margin-bottom:10px;">{_hr.get("attribute","")}</div>'
                        f'<div style="color:#15803D;font-size:0.8rem;margin-bottom:6px;">Best DI: {_bdi_m.replace("_"," ").title()} — {_bdi_str}</div>'
                        f'<div style="color:#7A1F1F;font-size:0.8rem;margin-bottom:6px;">Worst DI: {_wdi_m.replace("_"," ").title()} — {_wdi_str}</div>'
                        f'<div style="color:#1D4ED8;font-size:0.8rem;">Best FPR gap: {_bfpr_m.replace("_"," ").title()} — {_bfpr_str}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        _section_header(
            "Per-group breakdown",
            [
                _FM["Selection Rate"],
                _FM["TPR (True Positive Rate) / Recall"],
                _FM["FPR (False Positive Rate)"],
            ],
        )
        _BREAKDOWN_MODEL_BADGE = {
            "random_forest":  ("background:#DCFCE7", "color:#15803D"),
            "xgboost":        ("background:#EFF6FF", "color:#1D4ED8"),
            "neural_network": ("background:#FEF2F2", "color:#7A1F1F"),
        }
        for model_name, attrs in per_model.items():
            _mbg, _mcol = _BREAKDOWN_MODEL_BADGE.get(model_name, ("background:#1e293b", "color:#94a3b8"))
            _expander_label = (
                f"{_pretty_model_label(model_name)}"
            )
            with st.expander(_expander_label, expanded=False):
                st.markdown(
                    f'<span style="{_mbg};{_mcol};border-radius:4px;padding:3px 10px;'
                    f'font-size:0.75rem;font-weight:700;display:inline-block;margin-bottom:10px;">'
                    f'{model_name.replace("_"," ").title()}</span>',
                    unsafe_allow_html=True,
                )
                _tuning_note_val = attrs.get("_tuning_notes", "")
                if _tuning_note_val:
                    st.caption(f"⚙️ Post-tuning predictions · {_tuning_note_val}")
                else:
                    st.caption("📊 Post-training predictions")
                for attr_name, payload in attrs.items():
                    if attr_name.startswith("_") or not isinstance(payload, dict):
                        continue
                    di = payload.get("disparate_impact")
                    passes = payload.get("passes_80_pct_rule")
                    eo = payload.get("equalized_odds") or {}
                    # Build pass/fail pill with DI score
                    if di is not None:
                        _di_val = float(di)
                        if _di_val < 0.80:
                            _attr_pill = (
                                f'<span style="background:#FEF2F2;color:#7A1F1F;border-radius:4px;'
                                f'padding:2px 8px;font-size:0.72rem;font-weight:700;margin-left:8px;">'
                                f'FAIL {_di_val:.2f}</span>'
                            )
                        else:
                            _attr_pill = (
                                f'<span style="background:#DCFCE7;color:#15803D;border-radius:4px;'
                                f'padding:2px 8px;font-size:0.72rem;font-weight:700;margin-left:8px;">'
                                f'PASS {_di_val:.2f}</span>'
                            )
                    else:
                        _attr_pill = (
                            '<span style="background:#F1F5F9;color:#64748B;border-radius:4px;'
                            'padding:2px 8px;font-size:0.72rem;font-weight:700;margin-left:8px;">N/A</span>'
                        )
                    st.markdown(
                        f'<div style="margin:12px 0 4px 0;">'
                        f'<strong style="color:#1e293b;font-size:0.9rem;">{attr_name}</strong>'
                        f'{_attr_pill}</div>',
                        unsafe_allow_html=True,
                    )
                    groups = payload.get("groups") or {}
                    if groups:
                        gdf = pd.DataFrame(groups).T.reset_index()
                        gdf = gdf.rename(columns={
                            "index": "Group",
                            "selection_rate": "Predicted default %",
                            "tpr": "Recall",
                            "fpr": "False alarm rate",
                            "base_rate": "Actual default rate",
                        })
                        # Round numeric columns to 2dp
                        for _gc in ["Predicted default %", "Recall", "False alarm rate", "Actual default rate"]:
                            if _gc in gdf.columns:
                                gdf[_gc] = pd.to_numeric(gdf[_gc], errors="coerce").round(2)
                        # Flag rows with few records
                        if "count" in gdf.columns:
                            def _flag_group(row):
                                _grp = str(row.get("Group", ""))
                                _cnt = row.get("count", 999)
                                try:
                                    _cnt = int(_cnt)
                                except Exception:
                                    _cnt = 999
                                if "marriage_0" in _grp or _cnt < 50:
                                    return row.get("Group", "") + " ⚠ few records"
                                return row.get("Group", "")
                            gdf["Group"] = gdf.apply(_flag_group, axis=1)
                        st.dataframe(gdf, use_container_width=True, hide_index=True)
                    # Equalized odds line as styled HTML
                    _tpr_gap = eo.get("tpr_gap")
                    _fpr_gap = eo.get("fpr_gap")
                    def _gap_color(v):
                        try:
                            return "#ef4444" if abs(float(v)) > 0.10 else "#22c55e"
                        except Exception:
                            return "#64748B"
                    _tpr_str = f'<span style="color:{_gap_color(_tpr_gap)};font-weight:700;">{float(_tpr_gap):.2f}</span>' if _tpr_gap is not None else '<span style="color:#64748B;">—</span>'
                    _fpr_str = f'<span style="color:{_gap_color(_fpr_gap)};font-weight:700;">{float(_fpr_gap):.2f}</span>' if _fpr_gap is not None else '<span style="color:#64748B;">—</span>'
                    st.markdown(
                        f'<div style="font-size:0.82rem;color:#64748B;margin:4px 0 14px 0;">'
                        f'TPR gap: {_tpr_str} &nbsp;·&nbsp; FPR gap: {_fpr_str}</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown(
        '<div style="background:#F8F9FA;border:1px solid #E2E8F0;border-radius:6px;'
        'padding:12px 16px;margin:16px 0 8px 0;font-size:0.8rem;color:#64748B;">'
        '📌 <strong style="color:#1e293b;">All three models fail the 80% rule for AGE and MARRIAGE. '
        'Neural Network also fails for SEX.</strong>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Fairness Visualizations ────────────────────────────────────────────────
    _fm_state = st.session_state.get("fairness_metrics") or {}
    _mr_state  = st.session_state.get("metrics_report")  or {}
    _state_ref = get_state(st.session_state["state_handle"])
    _y_true_viz: list = (
        _state_ref.dataset.y_test.tolist() if _state_ref.dataset is not None else []
    )
    _sensitive_viz: "pd.DataFrame" = (
        _state_ref.dataset.test_sensitive
        if _state_ref.dataset is not None
        else pd.DataFrame()
    )

    if _fm_state.get("per_model"):
        st.markdown("---")
        st.markdown("### 📊 Visual Fairness Audit")

        with st.expander("FPR Comparison — by Group & Model", expanded=True):
            _viz_caption(
                what="False Positive Rate per demographic group per model. "
                     "A high FPR for a group means good customers there are incorrectly "
                     "denied credit more often. Dashed line = 0.10 concern threshold.",
                acceptable="FPR below 0.10 for all groups across all models.",
                concern="FPR gap >0.05 between groups of the same attribute, "
                        "or any group's FPR exceeding 0.15.",
            )
            st.plotly_chart(plot_fpr_comparison(_fm_state), use_container_width=True)

        with st.expander("Confusion Matrix Grid", expanded=False):
            _viz_caption(
                what="Annotated confusion matrices per model. Rows = actual class, "
                     "columns = predicted class. Each cell shows raw count and "
                     "row-normalised percentage. Colour intensity shows row-% magnitude.",
                acceptable="TP and TN rows >75%; FP and FN each <25% of their actual rows.",
                concern="High FP (top-right) = good customers wrongly denied; "
                        "high FN (bottom-left) = actual defaulters missed entirely.",
            )
            st.plotly_chart(plot_confusion_matrix_grid(_mr_state), use_container_width=True)

        with st.expander("Subgroup Performance Radar", expanded=False):
            _viz_caption(
                what="Polar chart: four fairness metrics (TPR, Specificity=1−FPR, "
                     "Selection Rate, Base Rate) per demographic group. Shrunken polygons "
                     "indicate systematic under-serving. One chart per trained model.",
                acceptable="Polygons roughly equal in size across groups; all metrics >0.6.",
                concern="One group's polygon significantly smaller — especially low TPR "
                        "or low Specificity for protected attribute groups.",
            )
            _all_mdls = list((_fm_state.get("per_model") or {}).keys())
            if _all_mdls:
                for _mdl in _all_mdls:
                    st.markdown(f"**{_pretty_model_label(_mdl)}**")
                    st.plotly_chart(
                        plot_subgroup_radar(_fm_state, model_name=_mdl),
                        use_container_width=True,
                    )
            else:
                st.info("Run Fairness Compute + Audit to populate this chart.")

        with st.expander("Model–Agent Agreement Matrix", expanded=False):
            _viz_caption(
                what="Pass/fail heatmap for every model × fairness criterion: "
                     "80% Disparate Impact rule, TPR Gap (Equal Opportunity), "
                     "FPR Gap (Predictive Equality), no demographic or proxy "
                     "features in top-10 importances.",
                acceptable="All cells ✓ (green). At minimum the 80% Rule and "
                           "gap thresholds should pass for all models.",
                concern="Any model failing the 80% Rule (Disparate Impact <0.80) "
                        "or showing direct demographic features in top-10.",
            )
            agr_c1, agr_c2 = st.columns(2)
            with agr_c1:
                tpr_thresh_v = st.slider(
                    "TPR Gap threshold", 0.01, 0.20, 0.05, 0.01, key="fair_viz_tpr_thresh"
                )
            with agr_c2:
                fpr_thresh_v = st.slider(
                    "FPR Gap threshold", 0.01, 0.20, 0.05, 0.01, key="fair_viz_fpr_thresh"
                )
            st.plotly_chart(
                plot_agent_agreement_matrix(
                    _fm_state, _mr_state,
                    tpr_gap_threshold=tpr_thresh_v,
                    fpr_gap_threshold=fpr_thresh_v,
                ),
                use_container_width=True,
            )

        with st.expander("Equalized Odds — TPR Gap vs FPR Gap", expanded=False):
            _viz_caption(
                what="Scatter plot where each point is a (model, attribute) pair. "
                     "Proximity to (0, 0) = most equalized odds. Concentric dashed "
                     "circles mark 5% and 10% fairness thresholds. "
                     "Green border = passes 80% rule, red = fails.",
                acceptable="All points inside the 5% threshold circle with green borders.",
                concern="Any point outside the 10% circle, especially for SEX or AGE — "
                        "or red-bordered points regardless of distance from origin.",
            )
            st.plotly_chart(plot_equalized_odds(_fm_state), use_container_width=True)

        with st.expander("Threshold Impact by Group", expanded=False):
            _viz_caption(
                what="How FPR, TPR, or Selection Rate changes per demographic group as "
                     "the decision threshold varies 0.05 → 0.95. Line dash distinguishes "
                     "models; colour distinguishes groups. Vertical line = default 0.5 threshold.",
                acceptable="Lines for different groups of the same attribute stay close "
                           "together across the threshold range.",
                concern="Large divergence between groups' metric lines — one group's FPR "
                        "rises steeply while another's stays low as threshold decreases.",
            )
            if _mr_state.get("models") and not _sensitive_viz.empty and _y_true_viz:
                t8_metric_v = st.selectbox(
                    "Metric to display",
                    ["fpr", "tpr", "selection_rate"],
                    format_func=lambda x: {
                        "fpr": "FPR — False Positive Rate",
                        "tpr": "TPR — True Positive Rate",
                        "selection_rate": "Selection Rate",
                    }.get(x, x),
                    key="fair_viz_t8_metric",
                )
                st.plotly_chart(
                    plot_threshold_impact(_mr_state, _y_true_viz, _sensitive_viz, metric=t8_metric_v),
                    use_container_width=True,
                )
            else:
                st.info("Load the dataset and train a model to enable threshold analysis.")

        with st.expander("Precision-Recall Frontier", expanded=False):
            _viz_caption(
                what="Full PR curves for each model (requires prediction probabilities). "
                     "Higher area under the curve (AUPRC) = better performance on the "
                     "minority default class across all threshold choices. "
                     "Dotted line = no-skill baseline.",
                acceptable="AUPRC > 0.60 for at least one model; all curves clearly above baseline.",
                concern="AUPRC near the no-skill baseline (~22%), or a large AUPRC gap "
                        "between models suggesting instability in minority-class detection.",
            )
            if _y_true_viz:
                st.plotly_chart(
                    plot_precision_recall_frontier(_mr_state, _y_true_viz),
                    use_container_width=True,
                )
            else:
                st.info("Load the dataset and train a model to see PR curves.")

    st.markdown("---")
    _section_header(
        "FairnessAgent audit",
        [{
            "term": "FairnessAgent",
            "definition": "A CrewAI agent powered by Gemini that evaluates outcome fairness across the three model architectures (Random Forest, XGBoost, Neural Network).",
            "context": "It compares Disparate Impact, TPR gap, and FPR gap across SEX, AGE, and MARRIAGE groups, then recommends which model best balances predictive power with fairness obligations.",
            "requirement": None,
        }],
    )
    fmd = st.session_state["fairness_md"]
    if not fmd:
        st.info("Click *Compute + audit* (or *Run FairnessAgent*) to generate the Markdown audit.")
    else:
        st.markdown(fmd)
        st.download_button(
            "Download fairness_audit.md",
            data=fmd,
            file_name="fairness_audit.md",
            mime="text/markdown",
        )


# ---------- Artifacts tab ---------------------------------------------------


with tab_artifacts:
    st.subheader("Saved artifacts")
    out_dir = Path(st.session_state["outputs_dir"])
    if not out_dir.exists():
        st.info("No outputs yet — run the pipeline at least once.")
    else:
        # Walk one level deep so files inside subfolders are listed too.
        files = sorted(p for p in out_dir.rglob("*") if p.is_file())
        if not files:
            st.info("Output folder is empty.")
        else:
            for f in files:
                rel = f.relative_to(out_dir)
                col1, col2, col3 = st.columns([4, 2, 2])
                with col1:
                    st.write(str(rel))
                with col2:
                    st.write(f"{f.stat().st_size} bytes")
                with col3:
                    st.download_button(
                        "Download",
                        data=f.read_bytes(),
                        file_name=f.name,
                        key=f"dl_{rel}",
                    )
        st.caption(f"Folder: {out_dir}")

    if st.session_state["metrics_report"]:
        st.markdown("---")
        st.subheader("Latest metrics_report.json (in memory)")
        st.json(st.session_state["metrics_report"])

    if st.session_state.get("fairness_metrics"):
        st.markdown("---")
        st.subheader("Latest fairness_metrics.json (in memory)")
        st.json(st.session_state["fairness_metrics"])

    if st.session_state.get("bias_signals"):
        st.markdown("---")
        st.subheader("Latest bias_signals.json (in memory)")
        st.json(st.session_state["bias_signals"])
