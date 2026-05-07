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
from typing import Any, Dict

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
        "dataset_summary": {},
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
    fairness = compute_fairness_for_all_models(
        metrics_report=state.metrics_report,
        y_true=state.dataset.y_test,
        sensitive_df=state.dataset.test_sensitive,
    )
    state.fairness_metrics = fairness
    st.session_state["fairness_metrics"] = fairness

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
    prescan = prescan_feature_importance(metrics) if metrics else None

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


def _pickle_path_for(model_name: str) -> Path | None:
    """Return the on-disk pickle path for a model, or None if missing."""
    p = Path(st.session_state["outputs_dir"]) / "models" / f"{model_name}.pkl"
    return p if p.exists() else None


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

    # ---- Stage 2: Train Model ---------------------------------------------
    st.subheader("Stage 2 — Train model")
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

    # Pickle download row for trained models.
    report = st.session_state["metrics_report"]
    if report and report.get("models"):
        st.markdown("**Download trained models (.pkl):**")
        dl_cols = st.columns(len(SUPPORTED_MODELS))
        for i, mname in enumerate(SUPPORTED_MODELS):
            with dl_cols[i]:
                pkl = _pickle_path_for(mname)
                if pkl is None:
                    st.caption(f"{_pretty_model_label(mname)} — not yet trained")
                else:
                    st.download_button(
                        f"⬇ {_pretty_model_label(mname)}.pkl",
                        data=pkl.read_bytes(),
                        file_name=f"{mname}.pkl",
                        mime="application/octet-stream",
                        key=f"dl_{mname}_pkl",
                    )

        st.markdown(f"**Best model so far:** `{report.get('best_model')}`")
        leaderboard_df = pd.DataFrame(report["leaderboard"])
        st.markdown("**Leaderboard**")
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

        st.markdown("**Per-model details**")
        for m in report["models"]:
            with st.expander(f"{m['model_name']}", expanded=False):
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    metrics_row = _format_metric_row(m)
                    st.write("Metrics")
                    st.dataframe(
                        pd.DataFrame([metrics_row]).T.rename(columns={0: "value"}),
                        use_container_width=True,
                    )
                    st.write("Confusion matrix (rows = actual, cols = predicted)")
                    cm = m.get("confusion_matrix") or [[0, 0], [0, 0]]
                    cm_df = pd.DataFrame(
                        cm,
                        index=["actual_0", "actual_1"],
                        columns=["pred_0", "pred_1"],
                    )
                    st.dataframe(cm_df, use_container_width=True)
                with col_b:
                    st.write("Top feature importances")
                    fi = m.get("feature_importance") or []
                    if fi:
                        fi_df = pd.DataFrame(fi).set_index("feature")[["importance"]]
                        st.bar_chart(fi_df, horizontal=False)
                    else:
                        st.write("(none)")
                if m.get("notes"):
                    st.caption(m["notes"])
                if m.get("hyperparameters"):
                    st.write("Hyperparameters used:")
                    st.json(m["hyperparameters"])
    else:
        st.info("No models trained yet — click one of the buttons above.")

    st.markdown("---")

    # ---- Stage 3: Tune Model ----------------------------------------------
    st.subheader("Stage 3 — Tune model")
    st.caption(
        "Apply class-imbalance and hyperparameter-search techniques to a "
        "specific model. The tuned result replaces the existing entry for "
        "that model in the leaderboard above."
    )
    if not (report and report.get("models")):
        st.info("Train at least one model in Stage 2 before tuning.")
    else:
        for mname in SUPPORTED_MODELS:
            with st.expander(f"Tune {_pretty_model_label(mname)}",
                             expanded=False):
                t_cols = st.columns(3)

                # --- SMOTE ------------------------------------------------
                with t_cols[0]:
                    st.markdown("**SMOTE**")
                    k = st.number_input(
                        "k_neighbors", 1, 20, 5, 1,
                        key=f"smote_k_{mname}",
                        help="Number of nearest neighbours used to synthesise minority samples.",
                    )
                    if st.button("Run SMOTE", key=f"smote_btn_{mname}"):
                        _apply_optimization(mname, "smote",
                                            {"k_neighbors": int(k)})

                # --- Focal Loss -------------------------------------------
                with t_cols[1]:
                    st.markdown("**Focal Loss**")
                    alpha = st.number_input(
                        "alpha", 0.0, 1.0, 0.25, 0.05,
                        key=f"focal_a_{mname}",
                        help="Class-balance weight (typical 0.25).",
                    )
                    gamma = st.number_input(
                        "gamma", 0.0, 5.0, 2.0, 0.25,
                        key=f"focal_g_{mname}",
                        help="Focusing parameter (typical 2.0).",
                    )
                    if st.button("Run Focal Loss", key=f"focal_btn_{mname}"):
                        _apply_optimization(mname, "focal_loss",
                                            {"alpha": float(alpha),
                                             "gamma": float(gamma)})

                # --- GridSearch -------------------------------------------
                with t_cols[2]:
                    st.markdown("**GridSearch CV**")
                    cv = st.number_input(
                        "CV folds", 2, 10, 3, 1,
                        key=f"gs_cv_{mname}",
                        help="Number of cross-validation folds.",
                    )
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
                            {"cv_folds": int(cv), "scoring": scoring,
                             "param_grid": grid},
                        )

    st.markdown("---")

    # ---- Stage 4: Explain Model -------------------------------------------
    st.subheader("Stage 4 — Explain model")
    if not (report and report.get("models")):
        st.info("Train at least one model in Stage 2 to enable the explainer.")
    else:
        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("Generate / refresh explanation",
                         type="primary", key="explain_btn"):
                _run_explainer()
        with action_cols[1]:
            if st.button("Run Fairness Audit", key="fair_audit_btn_run"):
                _run_fairness_audit()
        if st.session_state.get("explanation_md"):
            st.caption("Latest explainer brief is rendered on the **Explanation** tab.")
        if st.session_state.get("fairness_md"):
            st.caption("Latest fairness audit is rendered on the **Fairness** tab.")


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

        report = st.session_state.get("metrics_report") or {}
        if report:
            prescan = prescan_feature_importance(report)
            if prescan["demographic_flags"] or prescan["proxy_flags"]:
                _section_header(
                    "Top-10 feature-importance flags (post-training)",
                    [_BM["Direct Demographic Feature"], _BM["Proxy Feature"]],
                )
                if prescan["demographic_flags"]:
                    st.markdown("**Direct demographic features in top-10:**")
                    st.dataframe(
                        pd.DataFrame(prescan["demographic_flags"]),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "model":      st.column_config.TextColumn("Model",      help=BIAS_COLUMN_HELP["model"]),
                            "feature":    st.column_config.TextColumn("Feature",    help=BIAS_COLUMN_HELP["feature"]),
                            "rank":       st.column_config.NumberColumn("Rank",     help=BIAS_COLUMN_HELP["rank"]),
                            "importance": st.column_config.NumberColumn("Importance", format="%.4f", help=BIAS_COLUMN_HELP["importance"]),
                        },
                    )
                if prescan["proxy_flags"]:
                    st.markdown("**Proxy features in top-10:**")
                    st.dataframe(
                        pd.DataFrame(prescan["proxy_flags"]),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "model":      st.column_config.TextColumn("Model",      help=BIAS_COLUMN_HELP["model"]),
                            "feature":    st.column_config.TextColumn("Feature",    help=BIAS_COLUMN_HELP["feature"]),
                            "rank":       st.column_config.NumberColumn("Rank",     help=BIAS_COLUMN_HELP["rank"]),
                            "importance": st.column_config.NumberColumn("Importance", format="%.4f", help=BIAS_COLUMN_HELP["importance"]),
                        },
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
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "model": st.column_config.TextColumn(
                    "Model", help=COLUMN_HELP["model"]
                ),
                "attribute": st.column_config.TextColumn(
                    "Attribute", help=COLUMN_HELP["attribute"]
                ),
                "disparate_impact": st.column_config.NumberColumn(
                    "Disparate Impact",
                    format="%.4f",
                    help=COLUMN_HELP["disparate_impact"],
                ),
                "passes_80_pct_rule": st.column_config.CheckboxColumn(
                    "Passes 80% Rule",
                    help=COLUMN_HELP["passes_80_pct_rule"],
                ),
                "tpr_gap": st.column_config.NumberColumn(
                    "TPR Gap",
                    format="%.4f",
                    help=COLUMN_HELP["tpr_gap"],
                ),
                "fpr_gap": st.column_config.NumberColumn(
                    "FPR Gap",
                    format="%.4f",
                    help=COLUMN_HELP["fpr_gap"],
                ),
            },
        )
        st.caption(
            "Disparate Impact ≥ 0.80 satisfies the four-fifths rule. "
            "Smaller `tpr_gap` and `fpr_gap` indicate more equalised odds. "
            "Hover over any column header for its definition."
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
            st.dataframe(pd.DataFrame(head_rows), use_container_width=True, hide_index=True)

        _section_header(
            "Per-group breakdown",
            [
                _FM["Selection Rate"],
                _FM["TPR (True Positive Rate) / Recall"],
                _FM["FPR (False Positive Rate)"],
            ],
        )
        for model_name, attrs in per_model.items():
            with st.expander(f"{model_name}", expanded=False):
                for attr_name, payload in attrs.items():
                    st.markdown(f"**{attr_name}**")
                    groups = payload.get("groups") or {}
                    if groups:
                        gdf = pd.DataFrame(groups).T
                        st.dataframe(gdf, use_container_width=True)
                    di = payload.get("disparate_impact")
                    passes = payload.get("passes_80_pct_rule")
                    eo = payload.get("equalized_odds") or {}
                    st.write(
                        f"Disparate Impact: **{di}** "
                        f"({'PASSES' if passes else 'FAILS'} 80% rule) · "
                        f"TPR gap: {eo.get('tpr_gap')} · FPR gap: {eo.get('fpr_gap')}"
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
        # Walk one level deep so files inside subfolders (e.g. models/*.pkl)
        # are listed too. Directories themselves are skipped — read_bytes
        # would fail on them.
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
        st.subheader("Latest fairness_metrics.json (in memory)")
        st.json(st.session_state["fairness_metrics"])

    if st.session_state.get("bias_signals"):
        st.markdown("---")
        st.subheader("Latest bias_signals.json (in memory)")
        st.json(st.session_state["bias_signals"])
