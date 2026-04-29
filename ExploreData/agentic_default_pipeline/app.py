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

from agentic_default.data_loader import load_dataset  # noqa: E402
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
        CoordinatorAgent,
        ExplainerAgent,
    )

    st.session_state["_CoordinatorAgent"] = CoordinatorAgent
    st.session_state["_ExplainerAgent"] = ExplainerAgent
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

tab_run, tab_chat, tab_explain, tab_artifacts = st.tabs(
    ["Run", "Chat", "Explanation", "Artifacts"]
)


# ---------- Run tab ---------------------------------------------------------

with tab_run:
    st.subheader("1. Load the dataset")
    cols = st.columns([1, 3])
    with cols[0]:
        if st.button("Load / refresh dataset"):
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

    st.markdown("---")
    st.subheader("2. Train models")

    btn_cols = st.columns(4)
    with btn_cols[0]:
        if st.button("Random Forest"):
            _train_models(["random_forest"])
    with btn_cols[1]:
        if st.button("XGBoost"):
            _train_models(["xgboost"])
    with btn_cols[2]:
        if st.button("Neural Network"):
            _train_models(["neural_network"])
    with btn_cols[3]:
        if st.button("Run all + explain", type="primary"):
            _train_models(list(SUPPORTED_MODELS))
            _run_explainer()

    st.markdown("---")
    st.subheader("3. Results")

    report = st.session_state["metrics_report"]
    if not report:
        st.info("No models trained yet — click one of the buttons above.")
    else:
        st.markdown(f"**Best model:** `{report.get('best_model')}`")

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

        if st.button("Generate / refresh explanation"):
            _run_explainer()


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

            # The coordinator may have updated state — pull the latest into
            # the session so the Run tab and forms reflect it.
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


# ---------- Artifacts tab ---------------------------------------------------

with tab_artifacts:
    st.subheader("Saved artifacts")
    out_dir = Path(st.session_state["outputs_dir"])
    if not out_dir.exists():
        st.info("No outputs yet — run the pipeline at least once.")
    else:
        files = sorted(out_dir.iterdir())
        if not files:
            st.info("Output folder is empty.")
        else:
            for f in files:
                col1, col2, col3 = st.columns([4, 2, 2])
                with col1:
                    st.write(f.name)
                with col2:
                    st.write(f"{f.stat().st_size} bytes")
                with col3:
                    st.download_button(
                        "Download",
                        data=f.read_bytes(),
                        file_name=f.name,
                        key=f"dl_{f.name}",
                    )
        st.caption(f"Folder: {out_dir}")

    if st.session_state["metrics_report"]:
        st.markdown("---")
        st.subheader("Latest metrics_report.json (in memory)")
        st.json(st.session_state["metrics_report"])
