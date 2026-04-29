# Agentic Default-Payments Classification Pipeline

A small CrewAI pipeline that trains and explains classification models on the
Taiwan **default-of-credit-card-clients** dataset. Built for the Vector
Institute *Interpretability for LLMs and Agents* bootcamp.

The pipeline has three sequential agents, all powered by Gemini:

| Phase | Agent | Tools it can call | Output |
|-------|-------|-------------------|--------|
| 1 | **DataAgent** | `load_dataset`, `preview_records` | JSON dataset summary |
| 2 | **TrainerAgent** | `train_models`, `get_metrics` | JSON metrics report (per-model) |
| 3 | **ExplainerAgent** | (no tools — text only) | Markdown interpretability brief |

The classification stack trains three models and reports a leaderboard:

- **Random Forest** (sklearn)
- **XGBoost** — falls back to `GradientBoostingClassifier` if `xgboost` is not installed
- **Neural Network** — sklearn `MLPClassifier` (64 → 32 ReLU)

Per model the pipeline reports accuracy, balanced accuracy, precision, recall,
F1, ROC-AUC, average precision, the confusion matrix, training time, and the
top-10 feature importances (weight-magnitude proxy for the MLP).

## Project layout

```
Explore/agentic_default_pipeline/
├── README.md
├── app.py                         # Streamlit web UI (start here)
├── demo.py                        # plain Python end-to-end demo
├── run_pipeline.py                # CLI entry point (full agentic flow only)
├── TEAM_BRIEF.md                  # onboarding doc for non-technical teammates
├── notebooks/
│   └── demo.ipynb                 # same flow, in Jupyter
├── outputs/                       # per-run artifacts land here
└── src/agentic_default/
    ├── __init__.py
    ├── _runtime.py                # auto-declines CrewAI's trace y/N prompt
    ├── data_loader.py             # CSV → JSON, train/test split, metadata
    ├── ml_trainer.py              # RF / XGBoost / MLP + metrics + hp defaults
    ├── pipeline.py                # Crew orchestration
    ├── state.py                   # in-memory pipeline state (incl. hp dict)
    ├── prompts/
    │   ├── data_agent.txt
    │   ├── trainer_agent.txt
    │   ├── explainer_agent.txt
    │   └── coordinator_agent.txt  # chat / tuning prompt
    ├── tools/
    │   ├── dataset_tools.py       # LoadDatasetTool, PreviewRecordsTool
    │   └── training_tools.py      # TrainModelsTool, GetMetricsTool,
    │                              # Update/Get/ResetHyperparametersTool
    └── agents/
        ├── _crew_helpers.py       # build_crew + kickoff_quiet
        ├── llm.py                 # Gemini LLM helper
        ├── data_agent.py
        ├── trainer_agent.py
        ├── explainer_agent.py
        └── coordinator_agent.py   # chat-driven pipeline coordinator
```

## Setup

The bootcamp uses [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
# install the same dependency group used by the agentic VQA implementation —
# it already pulls crewai + Gemini support.
uv sync --group agentic-xai-eval

# (optional but recommended) install xgboost into the same environment
uv pip install xgboost
```

Then create a `.env` at the repo root with:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

(The bootcamp organizers will provide the key.)

## Run the Streamlit web app (recommended)

A browser UI with per-model "Run" buttons, a sidebar hyperparameter form, a
chat tab for natural-language tuning, and a dedicated tab that renders the
explainer's Markdown brief.

```bash
cd Explore/agentic_default_pipeline
uv run --env-file ../../.env streamlit run app.py
```

Streamlit prints a local URL (usually <http://localhost:8501>); open it in
your browser. What each tab does:

- **Run** — load the dataset, train each model individually (RF, XGBoost,
  Neural Network) or all together with the "Run all + explain" button.
  Shows the leaderboard, per-model metrics, confusion matrices, top
  feature importances, and the hyperparameters that were actually used.
- **Chat** — talk to the Pipeline Coordinator agent in plain English.
  Examples: *"Use 500 trees in random forest and re-run"*, *"Drop the
  XGBoost learning rate to 0.01"*, *"Which model has the best recall?"*,
  *"Reset hyperparameters"*. The coordinator uses tools to update
  pipeline state and retrains as needed; numbers it cites are pulled
  verbatim from the metrics report (it cannot fabricate values).
- **Explanation** — the explainer agent's full Markdown brief, rendered
  big and clean. Download button included.
- **Artifacts** — file links for the JSON / Markdown saved to
  `outputs/streamlit_run/`.

The sidebar has a hyperparameter form with expanders per model. The form
and the chat both read/write the same in-memory state, so a tweak in
either place is reflected in the other.

## Run the demo (recommended)

`demo.py` is a plain Python script — **no Jupyter required, no y/N prompts**.
It runs an offline ML sanity check first (works without the API key) and then
the full agentic pipeline (skipped automatically if `GEMINI_API_KEY` is unset).

```bash
cd Explore/agentic_default_pipeline
uv run --env-file ../../.env python demo.py
```

Useful flags:

```bash
# Only the offline ML sanity check — no LLM calls.
python demo.py --skip-agents

# Only the agentic phase (skip the offline preview).
python demo.py --skip-offline

# Train a smaller model subset.
python demo.py --models random_forest neural_network
```

## Run the full pipeline programmatically

```bash
cd Explore/agentic_default_pipeline
uv run --env-file ../../.env python run_pipeline.py \
    --models random_forest xgboost neural_network \
    --output-dir outputs/run_$(date +%Y%m%d_%H%M%S)
```

Both `demo.py` and `run_pipeline.py` write the following into the chosen
output directory:

- `dataset.json` — full dataset as a JSON record list
- `metadata.json` — dataset metadata (row counts, class balance, etc.)
- `dataset_summary.json` — what the DataAgent emitted
- `metrics_report.json` — what the TrainerAgent / training tool emitted
- `explanation.md` — the ExplainerAgent's Markdown interpretability brief

## A note on CrewAI's trace prompt

CrewAI 1.x can ask "Would you like to view the execution traces? (y/N)" at
the end of a run, which hangs in non-interactive contexts (Jupyter, CI, log
pipelines).

**Tracing itself is left ON** — the trace data is still collected (and
uploaded if you've configured a tracing backend). Only the interactive
prompt is silenced. Two mechanisms do this:

1. The package sets these env vars on import (using `os.environ.setdefault`,
   so your values always win):
   - `CREWAI_NON_INTERACTIVE=true`
   - `CREWAI_DISABLE_TRACE_PROMPT=true`
2. Each `crew.kickoff()` call is wrapped in `auto_decline_prompts()`, which
   monkey-patches `builtins.input` to auto-return `"n"` for the duration of
   the kickoff. The original prompt is echoed to stderr with an
   `[auto-declined]` prefix so you can see what was suppressed.

If you actually want the prompt back (e.g. running interactively), import
`agentic_default.pipeline.run_pipeline` and call your agents directly without
going through `kickoff_quiet`.

## Run programmatically

```python
from agentic_default.pipeline import run_pipeline

run = run_pipeline(
    models=["random_forest", "xgboost", "neural_network"],
    output_dir="outputs/run_demo",
)

print(run.dataset_summary)       # dict
print(run.metrics_report)        # dict (with leaderboard + per-model)
print(run.explanation_markdown)  # str — Markdown
```

The notebook in `notebooks/demo.ipynb` shows the same flow.

## Run the ML layer without the agents

Useful for unit-style sanity checks before you have an API key:

```python
from agentic_default.data_loader import load_dataset
from agentic_default.ml_trainer import train_and_evaluate

ds = load_dataset()
report = train_and_evaluate(
    ds.x_train, ds.y_train, ds.x_test, ds.y_test,
    feature_names=ds.feature_names,
)
print(report["leaderboard"])
```

## Notes on data

The bundled CSV has 30 000 rows, 23 features, and a binary target
`default payment next month`. The class balance is roughly 78 % no-default /
22 % default — heavy enough that ROC-AUC and average precision are more
informative than raw accuracy. The pipeline uses a stratified 80/20 split.

## Caveats and follow-ups

- The MLP's "feature importance" is a coarse proxy (sum of `|W1|` across
  hidden units). Use SHAP or permutation importance for a faithful view.
- The pipeline does not tune thresholds; everything is reported at the default
  0.5 decision boundary.
- `SEX`, `EDUCATION`, and `MARRIAGE` are sensitive demographics — a fairness
  audit (subgroup metrics, equalized-odds gap) belongs in a follow-up
  implementation.
