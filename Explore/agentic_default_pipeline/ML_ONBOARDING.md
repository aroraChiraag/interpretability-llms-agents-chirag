# ML Expert Onboarding — Agentic Default-Payments Pipeline

Welcome. This doc walks you through the dataset, the project layout, and
how everything fits together. Reading top-to-bottom will give you enough
context to start contributing on day 1.

## 1. The dataset

### Where it lives

```
Explore/
├── default_of_credit_card_clients.csv      # the raw data (30,000 rows)
├── customers_default_payments.ipynb        # exploratory notebook bundled with the bootcamp
└── agentic_default_pipeline/               # our project (everything else in this doc)
```

### What it is

Open-source dataset from a Taiwan-based study on credit-card default
prediction. The original research compared six data-mining methods for
estimating *probability of default*, and concluded that an artificial
neural network produced the best-calibrated probability estimates (the
study introduces a "Sorting Smoothing Method" to estimate true default
probability, and finds the ANN's predicted probability tracks closest to
that estimate).

For our bootcamp work, the response variable is the binary
`default payment next month` (1 = defaulted, 0 = did not). There are no
missing values. 30,000 customers, 23 features, no NA handling required.

### Feature dictionary

All amounts are in NT dollars. Repayment-status codes follow the original
study's scale.

| Column | Meaning |
|---|---|
| `ID` | Row identifier (we drop this — not a feature). |
| `LIMIT_BAL` | Amount of given credit, including the consumer's own line and any supplementary (family) credit. |
| `SEX` | 1 = male, 2 = female. |
| `EDUCATION` | 1 = graduate school, 2 = university, 3 = high school, 4 = others. |
| `MARRIAGE` | 1 = married, 2 = single, 3 = others. |
| `AGE` | Age in years. |
| `PAY_0` | Repayment status in **September 2005**. |
| `PAY_2` | Repayment status in **August 2005**. |
| `PAY_3` | Repayment status in **July 2005**. |
| `PAY_4` | Repayment status in **June 2005**. |
| `PAY_5` | Repayment status in **May 2005**. |
| `PAY_6` | Repayment status in **April 2005**. |
| `BILL_AMT1` … `BILL_AMT6` | Amount of bill statement, September 2005 → April 2005 (same month order as `PAY_*`). |
| `PAY_AMT1` … `PAY_AMT6` | Amount paid in each of those months, September 2005 → April 2005. |
| `default payment next month` | **Target.** 1 if the customer defaulted the month after the observation window, else 0. |

**`PAY_*` repayment-status scale** (from the source paper):
`-1` = paid on time; `1` = delay of one month; `2` = delay of two months;
…; `8` = delay of eight months; `9` = delay of nine months or more. You
will also see `-2` and `0` codes in the raw data — those are present but
not documented in the source paper; common interpretations in the
community are `-2` = no consumption / no balance and `0` = revolving
credit. Treat them as ordinal but flag them in any preprocessing.

### Class balance and what to watch for

Roughly **78 % no-default / 22 % default**. That makes raw accuracy
unreliable as a single-number metric — ROC-AUC, average precision, and
balanced accuracy are the ones to lead with. The pipeline's stratified
80/20 split preserves this ratio in both halves.

`PAY_0` (September 2005 repayment status) typically dominates feature
importance. That is *partly* legitimate signal (most-recent behaviour
predicts next-month behaviour) and *partly* a leakage smell — it is
literally the month right before the target window. Worth interrogating
in your first review pass.

`SEX`, `EDUCATION`, `MARRIAGE` are sensitive features. We use them as
inputs in this baseline, but they are exactly what the upcoming Bias and
Fairness agents (see §6) will scrutinise.

---

## 2. Project layout, file by file

```
Explore/agentic_default_pipeline/
├── README.md                       # User-facing setup & run guide
├── TEAM_BRIEF.md                   # Onboarding doc for non-technical teammates
├── ML_ONBOARDING.md                # ← this file
├── app.py                          # Streamlit web UI
├── demo.py                         # Plain Python end-to-end demo
├── run_pipeline.py                 # CLI entry point (full agentic flow)
├── notebooks/
│   └── demo.ipynb                  # Notebook walkthrough (same flow as demo.py)
├── outputs/                        # Per-run artifacts land here
└── src/agentic_default/            # The package
    ├── __init__.py                 # Applies non-interactive defaults on import
    ├── _runtime.py                 # Auto-decline CrewAI's trace y/N prompt
    ├── data_loader.py              # CSV → JSON, train/test split, metadata
    ├── ml_trainer.py               # RF / XGBoost / MLP + metrics + hp defaults
    ├── pipeline.py                 # Sequential Crew orchestration
    ├── state.py                    # In-memory shared PipelineState
    ├── prompts/
    │   ├── data_agent.txt
    │   ├── trainer_agent.txt
    │   ├── explainer_agent.txt
    │   └── coordinator_agent.txt
    ├── tools/
    │   ├── dataset_tools.py        # LoadDatasetTool, PreviewRecordsTool
    │   └── training_tools.py       # Train/Get/Update/Reset tools
    └── agents/
        ├── _crew_helpers.py        # build_crew + kickoff_quiet
        ├── llm.py                  # Gemini LLM helper
        ├── data_agent.py
        ├── trainer_agent.py
        ├── explainer_agent.py
        └── coordinator_agent.py    # Chat-driven pipeline coordinator
```

### 2.1 Top-level entry points

**`README.md`** (229 lines) — User setup guide. How to install the
`agentic-xai-eval` uv group, where to put the Gemini API key, how to
launch the Streamlit UI, what each tab does, and how the trace-prompt
suppression works.

**`TEAM_BRIEF.md`** (212 lines) — Plain-English onboarding for
non-technical teammates (business analysts, model risk). Useful context
even for ML folks who want to know what we're communicating to
stakeholders.

**`app.py`** (542 lines) — The Streamlit web app. Four tabs:
*Run* (per-model buttons + leaderboard + confusion matrices + feature
importance bar charts), *Chat* (Coordinator agent), *Explanation*
(rendered Markdown brief), *Artifacts* (downloadable JSON / Markdown
files). Sidebar has a hyperparameter form with one expander per model
that reads/writes the same `PipelineState` dict the chat agent uses.

**`demo.py`** (248 lines) — Pure Python end-to-end runner. Two phases:
an offline ML sanity check (no LLM calls — works without an API key) and
the full agentic pipeline (auto-skipped if `GEMINI_API_KEY` is unset).
This is the fastest way to convince yourself the ML layer works.

**`run_pipeline.py`** (104 lines) — Thin CLI entry point that calls
`agentic_default.pipeline.run_pipeline(...)`. Flags: `--csv`, `--models`,
`--output-dir`, `--gemini-model`. Use this for batch / scripted runs;
use `app.py` for exploration.

**`notebooks/demo.ipynb`** — Same flow as `demo.py` but in cells. Useful
for sharing a step-by-step walkthrough; not the recommended day-to-day
runner since CrewAI's interactive prompts historically misbehaved here
(now suppressed via `_runtime.py`, see §2.4).

### 2.2 The "real ML" layer (no LLMs)

These three files are the only ones that touch sklearn / numpy / pandas.
Everything else either calls into them or wraps them.

**`src/agentic_default/data_loader.py`** (293 lines) — pandas-based CSV
ingestion. Drops `ID`, separates `default payment next month` as the
target, returns numpy arrays for X (feature-scaled by `StandardScaler`)
and y. Provides:

- `csv_to_json_records(path, sample=N)` — load the raw CSV as a list of
  Python dicts (this is what's surfaced to LLM previews).
- `load_dataset(...)` → `LoadedDataset` dataclass containing
  `x_train`, `x_test`, `y_train`, `y_test`, `feature_names`, and a
  `DatasetMetadata` summary (row counts, class balance, etc.).
- `summarize_for_agent(metadata, preview)` — compact JSON-friendly
  description suitable for a prompt.

**`src/agentic_default/ml_trainer.py`** (379 lines) — model factories,
training loop, metric extraction, feature importances. Three models:
`RandomForestClassifier`, `XGBClassifier` (falls back to sklearn's
`GradientBoostingClassifier` if xgboost isn't installed),
`MLPClassifier` (64 → 32 ReLU). Public API:

- `default_hyperparameters()` — fresh deep copy of per-model defaults.
- `merge_hyperparameters(overrides)` — overlay user overrides on
  defaults.
- `train_and_evaluate(x_train, y_train, x_test, y_test, feature_names,
  models=..., random_state=..., output_dir=..., hyperparameters=...)`
  → dict with `models` (per-model results), `leaderboard` (sorted by
  ROC-AUC desc), `best_model`, `hyperparameters_used`.

Per-model output includes accuracy, balanced accuracy, precision,
recall, F1, ROC-AUC, average precision, the full sklearn
`classification_report` dict, the confusion matrix, the top-10 feature
importances (`|W1|` magnitude proxy for the MLP — flagged in the
`notes` field), the training time, and the actual hyperparameters used.

**`src/agentic_default/state.py`** (43 lines) — single-process
`PipelineState` dataclass containing the loaded dataset, the latest
metrics report, the explainer's Markdown, and the per-model
hyperparameter dict. Stored in a module-level dict keyed by `state_handle`
(the Streamlit app uses `"streamlit"`, the CLI uses `"default"`).
Thread-safe via a single `threading.Lock`. The reason the state exists:
numpy arrays are too big to ship through an LLM prompt, so the agents
pass small JSON summaries to each other while arrays sit in this store.

### 2.3 Tools layer (CrewAI `BaseTool` subclasses)

**`src/agentic_default/tools/dataset_tools.py`** (105 lines):

- `LoadDatasetTool` — calls `data_loader.load_dataset`, stores the
  arrays in `PipelineState`, returns a JSON summary plus a 3-row
  preview.
- `PreviewRecordsTool` — returns the first N rows of the loaded
  dataset as JSON, for inspection.

**`src/agentic_default/tools/training_tools.py`** (242 lines):

- `TrainModelsTool` — calls `train_and_evaluate`. Honours
  hyperparameter precedence: explicit arg > `PipelineState` > defaults.
  Returns a compact JSON report (the verbose `classification_report` is
  stripped for LLM consumption; full report is still in state).
- `GetMetricsTool` — re-read the latest report.
- `UpdateHyperparametersTool` — merge per-model overrides into state.
  Argument shape: `{"random_forest": {"n_estimators": 500}}`.
- `GetHyperparametersTool` — read the current per-model dict.
- `ResetHyperparametersTool` — restore package defaults.

Each tool has a Pydantic input schema; CrewAI uses the schema +
description to decide when to call it.

### 2.4 Runtime / infrastructure

**`src/agentic_default/_runtime.py`** (69 lines) — Imported by the
package `__init__`, applies two prompt-suppression env defaults
(`CREWAI_NON_INTERACTIVE`, `CREWAI_DISABLE_TRACE_PROMPT`) via
`os.environ.setdefault` so the user can override. Also exports
`auto_decline_prompts()` — a context manager that monkey-patches
`builtins.input` to return `"n"` while echoing `[auto-declined]` to
stderr. **Tracing itself is left enabled** — only the y/N viewer prompt
is short-circuited.

**`src/agentic_default/agents/_crew_helpers.py`** (27 lines) —
`build_crew(...)` constructs a `Crew` with the default tracing
configuration. `kickoff_quiet(crew)` runs `crew.kickoff()` inside the
`auto_decline_prompts` context. Every agent's `run()` method goes
through `kickoff_quiet` so notebooks and CI don't hang.

**`src/agentic_default/agents/llm.py`** (51 lines) — `build_gemini_llm()`
returns a CrewAI `LLM` configured against `gemini/<model>` with the key
from `GEMINI_API_KEY`. Default model: `gemini-2.5-flash`.

### 2.5 Prompts (plain-text instructions to Gemini)

These are *the most edited files in the project*. Tone, audience, and
output structure are all set here, not in code.

**`prompts/data_agent.txt`** (18 lines) — Tells the Data Agent to call
`load_dataset` once, optionally inspect with `preview_records`, and
emit a strict JSON summary. No prose, no Markdown fences.

**`prompts/trainer_agent.txt`** (20 lines) — Tells the Trainer Agent to
call `train_models` once, identify the best model by ROC-AUC, emit a
JSON object with `best_model`, `leaderboard`, `models`, and
`trainer_notes`. Explicitly says: do not interpret the metrics — that's
the explainer's job.

**`prompts/explainer_agent.txt`** (41 lines) — The high-leverage one.
Specifies a five-section Markdown structure: *Headline result*,
*Per-model metrics* (with confusion-matrix interpretation),
*Feature importance* (cross-model comparison + MLP proxy disclosure),
*Reading the metrics* (plain-language definitions tied to the class
balance), *Caveats and next steps*. Hard rule: quote numbers verbatim
from the JSON, never invent values.

**`prompts/coordinator_agent.txt`** (53 lines) — System prompt for the
chat agent. Six numbered decision rules ("if user asks for a
hyperparameter change, call `update_hyperparameters` then
`train_models`"; "if ambiguous, ask one short clarifying question"),
output style rules (2–6 sentences, numbers in **bold**, no JSON in
chat), and hard rules (never claim a retrain unless `train_models` ran
this turn; never invent metrics).

### 2.6 Agents layer

Each agent is a thin class around `crewai.Agent` + `Task` + `Crew`.

**`agents/data_agent.py`** (72 lines) — Loads the data prompt, attaches
`LoadDatasetTool` and `PreviewRecordsTool`, kicks off a one-task crew,
returns the agent's raw output (a JSON string).

**`agents/trainer_agent.py`** (71 lines) — Same shape, attaches
`TrainModelsTool` and `GetMetricsTool`.

**`agents/explainer_agent.py`** (68 lines) — No tools. Takes the
trainer's JSON report (and optionally the data summary) as part of the
task description, runs the Crew, returns the Markdown brief.

**`agents/coordinator_agent.py`** (115 lines) — The chat agent.
Attaches all six tools (`LoadDatasetTool`, `PreviewRecordsTool` is
omitted because it's not in the chat tool list — actually look at the
file; it includes the five non-preview tools plus `LoadDatasetTool`).
Each user turn is a fresh `Task` whose description includes the system
prompt + a 10-turn conversation transcript + the new user message.
Returns the agent's plain-text reply.

**`agents/__init__.py`** — exports `DataAgent`, `TrainerAgent`,
`ExplainerAgent`, `CoordinatorAgent`.

### 2.7 Orchestration

**`src/agentic_default/pipeline.py`** (152 lines) — The
`run_pipeline(...)` function. Resets state, runs the Data Agent,
parses its JSON output, runs the Trainer Agent, parses, runs the
Explainer Agent on the parsed report. Writes
`dataset_summary.json`, `metrics_report.json`, and `explanation.md` to
`output_dir`. Returns a `PipelineRun` dataclass with everything
including the raw outputs (useful for debugging when the JSON parse
fails).

`_maybe_parse_json(text)` is a forgiving JSON parser — it strips
Markdown code fences if the agent ignored its instructions and emitted
` ```json `.

---

## 3. End-to-end data flow

```
                                   ┌──────────────────┐
   default_of_credit_card.csv  ──▶ │   Data Agent     │  load_dataset, preview_records
                                   └────────┬─────────┘
                                            │ JSON summary  + arrays into PipelineState
                                            ▼
                                   ┌──────────────────┐
                                   │  Trainer Agent   │  train_models (reads hp from state)
                                   └────────┬─────────┘
                                            │ JSON metrics report + state.metrics_report
                                            ▼
                                   ┌──────────────────┐
                                   │ Explainer Agent  │  no tools — text only
                                   └────────┬─────────┘
                                            │ Markdown brief
                                            ▼
                                  outputs/<run>/explanation.md

   Streamlit Chat tab ─▶ Coordinator Agent ─▶ tools above (loops as needed)
```

**Numpy arrays** flow via `PipelineState` (they never enter prompts).
**JSON summaries and metric reports** flow through the agents as text.
**Hyperparameters** are a dict on `PipelineState` that both the UI form
in the sidebar and the chat coordinator read/write — that's why a
slider tweak and a "use 500 trees" chat message produce the same
behavior.

---

## 4. How to run things

### Day-zero sanity check (no API key)

```bash
cd Explore/agentic_default_pipeline
python demo.py --skip-agents
```

You should see the dataset summary print, the three models train, and a
leaderboard sorted by ROC-AUC.

### Full pipeline (CLI)

```bash
uv run --env-file ../../.env python demo.py
# or
uv run --env-file ../../.env python run_pipeline.py
```

### Streamlit UI

```bash
uv run --env-file ../../.env streamlit run app.py
```

Tabs: **Run** (per-model buttons + Run all + explain), **Chat** (talk to
Coordinator), **Explanation** (rendered Markdown), **Artifacts** (file
links).

### Programmatic

```python
from agentic_default.pipeline import run_pipeline
run = run_pipeline(models=["random_forest", "xgboost", "neural_network"],
                   output_dir="outputs/run_X")
print(run.metrics_report["leaderboard"])
print(run.explanation_markdown)
```

---

## 5. Where to start reading (suggested order)

1. `data_loader.py` and `ml_trainer.py` — the only files that do real
   ML. If you understand these, the rest is plumbing.
2. `prompts/*.txt` — the substantive content of the system. Tweaking
   these is high-leverage.
3. `tools/*.py` — see how Python functions become tools the LLM can
   invoke.
4. `agents/*.py` — short files. The pattern is the same in each.
5. `pipeline.py` — orchestration glue.
6. `app.py` — only if you care about the UI.

---

## 6. What's coming next (for context)

- **Coordinator Agent UI wiring** — chat is plumbed in `app.py` already
  but expect iteration on the prompt and edge cases (ambiguous requests,
  retrain confirmation flow).
- **Splitting the Explainer** into a **Bias Agent** (pre-training:
  dataset-level imbalance, sensitive-feature distributions) and a
  **Fairness Agent** (post-training: subgroup performance gaps over
  `SEX` / `EDUCATION` / `MARRIAGE`, demographic parity, equalized-odds
  difference, disparate impact).
- **Top-level Orchestrator agent** above the existing crew to route
  between Data → Trainer → Bias → Fairness → Explainer based on the
  user's question.
- Business-side teammates are drafting working definitions and concrete
  examples for what *bias* and *fairness* mean in our context — those
  definitions will drive the new agents' prompts.

---

## 7. Conventions and gotchas

- **Tracing is on, prompts are auto-declined.** Don't be surprised by
  `[auto-declined]` lines on stderr — that's normal. Tracing data is
  still being collected.
- **Hyperparameter dict is the source of truth.** If a model trains
  with the wrong hp, check `PipelineState.hyperparameters` first.
- **`PAY_0` will dominate feature importance.** Decide whether to keep
  it, dampen it, or carve out an "early-warning" variant of the dataset
  with `PAY_0` removed. Worth a discussion.
- **Class imbalance ≈ 78/22.** Lead with ROC-AUC and average precision;
  raw accuracy is misleading.
- **No missing values in the source data**, so no NA handling is
  needed.
- **`StandardScaler` is fit on train and applied to test.** Important
  for the MLP; tree models would be fine without it but we scale for
  consistency.
- **`xgboost` is optional** — if it's not in the env, the pipeline
  silently substitutes sklearn's `GradientBoostingClassifier` and
  records the substitution in the model's `notes` field.

Reach out if anything is unclear. Good first PR: tighten the
`explainer_agent.txt` prompt for a model-risk audience, or add a
fairness-by-subgroup table to `ml_trainer.train_and_evaluate`.
