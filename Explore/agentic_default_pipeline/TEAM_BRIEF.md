# Team brief — Agentic credit-default classification pipeline

Hey team,

Quick write-up so we're all on the same page before our next bootcamp session.
This is what I've built so far, why we built it that way, and where I'd love
your help. Read it whenever — happy to walk through anything live.

## The business framing

We're prototyping the same problem we look at every day at the bank: **given
a credit card customer's history, can we predict whether they'll default next
month?** The bootcamp gave us the open-source Taiwan default-of-credit-card
dataset (30,000 customers, 23 features — credit limit, age, education, the
last 6 months of repayment status and bill amounts, etc.). It's not our
data, but the shape is identical to what our risk team works with.

The bootcamp is about **interpretability** — not just "what does the model
predict" but "why, and can a human follow the reasoning?" Regulators care.
Risk committees care. So do we.

## What "agentic AI pipeline" means here (plain English)

In the past, we'd write one Python script: load data → train model →
print metrics. End of story.

What we've built instead is **three small AI assistants (agents) working
in sequence**, each one with a job, a set of allowed actions ("tools"), and
written instructions. They pass notes to each other in JSON. Think of it
like a relay race where each runner is a Gemini-powered agent:

1. **Data Agent** — opens the CSV, splits it into train/test, writes a
   short JSON summary about the dataset (row count, class balance,
   feature names, a few preview rows).
2. **Trainer Agent** — takes that summary, calls a tool that trains
   Random Forest, XGBoost, and a Neural Network, and emits a JSON metric
   report (precision, recall, F1, ROC-AUC, confusion matrices, top
   features per model).
3. **Explainer Agent** — reads the metric report and writes a
   plain-English Markdown brief: best model, what the confusion matrix
   means, which features matter, caveats around fairness and class
   imbalance.

The agents don't actually do the math — sklearn does. The agents *decide
when to call the math*, *summarize the results*, and *hand off cleanly*.
That's the whole point: it lets us swap in different models, different
data, different prompts and have the system still produce a coherent,
reproducible report.

## Why this matters for what we do at the bank

If you squint, this is a tiny version of what an internal "model risk
analyst assistant" could look like. A junior analyst asks "what does this
model do?" and instead of digging through a notebook for an hour, the
explainer agent gives them the same metrics summary every time, in the
same format, with the same caveats. Standardization + auditability.

That's the angle I want us to bring back to our team after the bootcamp.

## How the code is laid out

Everything lives under `Explore/agentic_default_pipeline/` in the repo.
The two files you'll touch most:

- **`demo.py`** at the project root — the runnable entry point. No
  Jupyter required, no interactive prompts.
- **`src/agentic_default/prompts/*.txt`** — three plain-text files, one
  per agent. These are the *instructions* we hand to Gemini. They are
  the most human-editable part of the project. **This is where I want
  most of your feedback.**

Lower-priority browsing:

- `src/agentic_default/data_loader.py` — pandas-based CSV loader, train/test
  split, feature scaling.
- `src/agentic_default/ml_trainer.py` — the actual sklearn / XGBoost /
  MLP training and metric calculation. No LLMs involved.
- `src/agentic_default/tools/` — thin wrappers around the two modules
  above that expose them as "tools" the agents can call.
- `src/agentic_default/agents/` — the three agents, each with its
  role/goal/backstory and the tools it's allowed to use.
- `src/agentic_default/pipeline.py` — chains the three agents together.

You don't need to understand CrewAI internals. The mental model is just:
prompt + tools + LLM = agent. Three of those, run in order.

## What you can do this week (no API key needed yet)

The bootcamp organizers haven't given us the Gemini API key, so the LLM
phase can't run yet. But you can do all of this offline:

### 1. Pull the repo and get it set up

```bash
git clone <our team's repo URL>
cd interpretability-llms-agents
uv sync --group agentic-xai-eval
```

### 2. Run the offline ML sanity check

This trains the three models without any LLM calls and prints the metrics
leaderboard. Confirms the data pipeline and ML layer work end-to-end:

```bash
cd Explore/agentic_default_pipeline
python demo.py --skip-agents
```

You should see something like:

```
Loading dataset...
  rows           : 30000
  features       : 23
  class balance  : {'0': 23364, '1': 6636}
  ...
Leaderboard (sorted by ROC-AUC):
  xgboost          roc_auc=0.78  f1=0.47  precision=0.65  recall=0.37
  random_forest    roc_auc=0.76  ...
  neural_network   roc_auc=0.74  ...
```

(Numbers won't be identical — random seed and library versions wobble
things — but the ordering will be similar.)

### 3. Read the prompts and tell me what's missing

Open these three files. Each is plain English. Read like a risk-team
stakeholder, not a developer:

- `src/agentic_default/prompts/data_agent.txt`
- `src/agentic_default/prompts/trainer_agent.txt`
- `src/agentic_default/prompts/explainer_agent.txt`

### 4. Tweak the explainer prompt — this is the high-leverage one

The explainer prompt (`explainer_agent.txt`) is where the **tone, depth,
and audience of the final report is set**. Right now it's tuned for
"bootcamp audience that already understands ML basics." For our actual
work, we'd want it tuned for a *risk committee* or a *compliance reviewer*.

Try editing it to one of these voices and tell me how the output changes
once we have the API key:

- Risk committee voice: "Lead with the headline number. Anchor everything
  to expected loss. Always quantify the cost of a false negative."
- Compliance / model-risk voice: "Cite which rows of the confusion matrix
  feed which regulatory metric. Always note assumptions. Always flag if
  PAY_0 is the dominant feature — that's a leakage smell."
- Customer-facing voice: "No jargon. Translate ROC-AUC into 'how often
  the model ranks an actual defaulter ahead of a non-defaulter'."

You don't need to write code. Edit the `.txt` file, save, push to a branch.

## What I need from you specifically

1. **Read the three prompt files** and write me 2–3 sentences each on:
   what would a stakeholder at *our* bank find missing? What's in the
   explainer's brief that we'd want to drop or expand?
2. **Sanity-check the metric definitions** in the explainer prompt —
   does it explain precision/recall/F1/ROC-AUC the way *you'd* want it
   explained to a non-ML person? Anything misleading?
3. **One worry I have:** the explainer agent could hallucinate. It's
   instructed to quote numbers verbatim from the JSON report, but I
   haven't built a strict numeric-grounding check yet. If you spot a
   number in its Markdown that doesn't match the JSON, that's the
   highest-priority bug to flag.
4. **Fairness lens:** SEX, EDUCATION, and MARRIAGE are in the feature
   list. We use sensitive features at the bank too (with controls).
   Look at the explainer prompt's "Caveats" section and tell me if the
   fairness language is strong enough for what our model risk team
   would want.

## What you don't need to worry about

- The CrewAI / Gemini wiring. I've handled it. If something breaks at
  that layer, ping me.
- The sklearn / XGBoost / MLP code. It's standard tabular ML — same
  stack we use for our internal scorecards.
- The y/N "view execution traces" prompt. It used to hang the run; I've
  added a context manager that auto-declines it while keeping tracing on.

## Once the API keys arrive

Run the full pipeline:

```bash
# add GEMINI_API_KEY=... to .env at the repo root
cd Explore/agentic_default_pipeline
uv run --env-file ../../.env python demo.py
```

Output lands in `outputs/demo_run/`:
- `dataset_summary.json` — what the Data Agent emitted
- `metrics_report.json` — what the Trainer Agent emitted
- `explanation.md` — the final human-readable brief

The Markdown file is the artifact we should be reviewing as a team.
We'll iterate on the prompts, re-run, and compare.

## TL;DR

I built three little AI agents that load the data, train the models,
and write up the results. The interesting work — for *us*, at a bank —
isn't the model code. It's the **prompts** that decide what the report
emphasises and who it's written for. That's where I need your eyes.

Ping me on Slack with questions. Aim to push your prompt edits to a
branch by EOD Friday.

— [Your name]
