"""Plain Python end-to-end demo of the agentic default-classification pipeline.

Run this from a terminal — it does not require Jupyter and will not prompt for
any y/N input.

The script runs in two phases:

1. **Offline ML sanity check** — loads the CSV, trains the three models with
   plain Python (no LLM calls). This works even if you do not yet have a
   Gemini API key.
2. **Full agentic pipeline** — DataAgent → TrainerAgent → ExplainerAgent.
   This phase is skipped automatically if ``GEMINI_API_KEY`` is not set, so
   the script always exits cleanly.

Examples
--------
Run the whole thing with the bundled CSV::

    cd Explore/agentic_default_pipeline
    uv run --env-file ../../.env python demo.py

Run only the offline ML sanity check (no API key needed)::

    python demo.py --skip-agents

Run with a smaller model subset::

    python demo.py --models random_forest neural_network
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Path setup so this script works both from the project folder and from the
# repo root.
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Best-effort .env load (no hard dependency).
try:
    from dotenv import load_dotenv

    load_dotenv(THIS_DIR.parent.parent / ".env")
except Exception:
    pass

# Importing the package applies the non-interactive env defaults that suppress
# the CrewAI "view execution traces? (y/N)" prompt.
import agentic_default  # noqa: F401, E402

from agentic_default.data_loader import (  # noqa: E402
    csv_to_json_records,
    load_dataset,
)
from agentic_default.ml_trainer import SUPPORTED_MODELS, train_and_evaluate  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the agentic credit-card-default classification demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Outputs land in <project>/outputs/demo_run/ (or whatever
            --output-dir you pass).
            """
        ).strip(),
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to the input CSV. Defaults to the bundled file in Explore/.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        choices=list(SUPPORTED_MODELS),
        help="Which classifiers to train.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(THIS_DIR / "outputs" / "demo_run"),
        help="Where to drop dataset_summary.json, metrics_report.json, explanation.md.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help="Gemini model name used by every agent.",
    )
    parser.add_argument(
        "--skip-agents",
        action="store_true",
        help="Skip the LLM/agent phase and only run the offline ML sanity check.",
    )
    parser.add_argument(
        "--skip-offline",
        action="store_true",
        help="Skip the offline ML sanity check and go straight to the agentic pipeline.",
    )
    return parser.parse_args()


def banner(title: str) -> None:
    line = "=" * 72
    print()
    print(line)
    print(title)
    print(line)


# ---------------------------------------------------------------------------
# Phase 1 — offline ML sanity check (no LLMs)
# ---------------------------------------------------------------------------


def run_offline_phase(csv_path: str | None, models: List[str]) -> None:
    banner("Phase 1 — Offline ML sanity check (no LLM calls)")

    print("Loading dataset...")
    ds = load_dataset(csv_path=csv_path)
    print(f"  rows           : {ds.metadata.n_rows}")
    print(f"  features       : {ds.metadata.n_features}")
    print(f"  class balance  : {ds.metadata.class_balance}")
    print(f"  train / test   : {ds.metadata.train_size} / {ds.metadata.test_size}")
    print(f"  feature names  : {ds.feature_names[:6]} ...")

    preview = csv_to_json_records(csv_path=csv_path, sample=1)
    print("\nFirst CSV row as JSON:")
    print(json.dumps(preview[0], indent=2))

    print(f"\nTraining models: {models}")
    report = train_and_evaluate(
        ds.x_train,
        ds.y_train,
        ds.x_test,
        ds.y_test,
        feature_names=ds.feature_names,
        models=models,
    )
    print("\nLeaderboard (sorted by ROC-AUC):")
    for row in report["leaderboard"]:
        print(
            f"  {row['model_name']:<16}"
            f"  roc_auc={row['roc_auc']:.4f}"
            f"  f1={row['f1']:.4f}"
            f"  precision={row['precision']:.4f}"
            f"  recall={row['recall']:.4f}"
            f"  accuracy={row['accuracy']:.4f}"
        )
    print(f"\nBest model: {report['best_model']}")


# ---------------------------------------------------------------------------
# Phase 2 — full agentic pipeline (Data → Trainer → Explainer)
# ---------------------------------------------------------------------------


def run_agentic_phase(
    csv_path: str | None,
    models: List[str],
    output_dir: str,
    gemini_model: str,
) -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        banner("Phase 2 — Agentic pipeline (SKIPPED)")
        print("GEMINI_API_KEY is not set, so the agent phase is being skipped.")
        print("Add it to your .env at the repo root and re-run, or pass --skip-agents")
        print("to silence this notice.")
        return

    banner("Phase 2 — Agentic pipeline (Data → Trainer → Explainer)")
    # Lazy import so the offline phase still works even if crewai is missing.
    from agentic_default.pipeline import run_pipeline  # noqa: WPS433

    print(f"Gemini model    : {gemini_model}")
    print(f"Models to train : {models}")
    print(f"Output dir      : {output_dir}")
    print("\nKicking off the crew. This will take a minute or two...\n")

    run = run_pipeline(
        csv_path=csv_path,
        models=models,
        output_dir=output_dir,
        gemini_model=gemini_model,
    )

    banner("DataAgent — JSON summary")
    print(json.dumps(run.dataset_summary, indent=2)[:2000])

    banner("TrainerAgent — leaderboard")
    print(json.dumps(run.metrics_report.get("leaderboard", []), indent=2))
    print(f"\nBest model: {run.metrics_report.get('best_model')}")

    banner("ExplainerAgent — Markdown brief")
    print(run.explanation_markdown)

    banner("Saved artifacts")
    out_path = Path(output_dir)
    if out_path.exists():
        for f in sorted(out_path.iterdir()):
            print(f"  {f.name}  ({f.stat().st_size} bytes)")
    print(f"\nFull artifacts saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    if not args.skip_offline:
        run_offline_phase(csv_path=args.csv, models=args.models)

    if not args.skip_agents:
        run_agentic_phase(
            csv_path=args.csv,
            models=args.models,
            output_dir=args.output_dir,
            gemini_model=args.gemini_model,
        )

    banner("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
