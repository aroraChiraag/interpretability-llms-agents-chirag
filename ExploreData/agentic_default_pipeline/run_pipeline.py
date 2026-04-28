"""Entry-point script: run the agentic default-classification pipeline.

Examples
--------
Run with the bundled CSV and all three models:

    python run_pipeline.py

Specify a different CSV and only the tree-based models:

    python run_pipeline.py --csv /path/to/data.csv \
        --models random_forest xgboost \
        --output-dir outputs/run_2026_04_28

The script expects ``GEMINI_API_KEY`` in the environment (e.g. via a ``.env``
file at the repo root). If the key is missing it will exit with a clear error
message.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the local src/ importable when executing the script directly.
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

from agentic_default.ml_trainer import SUPPORTED_MODELS  # noqa: E402
from agentic_default.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("--csv", default=None, help="Path to the input CSV.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        choices=list(SUPPORTED_MODELS),
        help="Which classifiers to train.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(THIS_DIR / "outputs" / "latest"),
        help="Where to drop dataset_summary.json, metrics_report.json, explanation.md.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "ERROR: GEMINI_API_KEY is not set. Add it to your .env at the repo root, "
            "or `export GEMINI_API_KEY=...` before running.",
            file=sys.stderr,
        )
        return 2

    print(f"Running agentic pipeline with models: {args.models}")
    run = run_pipeline(
        csv_path=args.csv,
        models=args.models,
        output_dir=args.output_dir,
        gemini_model=args.gemini_model,
    )
    print("=" * 72)
    print("DATASET SUMMARY")
    print("=" * 72)
    print(run.dataset_summary)
    print()
    print("=" * 72)
    print("METRICS REPORT (leaderboard)")
    print("=" * 72)
    print(run.metrics_report.get("leaderboard"))
    print()
    print("=" * 72)
    print("EXPLANATION (Markdown, truncated at 1k chars)")
    print("=" * 72)
    print(run.explanation_markdown[:1000])
    print(f"\nFull artifacts saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
