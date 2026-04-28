"""CrewAI tools that wrap data loading.

Each tool subclasses ``crewai.tools.BaseTool`` and exposes a Pydantic input
schema. Tools return JSON strings — large numpy arrays stay in
``state.PipelineState`` and only summary information crosses the agent
boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..data_loader import (
    DEFAULT_CSV_PATH,
    csv_to_json_records,
    load_dataset,
    summarize_for_agent,
)
from ..state import get_state


class LoadDatasetInput(BaseModel):
    """Input schema for :class:`LoadDatasetTool`."""

    csv_path: Optional[str] = Field(
        default=None,
        description="Absolute path to the CSV. Leave empty to use the bundled file.",
    )
    test_size: float = Field(
        default=0.2,
        description="Fraction of rows reserved for the held-out test split.",
    )
    random_state: int = Field(default=42, description="Seed for reproducibility.")
    snapshot_dir: Optional[str] = Field(
        default=None,
        description="Optional folder where dataset.json and metadata.json should be written.",
    )
    state_handle: str = Field(
        default="default",
        description="Handle used to look up the in-memory pipeline state.",
    )


class LoadDatasetTool(BaseTool):
    """Load the credit-card-default CSV into the pipeline state."""

    name: str = "load_dataset"
    description: str = (
        "Load the Taiwan default-of-credit-card-clients CSV, split it into "
        "train/test, store the arrays in pipeline state, and return a JSON "
        "summary (row counts, feature names, class balance, preview records)."
    )
    args_schema: Type[BaseModel] = LoadDatasetInput

    def _run(
        self,
        csv_path: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        snapshot_dir: Optional[str] = None,
        state_handle: str = "default",
    ) -> str:
        path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
        snap = Path(snapshot_dir) if snapshot_dir else None
        dataset = load_dataset(
            csv_path=path,
            test_size=test_size,
            random_state=random_state,
            snapshot_dir=snap,
        )
        get_state(state_handle).dataset = dataset

        preview = csv_to_json_records(csv_path=path, sample=3)
        summary = summarize_for_agent(dataset.metadata, preview)
        summary["state_handle"] = state_handle
        return json.dumps(summary, indent=2)


class PreviewRecordsInput(BaseModel):
    """Input schema for :class:`PreviewRecordsTool`."""

    state_handle: str = Field(default="default")
    n: int = Field(default=5, description="Number of records to preview.")


class PreviewRecordsTool(BaseTool):
    """Return the first N JSON records of the loaded dataset."""

    name: str = "preview_records"
    description: str = (
        "Return the first N rows of the previously loaded dataset as JSON. "
        "Call load_dataset first."
    )
    args_schema: Type[BaseModel] = PreviewRecordsInput

    def _run(self, state_handle: str = "default", n: int = 5) -> str:
        ds = get_state(state_handle).dataset
        if ds is None:
            return json.dumps({"error": "Dataset not loaded yet — call load_dataset first."})
        return json.dumps(ds.records_json[:n], indent=2)
