"""Data loading utilities for the Taiwan default-of-credit-card-clients dataset.

This module loads the source CSV, converts it to JSON-friendly Python
structures, performs a deterministic train/test split, and provides metadata
that downstream agents can describe in natural language.

It also keeps the **unscaled, original** values of the sensitive demographic
columns for the test split. The fairness pipeline reads from those — the
scaled feature matrix is no good for grouping rows back into demographic
buckets like "male" / "female" or "married" / "single".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------- public constants ------------------------------------------------

#: Default location of the bundled CSV (one folder up from this package).
DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parents[3] / "default_of_credit_card_clients.csv"
)

#: Name of the binary target column in the source CSV.
TARGET_COLUMN = "default payment next month"

#: Categorical columns that benefit from being treated as discrete.
CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE"]

#: PAY_* columns are repayment-status codes (categorical-ish but ordinal).
PAY_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

#: Sensitive columns the fairness pipeline buckets predictions over.
SENSITIVE_COLUMNS = ["SEX", "AGE", "MARRIAGE"]


# ---------- data containers -------------------------------------------------


@dataclass
class DatasetMetadata:
    """Lightweight summary of the loaded dataset."""

    n_rows: int
    n_features: int
    feature_names: List[str]
    target_name: str
    class_balance: Dict[str, int]
    train_size: int
    test_size: int
    csv_path: str
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


@dataclass
class LoadedDataset:
    """Container with the train/test arrays plus metadata."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    metadata: DatasetMetadata
    records_json: List[Dict[str, Any]]
    #: Original, unscaled values of SENSITIVE_COLUMNS for the test rows,
    #: aligned with x_test/y_test ordering. Used by the fairness pipeline.
    test_sensitive: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------- public API ------------------------------------------------------


def csv_to_json_records(
    csv_path: Optional[Path] = None,
    sample: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load the credit-card-default CSV as a list of JSON records."""
    csv_path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if sample is not None:
        df = df.head(sample)
    return json.loads(df.to_json(orient="records"))


def write_json_snapshot(
    records: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Persist a list of records as a UTF-8 JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return output_path


def load_dataset(
    csv_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
    snapshot_dir: Optional[Path] = None,
) -> LoadedDataset:
    """Load the dataset and produce a clean train/test split."""
    csv_path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column {TARGET_COLUMN!r} missing from CSV {csv_path}.")

    y = df[TARGET_COLUMN].astype(int).to_numpy()
    x_df = df.drop(columns=[TARGET_COLUMN])
    feature_names = list(x_df.columns)
    x = x_df.to_numpy(dtype=float)

    # We need the indices so we can carry the original sensitive columns
    # through to the test split.
    indices = np.arange(len(x))
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x, y, indices, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale_features:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    # Build the test-side sensitive-attribute frame from the *unscaled* values.
    sensitive_present = [c for c in SENSITIVE_COLUMNS if c in x_df.columns]
    test_sensitive = x_df.iloc[idx_test][sensitive_present].reset_index(drop=True)

    class_counts = {
        str(int(k)): int(v)
        for k, v in zip(*np.unique(y, return_counts=True))
    }

    metadata = DatasetMetadata(
        n_rows=int(len(df)),
        n_features=len(feature_names),
        feature_names=feature_names,
        target_name=TARGET_COLUMN,
        class_balance=class_counts,
        train_size=int(len(x_train)),
        test_size=int(len(x_test)),
        csv_path=str(csv_path),
        extras={
            "categorical_columns": CATEGORICAL_COLUMNS,
            "pay_columns": PAY_COLUMNS,
            "sensitive_columns": sensitive_present,
            "scaled": bool(scale_features),
            "random_state": random_state,
        },
    )

    records_json = json.loads(df.to_json(orient="records"))

    if snapshot_dir is not None:
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        write_json_snapshot(records_json, snapshot_dir / "dataset.json")
        write_json_snapshot([metadata.to_dict()], snapshot_dir / "metadata.json")

    return LoadedDataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        metadata=metadata,
        records_json=records_json,
        test_sensitive=test_sensitive,
    )


def summarize_for_agent(metadata: DatasetMetadata, preview: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a compact JSON-friendly description for an agent prompt."""
    return {
        "summary": {
            "rows": metadata.n_rows,
            "features": metadata.n_features,
            "target": metadata.target_name,
            "class_balance": metadata.class_balance,
            "train_size": metadata.train_size,
            "test_size": metadata.test_size,
        },
        "feature_names": metadata.feature_names,
        "preview_records": preview,
    }


def split_arrays_to_dict(dataset: LoadedDataset) -> Dict[str, Any]:
    """Shape the LoadedDataset into a plain dict (useful for serialization)."""
    return {
        "x_train_shape": list(dataset.x_train.shape),
        "x_test_shape": list(dataset.x_test.shape),
        "feature_names": dataset.feature_names,
        "metadata": dataset.metadata.to_dict(),
    }


def quick_class_imbalance_ratio(y: np.ndarray) -> Tuple[float, str]:
    """Return (majority_fraction, description) for a binary label array."""
    unique, counts = np.unique(y, return_counts=True)
    fractions = counts / counts.sum()
    majority = float(fractions.max())
    minority = float(fractions.min())
    desc = (
        f"binary target with majority class fraction={majority:.3f} "
        f"and minority fraction={minority:.3f}"
    )
    return majority, desc
