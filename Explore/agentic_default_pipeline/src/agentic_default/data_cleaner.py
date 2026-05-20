"""Data cleaning utilities for the Taiwan credit-card-default dataset.

Three cleaning passes are applied in order:

1. **Deduplication** — remove exact-duplicate rows.
2. **EDUCATION code remapping** — undocumented codes 0, 5, 6 are remapped to
   4 (Others), consistent with the source paper's documented range {1–4}.
3. **Outlier capping (Winsorisation)** — LIMIT_BAL, BILL_AMT1–6, and
   PAY_AMT1–6 are clipped at the 1st and 99th percentiles to reduce the
   influence of extreme values on model training.

After cleaning, the raw records are re-split and re-scaled so the resulting
``LoadedDataset`` is a drop-in replacement for the one produced by
``data_loader.load_dataset``.

Usage::

    from agentic_default.data_loader import load_dataset
    from agentic_default.data_cleaner import clean_dataset

    raw_ds = load_dataset()
    cleaned_ds, report = clean_dataset(raw_ds)
    print(report.to_dict())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data_loader import (
    SENSITIVE_COLUMNS,
    TARGET_COLUMN,
    DatasetMetadata,
    LoadedDataset,
)


# ---------------------------------------------------------------------------
# Column groups to Winsorise
# ---------------------------------------------------------------------------

_WINSORISE_COLS = [
    "LIMIT_BAL",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",
    "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6",
]

#: EDUCATION codes not described in the original paper (should be → Others=4).
_UNDOCUMENTED_EDUCATION_CODES: frozenset = frozenset({0, 5, 6})
_EDUCATION_REPLACEMENT: int = 4  # "Others"


# ---------------------------------------------------------------------------
# CleaningReport
# ---------------------------------------------------------------------------


@dataclass
class CleaningReport:
    """Summary of every change applied by :func:`clean_dataset`."""

    rows_before: int = 0
    rows_after: int = 0
    duplicates_removed: int = 0
    missing_rows_dropped: int = 0
    education_codes_remapped: int = 0
    #: Winsorisation bounds applied per column {col: {"lower": x, "upper": y}}.
    outliers_capped: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def total_rows_removed(self) -> int:
        """Total rows removed across all cleaning passes."""
        return self.rows_before - self.rows_after

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "total_rows_removed": self.total_rows_removed,
            "duplicates_removed": self.duplicates_removed,
            "missing_rows_dropped": self.missing_rows_dropped,
            "education_codes_remapped": self.education_codes_remapped,
            "outliers_capped": self.outliers_capped,
        }


# ---------------------------------------------------------------------------
# Core cleaning logic
# ---------------------------------------------------------------------------


def clean_records(
    records: List[Dict[str, Any]],
    winsorise_percentile: float = 1.0,
) -> Tuple[List[Dict[str, Any]], CleaningReport]:
    """Apply all cleaning passes to a list of raw JSON records.

    Parameters
    ----------
    records:
        Raw records as returned by :func:`~data_loader.load_dataset` or
        :func:`~data_loader.csv_to_json_records`.
    winsorise_percentile:
        Lower/upper percentile boundary for Winsorisation.
        Default ``1.0`` clips at the 1st and 99th percentiles.

    Returns
    -------
    cleaned_records : list of dict
    report : CleaningReport
    """
    df = pd.DataFrame(records)
    report = CleaningReport(rows_before=len(df))

    # --- Pass 1: Deduplication ----------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    report.duplicates_removed = before - len(df)

    # --- Pass 2: Missing values ---------------------------------------------
    before = len(df)
    df = df.dropna()
    report.missing_rows_dropped = before - len(df)

    # --- Pass 3: EDUCATION code remapping -----------------------------------
    if "EDUCATION" in df.columns:
        mask = df["EDUCATION"].isin(_UNDOCUMENTED_EDUCATION_CODES)
        report.education_codes_remapped = int(mask.sum())
        df.loc[mask, "EDUCATION"] = _EDUCATION_REPLACEMENT

    # --- Pass 4: Winsorisation ----------------------------------------------
    lower_pct = winsorise_percentile
    upper_pct = 100.0 - winsorise_percentile
    for col in _WINSORISE_COLS:
        if col not in df.columns:
            continue
        lo = float(np.percentile(df[col], lower_pct))
        hi = float(np.percentile(df[col], upper_pct))
        report.outliers_capped[col] = {"lower": round(lo, 2), "upper": round(hi, 2)}
        df[col] = df[col].clip(lower=lo, upper=hi)

    report.rows_after = len(df)
    cleaned_records = json.loads(df.to_json(orient="records"))
    return cleaned_records, report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_dataset(
    dataset: LoadedDataset,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
    winsorise_percentile: float = 1.0,
    snapshot_dir: Optional[Path] = None,
) -> Tuple[LoadedDataset, CleaningReport]:
    """Clean the raw records and rebuild a :class:`~data_loader.LoadedDataset`.

    Cleaning is applied to the raw records *before* the train/test split so
    no cleaned rows accidentally appear in only one partition.

    Parameters
    ----------
    dataset:
        Original ``LoadedDataset`` from :func:`~data_loader.load_dataset`.
    test_size, random_state, scale_features:
        Forwarded to the re-split / re-scale step.  Should match the values
        used when the dataset was first loaded.
    winsorise_percentile:
        Percentile bound for Winsorisation (default 1.0 → 1st/99th pct).
    snapshot_dir:
        Optional directory. When supplied, ``cleaned_dataset.json`` and
        ``cleaning_report.json`` are written here.

    Returns
    -------
    cleaned_dataset : LoadedDataset
        Drop-in replacement for the original — same shape convention,
        re-split and re-scaled.
    report : CleaningReport
        Detailed record of every change applied.
    """
    cleaned_records, report = clean_records(
        dataset.records_json, winsorise_percentile=winsorise_percentile
    )

    # --- Rebuild DataFrame and re-split -------------------------------------
    df = pd.DataFrame(cleaned_records)

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column {TARGET_COLUMN!r} not found in cleaned records."
        )

    y = df[TARGET_COLUMN].astype(int).to_numpy()
    x_df = df.drop(columns=[TARGET_COLUMN])
    feature_names = list(x_df.columns)
    x = x_df.to_numpy(dtype=float)

    indices = np.arange(len(x))
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x, y, indices, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale_features:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    # Preserve original sensitive-attribute columns (unscaled) for fairness.
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
        csv_path=dataset.metadata.csv_path,
        extras={
            **dataset.metadata.extras,
            "cleaned": True,
            "duplicates_removed": report.duplicates_removed,
            "missing_rows_dropped": report.missing_rows_dropped,
            "education_codes_remapped": report.education_codes_remapped,
            "winsorise_percentile": winsorise_percentile,
        },
    )

    cleaned_ds = LoadedDataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        metadata=metadata,
        records_json=cleaned_records,
        test_sensitive=test_sensitive,
    )

    # --- Optional snapshot --------------------------------------------------
    if snapshot_dir is not None:
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "cleaned_dataset.json").write_text(
            json.dumps(cleaned_records, indent=2), encoding="utf-8"
        )
        (snapshot_dir / "cleaning_report.json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )

    return cleaned_ds, report
