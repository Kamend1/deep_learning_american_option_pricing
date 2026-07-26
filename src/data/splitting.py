"""Deterministic dataset splitting and out-of-domain selection utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Configuration for deterministic train/validation/test allocation."""

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    stratify_column: str | None = "exercise_now"
    id_column: str = "sample_id"
    split_column: str = "split"

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(fraction <= 0.0 for fraction in fractions):
            raise ValueError("All split fractions must be positive.")
        if not np.isclose(sum(fractions), 1.0, atol=1e-12):
            raise ValueError("Split fractions must sum to 1.0.")
        if not self.id_column:
            raise ValueError("id_column cannot be empty.")
        if not self.split_column:
            raise ValueError("split_column cannot be empty.")


@dataclass(frozen=True, slots=True)
class OODRangeSpec:
    """Range-based definition of one out-of-domain evaluation set."""

    name: str
    description: str
    ranges: Mapping[str, tuple[float | None, float | None]]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("OOD specification name cannot be empty.")
        if not self.ranges:
            raise ValueError("OOD specification must contain at least one range.")
        for column, (lower, upper) in self.ranges.items():
            if not column:
                raise ValueError("OOD range column names cannot be empty.")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(
                    f"OOD lower bound cannot exceed upper bound for {column}."
                )


DEFAULT_OOD_SPECS: tuple[OODRangeSpec, ...] = (
    OODRangeSpec(
        name="high_volatility",
        description="Volatility above the core training maximum.",
        ranges={"volatility": (0.80, 1.20)},
    ),
    OODRangeSpec(
        name="long_maturity",
        description="Maturities beyond the two-year core training horizon.",
        ranges={"time_to_maturity": (2.00, 4.00)},
    ),
    OODRangeSpec(
        name="deep_in_the_money",
        description="Spot-to-strike ratios below the core domain.",
        ranges={"moneyness": (0.25, 0.50)},
    ),
    OODRangeSpec(
        name="deep_out_of_the_money",
        description="Spot-to-strike ratios above the core domain.",
        ranges={"moneyness": (1.50, 2.00)},
    ),
    OODRangeSpec(
        name="high_interest_rate",
        description="Interest rates above the core training maximum.",
        ranges={"risk_free_rate": (0.10, 0.20)},
    ),
    OODRangeSpec(
        name="high_dividend_yield",
        description="Dividend yields above the core training maximum.",
        ranges={"dividend_yield": (0.08, 0.15)},
    ),
)


def _allocate_counts(size: int, fractions: tuple[float, float, float]) -> np.ndarray:
    """Allocate an integer group size by the largest-remainder method."""

    exact = np.asarray(fractions, dtype=float) * size
    counts = np.floor(exact).astype(int)
    remainder = int(size - counts.sum())
    if remainder:
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def create_train_validation_test_split(
    dataset: pd.DataFrame,
    *,
    config: SplitConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Create deterministic, disjoint splits with optional stratification."""

    cfg = config or SplitConfig()
    if dataset.empty:
        raise ValueError("Dataset cannot be empty.")
    if cfg.id_column not in dataset.columns:
        raise ValueError(f"Dataset is missing id column {cfg.id_column!r}.")
    if dataset[cfg.id_column].isna().any():
        raise ValueError("Split identifiers cannot be missing.")
    if dataset[cfg.id_column].duplicated().any():
        raise ValueError("Split identifiers must be unique.")
    if cfg.split_column in dataset.columns:
        raise ValueError(
            f"Dataset already contains reserved split column {cfg.split_column!r}."
        )
    if cfg.stratify_column is not None and cfg.stratify_column not in dataset.columns:
        raise ValueError(
            f"Dataset is missing stratification column {cfg.stratify_column!r}."
        )

    rng = np.random.default_rng(cfg.seed)
    assignments = pd.Series(index=dataset.index, dtype="object")
    fractions = (
        cfg.train_fraction,
        cfg.validation_fraction,
        cfg.test_fraction,
    )
    labels = np.asarray(["train", "validation", "test"], dtype=object)

    if cfg.stratify_column is None:
        groups = [(None, dataset.index.to_numpy())]
    else:
        if dataset[cfg.stratify_column].isna().any():
            raise ValueError("Stratification values cannot be missing.")
        groups = [
            (value, group.index.to_numpy())
            for value, group in dataset.groupby(cfg.stratify_column, sort=True)
        ]

    for _, group_indices in groups:
        shuffled = rng.permutation(group_indices)
        counts = _allocate_counts(len(shuffled), fractions)
        boundaries = np.cumsum(counts)
        start = 0
        for label, end in zip(labels, boundaries, strict=True):
            assignments.loc[shuffled[start:end]] = label
            start = int(end)

    if assignments.isna().any():
        raise RuntimeError("Internal split error: some rows were not assigned.")

    result: dict[str, pd.DataFrame] = {}
    for label in labels:
        subset = dataset.loc[assignments == label].copy()
        subset[cfg.split_column] = label
        result[str(label)] = subset.reset_index(drop=True)

    integrity = validate_split_integrity(result, config=cfg)
    if not bool(integrity["passed"].all()):
        failed = integrity.loc[~integrity["passed"], "check"].tolist()
        raise RuntimeError(f"Generated split failed integrity checks: {failed}")

    return result


def validate_split_integrity(
    splits: Mapping[str, pd.DataFrame],
    *,
    config: SplitConfig | None = None,
) -> pd.DataFrame:
    """Verify split names, uniqueness, disjointness, and assignment labels."""

    cfg = config or SplitConfig()
    required_names = {"train", "validation", "test"}
    observed_names = set(splits)
    records: list[dict[str, object]] = []

    records.append(
        {
            "check": "Required split names are present",
            "observed": sorted(observed_names),
            "expected": sorted(required_names),
            "passed": observed_names == required_names,
        }
    )

    if observed_names != required_names:
        return pd.DataFrame.from_records(records)

    all_ids: list[object] = []
    total_rows = 0
    labels_valid = True
    ids_available = True
    for name in ("train", "validation", "test"):
        frame = splits[name]
        total_rows += len(frame)
        if cfg.id_column not in frame.columns:
            ids_available = False
        else:
            all_ids.extend(frame[cfg.id_column].tolist())
        if cfg.split_column not in frame.columns or not (frame[cfg.split_column] == name).all():
            labels_valid = False

    duplicate_ids = len(all_ids) - len(set(all_ids)) if ids_available else None
    records.extend(
        [
            {
                "check": "ID column exists in every split",
                "observed": ids_available,
                "expected": True,
                "passed": ids_available,
            },
            {
                "check": "Split identifiers are disjoint",
                "observed": duplicate_ids,
                "expected": 0,
                "passed": ids_available and duplicate_ids == 0,
            },
            {
                "check": "Stored split labels match their containers",
                "observed": labels_valid,
                "expected": True,
                "passed": labels_valid,
            },
            {
                "check": "All split rows are represented by unique IDs",
                "observed": len(set(all_ids)) if ids_available else None,
                "expected": total_rows,
                "passed": ids_available and len(set(all_ids)) == total_rows,
            },
        ]
    )

    return pd.DataFrame.from_records(records)


def build_split_manifest(
    splits: Mapping[str, pd.DataFrame],
    *,
    config: SplitConfig | None = None,
    dataset_name: str = "american_put_core",
) -> dict[str, object]:
    """Build a JSON-serializable manifest for frozen split assignments."""

    cfg = config or SplitConfig()
    counts = {name: int(len(frame)) for name, frame in splits.items()}
    total = int(sum(counts.values()))

    exercise_distribution: dict[str, dict[str, float | int]] = {}
    if cfg.stratify_column is not None:
        for name, frame in splits.items():
            if cfg.stratify_column in frame.columns:
                counts_by_class = frame[cfg.stratify_column].value_counts().to_dict()
                exercise_distribution[name] = {
                    str(key): int(value) for key, value in counts_by_class.items()
                }

    return {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "split_config": asdict(cfg),
        "total_observations": total,
        "split_counts": counts,
        "split_shares": {
            name: (count / total if total else 0.0) for name, count in counts.items()
        },
        "stratification_distribution": exercise_distribution,
    }


def save_split_assignments(
    splits: Mapping[str, pd.DataFrame],
    path: str | Path,
    *,
    config: SplitConfig | None = None,
) -> Path:
    """Save only identifiers and frozen split labels as CSV or Parquet."""

    cfg = config or SplitConfig()
    rows = []
    for name, frame in splits.items():
        if cfg.id_column not in frame.columns:
            raise ValueError(f"Split {name!r} is missing {cfg.id_column!r}.")
        rows.append(
            frame[[cfg.id_column]].assign(**{cfg.split_column: name})
        )

    assignments = pd.concat(rows, ignore_index=True).sort_values(cfg.id_column)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        assignments.to_csv(output_path, index=False)
    elif output_path.suffix.lower() == ".parquet":
        assignments.to_parquet(output_path, index=False)
    else:
        raise ValueError("Split assignment path must end in .csv or .parquet.")
    return output_path


def save_split_manifest(manifest: Mapping[str, object], path: str | Path) -> Path:
    """Save a split manifest as indented JSON."""

    output_path = Path(path)
    if output_path.suffix.lower() != ".json":
        raise ValueError("Split manifest path must end in .json.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def create_out_of_domain_sets(
    dataset: pd.DataFrame,
    *,
    specs: tuple[OODRangeSpec, ...] = DEFAULT_OOD_SPECS,
) -> dict[str, pd.DataFrame]:
    """Select named OOD subsets from a separately generated stress pool.

    Bounds are inclusive. A row must satisfy every range in a specification.
    Overlap across different OOD sets is allowed and remains visible.
    """

    result: dict[str, pd.DataFrame] = {}
    for spec in specs:
        mask = pd.Series(True, index=dataset.index)
        for column, (lower, upper) in spec.ranges.items():
            if column not in dataset.columns:
                raise ValueError(
                    f"Dataset is missing OOD column {column!r} for {spec.name!r}."
                )
            if lower is not None:
                mask &= dataset[column] >= lower
            if upper is not None:
                mask &= dataset[column] <= upper
        subset = dataset.loc[mask].copy()
        subset["ood_set"] = spec.name
        result[spec.name] = subset.reset_index(drop=True)
    return result


def validate_ood_exclusion(
    training_set: pd.DataFrame,
    ood_sets: Mapping[str, pd.DataFrame],
    *,
    id_column: str = "sample_id",
) -> pd.DataFrame:
    """Verify that no OOD identifiers appear in the training split."""

    if id_column not in training_set.columns:
        raise ValueError(f"Training set is missing {id_column!r}.")
    training_ids = set(training_set[id_column])
    records = []
    for name, frame in ood_sets.items():
        if id_column not in frame.columns:
            raise ValueError(f"OOD set {name!r} is missing {id_column!r}.")
        overlap = training_ids.intersection(frame[id_column])
        records.append(
            {
                "ood_set": name,
                "observations": int(len(frame)),
                "overlap_with_training": int(len(overlap)),
                "passed": len(overlap) == 0,
            }
        )
    return pd.DataFrame.from_records(records)


__all__ = [
    "DEFAULT_OOD_SPECS",
    "OODRangeSpec",
    "SplitConfig",
    "build_split_manifest",
    "create_out_of_domain_sets",
    "create_train_validation_test_split",
    "save_split_assignments",
    "save_split_manifest",
    "validate_ood_exclusion",
    "validate_split_integrity",
]
