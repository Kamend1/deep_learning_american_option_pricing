"""PyTorch dataset and DataLoader utilities for American option pricing."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset


FEATURE_COLUMNS: tuple[str, ...] = (
    "log_moneyness",
    "time_to_maturity",
    "risk_free_rate",
    "dividend_yield",
    "volatility",
)
DIRECT_TARGET_COLUMN = "normalized_american_price"


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    """Configuration for deterministic regression DataLoaders."""

    batch_size: int = 1024
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")


def read_parquet_components(
    paths: Sequence[str | Path],
    *,
    columns: Sequence[str],
    split: str | None = None,
    row_limit: int | None = None,
) -> pd.DataFrame:
    """Read selected columns from one or more production Parquet components."""

    if not paths:
        raise ValueError("At least one Parquet path is required.")
    frames: list[pd.DataFrame] = []
    for path_like in paths:
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(path)
        filters = [("split", "=", split)] if split is not None else None
        frame = pd.read_parquet(path, columns=list(columns), filters=filters)
        if row_limit is not None:
            remaining = row_limit - sum(len(existing) for existing in frames)
            if remaining <= 0:
                break
            frame = frame.head(remaining)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(columns))
    result = pd.concat(frames, ignore_index=True)
    if row_limit is not None:
        result = result.head(row_limit).copy()
    return result


def fit_feature_scaler(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> StandardScaler:
    """Fit a StandardScaler on training observations only."""

    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Training frame is missing feature columns: {missing}")
    values = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Training features contain NaN or infinite values.")
    scaler = StandardScaler()
    scaler.fit(values)
    return scaler


def save_feature_scaler(scaler: StandardScaler, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output)
    return output


def load_feature_scaler(path: str | Path) -> StandardScaler:
    scaler = joblib.load(Path(path))
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Saved object is not a StandardScaler.")
    return scaler


class AmericanOptionDataset(Dataset[dict[str, object]]):
    """In-memory tensor dataset preserving source row identifiers."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        scaler: StandardScaler,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        target_column: str = DIRECT_TARGET_COLUMN,
        id_column: str = "sample_id",
        weight_column: str | None = None,
    ) -> None:
        required = [*feature_columns, target_column, id_column]
        if weight_column is not None:
            required.append(weight_column)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Dataset frame is missing columns: {missing}")

        feature_values = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
        target_values = frame[target_column].to_numpy(dtype=np.float64)
        if not np.isfinite(feature_values).all():
            raise ValueError("Features contain NaN or infinite values.")
        if not np.isfinite(target_values).all():
            raise ValueError("Targets contain NaN or infinite values.")

        scaled = scaler.transform(feature_values).astype(np.float32, copy=False)
        self.features = torch.from_numpy(scaled)
        self.targets = torch.from_numpy(
            target_values.astype(np.float32, copy=False)
        ).unsqueeze(1)
        self.row_ids = frame[id_column].to_numpy(copy=True)
        self.sample_weights = None
        if weight_column is not None:
            weight_values = frame[weight_column].to_numpy(dtype=np.float64)
            if not np.isfinite(weight_values).all():
                raise ValueError("Sample weights contain NaN or infinite values.")
            if np.any(weight_values < 0.0):
                raise ValueError("Sample weights cannot be negative.")
            self.sample_weights = torch.from_numpy(
                weight_values.astype(np.float32, copy=False)
            ).unsqueeze(1)
        self.feature_columns = tuple(feature_columns)
        self.target_column = target_column
        self.id_column = id_column
        self.weight_column = weight_column

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, object]:
        row_id = self.row_ids[index]
        if isinstance(row_id, np.generic):
            row_id = row_id.item()
        item: dict[str, object] = {
            "features": self.features[index],
            "target": self.targets[index],
            "row_id": row_id,
        }
        if self.sample_weights is not None:
            item["sample_weight"] = self.sample_weights[index]
        return item


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_regression_loader(
    dataset: AmericanOptionDataset,
    *,
    config: LoaderConfig,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    """Create a reproducible DataLoader for one split."""

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=_seed_worker if config.num_workers else None,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )


__all__ = [
    "AmericanOptionDataset",
    "DIRECT_TARGET_COLUMN",
    "FEATURE_COLUMNS",
    "LoaderConfig",
    "create_regression_loader",
    "fit_feature_scaler",
    "load_feature_scaler",
    "read_parquet_components",
    "save_feature_scaler",
]


class MultiTaskAmericanOptionDataset(Dataset[dict[str, object]]):
    """Tensor dataset for residual regression and exercise classification."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        scaler: StandardScaler,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        residual_target_column: str = "normalized_floor_residual",
        exercise_target_column: str = "exercise_now",
        id_column: str = "sample_id",
        weight_column: str | None = None,
        normalized_european_column: str = "normalized_european_price",
        normalized_intrinsic_column: str = "normalized_intrinsic_value",
        normalized_american_column: str = "normalized_american_price",
    ) -> None:
        required = [
            *feature_columns,
            residual_target_column,
            exercise_target_column,
            id_column,
            normalized_european_column,
            normalized_intrinsic_column,
            normalized_american_column,
        ]
        if weight_column is not None:
            required.append(weight_column)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Dataset frame is missing columns: {missing}")

        feature_values = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
        residual_values = frame[residual_target_column].to_numpy(dtype=np.float64)
        exercise_values = frame[exercise_target_column].to_numpy(dtype=np.float64)
        european_values = frame[normalized_european_column].to_numpy(dtype=np.float64)
        intrinsic_values = frame[normalized_intrinsic_column].to_numpy(dtype=np.float64)
        american_values = frame[normalized_american_column].to_numpy(dtype=np.float64)

        arrays = (
            feature_values,
            residual_values,
            exercise_values,
            european_values,
            intrinsic_values,
            american_values,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Multi-task dataset contains NaN or infinite values.")
        if not np.isin(exercise_values, [0.0, 1.0]).all():
            raise ValueError("Exercise targets must be binary.")

        scaled = scaler.transform(feature_values).astype(np.float32, copy=False)
        self.features = torch.from_numpy(scaled)
        self.residual_targets = torch.from_numpy(
            residual_values.astype(np.float32, copy=False)
        ).unsqueeze(1)
        self.exercise_targets = torch.from_numpy(
            exercise_values.astype(np.float32, copy=False)
        ).unsqueeze(1)
        self.normalized_european = torch.from_numpy(
            european_values.astype(np.float32, copy=False)
        ).unsqueeze(1)
        self.normalized_intrinsic = torch.from_numpy(
            intrinsic_values.astype(np.float32, copy=False)
        ).unsqueeze(1)
        self.normalized_american = torch.from_numpy(
            american_values.astype(np.float32, copy=False)
        ).unsqueeze(1)
        self.row_ids = frame[id_column].to_numpy(copy=True)
        self.sample_weights = None
        if weight_column is not None:
            weight_values = frame[weight_column].to_numpy(dtype=np.float64)
            if not np.isfinite(weight_values).all() or np.any(weight_values < 0.0):
                raise ValueError("Sample weights must be finite and non-negative.")
            self.sample_weights = torch.from_numpy(
                weight_values.astype(np.float32, copy=False)
            ).unsqueeze(1)

        self.feature_columns = tuple(feature_columns)
        self.residual_target_column = residual_target_column
        self.exercise_target_column = exercise_target_column
        self.id_column = id_column
        self.weight_column = weight_column

    def __len__(self) -> int:
        return len(self.residual_targets)

    def __getitem__(self, index: int) -> dict[str, object]:
        row_id = self.row_ids[index]
        if isinstance(row_id, np.generic):
            row_id = row_id.item()
        item: dict[str, object] = {
            "features": self.features[index],
            "residual_target": self.residual_targets[index],
            "exercise_target": self.exercise_targets[index],
            "normalized_european": self.normalized_european[index],
            "normalized_intrinsic": self.normalized_intrinsic[index],
            "normalized_american": self.normalized_american[index],
            "row_id": row_id,
        }
        if self.sample_weights is not None:
            item["sample_weight"] = self.sample_weights[index]
        return item


def create_multitask_loader(
    dataset: MultiTaskAmericanOptionDataset,
    *,
    config: LoaderConfig,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    """Create a deterministic DataLoader for multi-task observations."""

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=_seed_worker if config.num_workers else None,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )


__all__.extend(
    [
        "MultiTaskAmericanOptionDataset",
        "create_multitask_loader",
    ]
)
