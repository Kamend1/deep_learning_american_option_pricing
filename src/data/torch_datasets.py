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
    ) -> None:
        required = [*feature_columns, target_column, id_column]
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
        self.feature_columns = tuple(feature_columns)
        self.target_column = target_column
        self.id_column = id_column

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, object]:
        row_id = self.row_ids[index]
        if isinstance(row_id, np.generic):
            row_id = row_id.item()
        return {
            "features": self.features[index],
            "target": self.targets[index],
            "row_id": row_id,
        }


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
