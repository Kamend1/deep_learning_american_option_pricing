"""Tests for scaling, tensor conversion, and DataLoader behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.torch_datasets import (
    AmericanOptionDataset,
    FEATURE_COLUMNS,
    LoaderConfig,
    create_regression_loader,
    fit_feature_scaler,
)


def make_frame(size: int = 32) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "sample_id": np.arange(size),
            "log_moneyness": rng.normal(size=size),
            "time_to_maturity": rng.uniform(0.05, 2.0, size=size),
            "risk_free_rate": rng.uniform(0.0, 0.1, size=size),
            "dividend_yield": rng.uniform(0.0, 0.08, size=size),
            "volatility": rng.uniform(0.05, 0.8, size=size),
            "normalized_american_price": rng.uniform(0.0, 0.6, size=size),
        }
    )


def test_dataset_shapes_dtypes_and_ids() -> None:
    frame = make_frame()
    scaler = fit_feature_scaler(frame)
    dataset = AmericanOptionDataset(frame, scaler=scaler)
    item = dataset[0]

    assert len(dataset) == len(frame)
    assert item["features"].shape == (len(FEATURE_COLUMNS),)
    assert item["features"].dtype == torch.float32
    assert item["target"].shape == (1,)
    assert item["target"].dtype == torch.float32
    assert item["row_id"] == 0


def test_scaler_centers_training_features() -> None:
    frame = make_frame(128)
    scaler = fit_feature_scaler(frame)
    transformed = scaler.transform(frame.loc[:, FEATURE_COLUMNS])
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)


def test_non_shuffled_loader_preserves_row_order() -> None:
    frame = make_frame(20)
    scaler = fit_feature_scaler(frame)
    dataset = AmericanOptionDataset(frame, scaler=scaler)
    loader = create_regression_loader(
        dataset,
        config=LoaderConfig(batch_size=6, num_workers=0, seed=42),
        shuffle=False,
    )
    observed: list[int] = []
    for batch in loader:
        observed.extend(batch["row_id"].tolist())
    assert observed == frame["sample_id"].tolist()
