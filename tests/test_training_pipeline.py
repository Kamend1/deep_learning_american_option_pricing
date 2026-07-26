"""Small end-to-end training-pipeline test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.torch_datasets import (
    AmericanOptionDataset,
    LoaderConfig,
    create_regression_loader,
    fit_feature_scaler,
)
from src.models.direct_pricer import DirectAmericanPutMLP, DirectMLPConfig
from src.training.checkpointing import load_checkpoint
from src.training.loops import TrainingConfig, fit_regression_model


def make_learnable_frame(size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(size, 5))
    target = np.maximum(
        0.0,
        0.12
        + 0.05 * features[:, 0]
        + 0.03 * features[:, 1]
        - 0.02 * features[:, 2],
    )
    return pd.DataFrame(
        {
            "sample_id": np.arange(size),
            "log_moneyness": features[:, 0],
            "time_to_maturity": features[:, 1],
            "risk_free_rate": features[:, 2],
            "dividend_yield": features[:, 3],
            "volatility": features[:, 4],
            "normalized_american_price": target,
        }
    )


def test_training_saves_checkpoint_and_reduces_loss(tmp_path) -> None:
    train = make_learnable_frame(256, 1)
    validation = make_learnable_frame(64, 2)
    scaler = fit_feature_scaler(train)
    train_dataset = AmericanOptionDataset(train, scaler=scaler)
    validation_dataset = AmericanOptionDataset(validation, scaler=scaler)
    loader_config = LoaderConfig(batch_size=64, num_workers=0, seed=42)
    train_loader = create_regression_loader(
        train_dataset,
        config=loader_config,
        shuffle=True,
        drop_last=True,
    )
    validation_loader = create_regression_loader(
        validation_dataset,
        config=loader_config,
        shuffle=False,
    )

    model = DirectAmericanPutMLP(
        DirectMLPConfig(hidden_sizes=(32, 16), batch_norm_after=(0,))
    )
    checkpoint_path = tmp_path / "best.pt"
    history = fit_regression_model(
        model,
        train_loader,
        validation_loader,
        loss_fn=torch.nn.SmoothL1Loss(),
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        config=TrainingConfig(
            epochs=8,
            learning_rate=3e-3,
            early_stopping_patience=8,
            scheduler_patience=3,
            mixed_precision=False,
            seed=42,
        ),
        model_config=model.config.to_dict(),
    )

    assert checkpoint_path.exists()
    assert len(history) >= 2
    assert history["train_loss"].iloc[-1] < history["train_loss"].iloc[0]
    checkpoint = load_checkpoint(checkpoint_path)
    assert "model_state_dict" in checkpoint
