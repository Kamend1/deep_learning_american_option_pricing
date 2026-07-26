"""Miniature end-to-end integration test for the final static model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.multihead_targets import add_integrated_targets
from src.data.torch_datasets import (
    IntegratedMultiHeadDataset,
    LoaderConfig,
    create_integrated_multihead_loader,
    fit_feature_scaler,
)
from src.evaluation.integrated_model_comparison import (
    evaluate_integrated_prediction_frame,
)
from src.models.integrated_multihead_pricer import (
    IntegratedAmericanPutMultiHeadMLP,
    IntegratedMultiHeadConfig,
)
from src.training.multihead_losses import (
    IntegratedMultiHeadLoss,
    multihead_loss_preset,
)
from src.training.multihead_loops import (
    IntegratedTrainingConfig,
    fit_integrated_multihead_model,
    predict_integrated_multihead_model,
)


pytestmark = pytest.mark.integration


def _frame(rows: int = 384) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    strike = np.full(rows, 100.0)
    moneyness = rng.uniform(0.6, 1.4, rows)
    spot = strike * moneyness
    maturity = rng.uniform(0.05, 2.0, rows)
    rate = rng.uniform(0.0, 0.10, rows)
    dividend = rng.uniform(0.0, 0.08, rows)
    volatility = rng.uniform(0.08, 0.70, rows)
    intrinsic = np.maximum(strike - spot, 0.0)
    continuation = np.maximum(
        intrinsic + rng.normal(0.0, 2.0, rows),
        0.0,
    )
    exercise = intrinsic >= continuation - 1e-12
    european = np.maximum(intrinsic - rng.uniform(0.0, 1.0, rows), 0.0)
    floor = np.maximum(european, intrinsic)
    residual = rng.uniform(0.0, 1.5, rows)
    american = floor + residual
    split = np.array(["train"] * 256 + ["validation"] * 64 + ["test"] * 64)
    return pd.DataFrame(
        {
            "sample_id": np.arange(rows),
            "split": split,
            "spot": spot,
            "strike": strike,
            "moneyness": moneyness,
            "log_moneyness": np.log(moneyness),
            "time_to_maturity": maturity,
            "risk_free_rate": rate,
            "dividend_yield": dividend,
            "volatility": volatility,
            "intrinsic_value": intrinsic,
            "continuation_value": continuation,
            "european_price": european,
            "american_price": american,
            "exercise_now": exercise,
        }
    )


def test_integrated_static_pipeline(tmp_path: Path) -> None:
    frame = add_integrated_targets(_frame(), copy=False)
    train = frame.loc[frame["split"] == "train"].copy()
    validation = frame.loc[frame["split"] == "validation"].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    scaler = fit_feature_scaler(train)
    loader_config = LoaderConfig(batch_size=64, num_workers=0, seed=42)

    def loader(data: pd.DataFrame, *, shuffle: bool):
        dataset = IntegratedMultiHeadDataset(data, scaler=scaler)
        return create_integrated_multihead_loader(
            dataset,
            config=loader_config,
            shuffle=shuffle,
            drop_last=shuffle,
        )

    train_loader = loader(train, shuffle=True)
    validation_loader = loader(validation, shuffle=False)
    test_loader = loader(test, shuffle=False)
    model_config = IntegratedMultiHeadConfig(
        shared_hidden_sizes=(32, 24),
        batch_norm_after=(0,),
        dropout=0.0,
        residual_head_sizes=(12,),
        direct_head_sizes=(12,),
        continuation_head_sizes=(12,),
        exercise_head_sizes=(12,),
    )
    model = IntegratedAmericanPutMultiHeadMLP(model_config)
    loss_fn = IntegratedMultiHeadLoss(
        config=multihead_loss_preset("balanced"),
        positive_class_weight=1.0,
    )
    checkpoint = tmp_path / "integrated.pt"
    history = fit_integrated_multihead_model(
        model,
        train_loader,
        validation_loader,
        loss_fn=loss_fn,
        config=IntegratedTrainingConfig(
            epochs=3,
            early_stopping_patience=3,
            scheduler_patience=2,
            mixed_precision=False,
            seed=42,
        ),
        device=torch.device("cpu"),
        checkpoint_path=checkpoint,
        model_config=model_config.to_dict(),
    )

    assert checkpoint.exists()
    assert len(history) >= 1
    reloaded = IntegratedAmericanPutMultiHeadMLP(model_config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(state["model_state_dict"])
    predictions = predict_integrated_multihead_model(
        reloaded,
        test_loader,
        device=torch.device("cpu"),
    )
    metrics = evaluate_integrated_prediction_frame(predictions)

    assert len(predictions) == len(test)
    assert np.isfinite(predictions.select_dtypes(include=[np.number])).all().all()
    assert metrics["constrained_observations"] == float(len(test))
    assert (
        predictions["predicted_normalized_american_price"]
        >= predictions["normalized_european_price"] - 1e-8
    ).all()
    assert (
        predictions["predicted_normalized_american_price"]
        >= predictions["normalized_intrinsic_value"] - 1e-8
    ).all()
