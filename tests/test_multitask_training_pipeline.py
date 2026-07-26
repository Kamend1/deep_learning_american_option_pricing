from pathlib import Path

import pandas as pd
import torch

from src.data.torch_datasets import (
    LoaderConfig,
    MultiTaskAmericanOptionDataset,
    create_multitask_loader,
    fit_feature_scaler,
)
from src.models.multitask_pricer import MultiTaskAmericanPutMLP
from src.training.multitask_loops import (
    MultiTaskTrainingConfig,
    fit_multitask_model,
    predict_multitask_model,
)
from src.training.multitask_losses import MultiTaskPricingLoss


def _frame(rows: int = 64) -> pd.DataFrame:
    values = []
    for index in range(rows):
        log_m = -0.4 + 0.8 * index / (rows - 1)
        exercise = float(log_m < -0.1)
        european = max(-log_m, 0.0) * 0.3 + 0.02
        intrinsic = max(-log_m, 0.0)
        floor = max(european, intrinsic)
        residual = 0.01 + 0.02 * abs(log_m)
        values.append(
            {
                "sample_id": index,
                "log_moneyness": log_m,
                "time_to_maturity": 1.0,
                "risk_free_rate": 0.05,
                "dividend_yield": 0.01,
                "volatility": 0.2,
                "normalized_floor_residual": residual,
                "exercise_now": exercise,
                "normalized_european_price": european,
                "normalized_intrinsic_value": intrinsic,
                "normalized_american_price": floor + residual,
            }
        )
    return pd.DataFrame(values)


def test_tiny_multitask_pipeline_creates_checkpoint(tmp_path: Path) -> None:
    frame = _frame()
    train = frame.iloc[:48].copy()
    validation = frame.iloc[48:].copy()
    scaler = fit_feature_scaler(train)
    config = LoaderConfig(batch_size=16, seed=42)
    train_loader = create_multitask_loader(
        MultiTaskAmericanOptionDataset(train, scaler=scaler),
        config=config,
        shuffle=True,
    )
    validation_loader = create_multitask_loader(
        MultiTaskAmericanOptionDataset(validation, scaler=scaler),
        config=config,
        shuffle=False,
    )
    model = MultiTaskAmericanPutMLP()
    checkpoint = tmp_path / "best.pt"
    history = fit_multitask_model(
        model,
        train_loader,
        validation_loader,
        loss_fn=MultiTaskPricingLoss(positive_class_weight=1.0),
        config=MultiTaskTrainingConfig(
            epochs=3,
            early_stopping_patience=3,
            scheduler_patience=1,
            mixed_precision=False,
        ),
        device=torch.device("cpu"),
        checkpoint_path=checkpoint,
    )
    assert checkpoint.exists()
    assert len(history) >= 1
    predictions = predict_multitask_model(
        model,
        validation_loader,
        device=torch.device("cpu"),
    )
    assert len(predictions) == len(validation)
    assert predictions["exercise_probability"].between(0.0, 1.0).all()
