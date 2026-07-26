import pandas as pd

from src.data.torch_datasets import (
    MultiTaskAmericanOptionDataset,
    fit_feature_scaler,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "log_moneyness": [-0.1, 0.0, 0.1],
            "time_to_maturity": [0.5, 1.0, 1.5],
            "risk_free_rate": [0.03, 0.04, 0.05],
            "dividend_yield": [0.01, 0.02, 0.03],
            "volatility": [0.2, 0.3, 0.4],
            "normalized_floor_residual": [0.01, 0.02, 0.03],
            "exercise_now": [0, 1, 0],
            "normalized_european_price": [0.1, 0.2, 0.3],
            "normalized_intrinsic_value": [0.05, 0.21, 0.2],
            "normalized_american_price": [0.11, 0.23, 0.33],
        }
    )


def test_multitask_dataset_shapes_and_row_ids() -> None:
    frame = _frame()
    scaler = fit_feature_scaler(frame)
    dataset = MultiTaskAmericanOptionDataset(frame, scaler=scaler)
    item = dataset[1]
    assert item["features"].shape == (5,)
    assert item["residual_target"].shape == (1,)
    assert item["exercise_target"].shape == (1,)
    assert item["row_id"] == 2
