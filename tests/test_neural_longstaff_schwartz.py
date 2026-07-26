from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.neural_longstaff_schwartz import (
    ContinuationNetworkConfig,
    ContinuationValueNetwork,
    ContractPathBatch,
    NeuralLSMPolicy,
)
from src.pricing.simulation import GBMContract, simulate_contract_paths
from src.training.lsm_training import (
    NeuralLSMTrainingConfig,
    evaluate_neural_lsm_policy,
    fit_neural_lsm_policy,
    validate_contract_separation,
)


def make_batch(contract_id: str, spot: float, seed: int) -> ContractPathBatch:
    contract = GBMContract(
        contract_id=contract_id,
        spot=spot,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        dividend_yield=0.01,
        volatility=0.25,
    )
    paths = simulate_contract_paths(contract, n_paths=700, n_steps=5, seed=seed)
    return ContractPathBatch(contract=contract, paths=paths)


def test_continuation_network_output_is_nonnegative_and_differentiable() -> None:
    model = ContinuationValueNetwork(
        ContinuationNetworkConfig(hidden_dims=(16, 8))
    )
    features = torch.randn(32, 5, requires_grad=True)
    output = model(features)
    assert output.shape == (32,)
    assert torch.all(output >= 0.0)
    output.mean().backward()
    assert features.grad is not None


def test_contract_leakage_is_rejected() -> None:
    batch = make_batch("same", 90.0, 1)
    with pytest.raises(ValueError):
        validate_contract_separation([batch], [batch])


def test_small_neural_policy_can_train_evaluate_and_serialize(tmp_path: Path) -> None:
    training = [
        make_batch("train_1", 85.0, 1),
        make_batch("train_2", 95.0, 2),
        make_batch("train_3", 105.0, 3),
    ]
    validation = [make_batch("validation_1", 90.0, 4)]
    test_batch = make_batch("test_1", 92.0, 5)
    config = NeuralLSMTrainingConfig(
        network=ContinuationNetworkConfig(hidden_dims=(16, 8)),
        epochs=3,
        batch_size=128,
        patience=2,
        minimum_samples_per_step=16,
        maximum_samples_per_step=2000,
        seed=42,
        device="cpu",
    )
    policy, history = fit_neural_lsm_policy(training, validation, config=config)
    assert policy.n_steps == 5
    assert policy.steps
    assert not history.empty
    result = evaluate_neural_lsm_policy(policy, test_batch)
    assert np.isfinite(result.price)
    assert result.price >= max(
        test_batch.contract.strike - test_batch.contract.spot, 0.0
    )

    path = tmp_path / "policy.pt"
    policy.save(path)
    loaded = NeuralLSMPolicy.load(path)
    loaded_result = evaluate_neural_lsm_policy(loaded, test_batch)
    assert loaded_result.price == pytest.approx(result.price, abs=1e-10)
