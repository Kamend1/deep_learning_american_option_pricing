import numpy as np
import pytest

from src.pricing.simulation import (
    GBMContract,
    generate_antithetic_normals,
    sample_contracts_latin_hypercube,
    simulate_contract_paths,
    validate_simulated_moments,
)


def representative_contract() -> GBMContract:
    return GBMContract(
        contract_id="test",
        spot=100.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        dividend_yield=0.02,
        volatility=0.20,
    )


def test_antithetic_normals_are_exact_pairs() -> None:
    shocks = generate_antithetic_normals(
        n_paths=100,
        n_steps=8,
        seed=42,
    )
    assert shocks.shape == (100, 8)
    np.testing.assert_allclose(shocks[:50], -shocks[50:])


def test_simulation_is_reproducible_and_starts_at_spot() -> None:
    contract = representative_contract()
    first = simulate_contract_paths(contract, n_paths=1000, n_steps=12, seed=7)
    second = simulate_contract_paths(contract, n_paths=1000, n_steps=12, seed=7)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1000, 13)
    assert np.all(first[:, 0] == contract.spot)
    assert np.all(first > 0.0)


def test_terminal_moments_are_close_to_theoretical_values() -> None:
    contract = representative_contract()
    paths = simulate_contract_paths(
        contract,
        n_paths=50_000,
        n_steps=12,
        seed=11,
    )
    result = validate_simulated_moments(paths, contract)
    assert result.mean_relative_error < 0.01
    assert result.variance_relative_error < 0.05


def test_latin_hypercube_contract_sampling_is_reproducible() -> None:
    ranges = {
        "moneyness": (0.7, 1.2),
        "time_to_maturity": (0.1, 2.0),
        "risk_free_rate": (0.0, 0.1),
        "dividend_yield": (0.0, 0.08),
        "volatility": (0.1, 0.6),
    }
    first = sample_contracts_latin_hypercube(
        n_contracts=10, parameter_ranges=ranges, seed=42, prefix="a"
    )
    second = sample_contracts_latin_hypercube(
        n_contracts=10, parameter_ranges=ranges, seed=42, prefix="a"
    )
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert all(70.0 <= item.spot <= 120.0 for item in first)


@pytest.mark.parametrize("n_paths", [0, -1])
def test_invalid_path_count_is_rejected(n_paths: int) -> None:
    with pytest.raises(ValueError):
        generate_antithetic_normals(n_paths=n_paths, n_steps=5, seed=42)
