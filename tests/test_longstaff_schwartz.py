import numpy as np
import pytest

from src.pricing.binomial_tree import crr_option_price
from src.pricing.longstaff_schwartz import (
    evaluate_longstaff_schwartz_policy,
    fit_longstaff_schwartz_policy,
    longstaff_schwartz_put_price,
)
from src.pricing.simulation import GBMContract, simulate_contract_paths


def contract() -> GBMContract:
    return GBMContract(
        contract_id="lsm",
        spot=90.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        dividend_yield=0.01,
        volatility=0.20,
    )


def test_policy_fit_and_independent_valuation_are_finite() -> None:
    item = contract()
    training = simulate_contract_paths(item, n_paths=5000, n_steps=20, seed=1)
    valuation = simulate_contract_paths(item, n_paths=7000, n_steps=20, seed=2)
    policy, diagnostics = fit_longstaff_schwartz_policy(
        training,
        strike=item.strike,
        time_to_maturity=item.time_to_maturity,
        risk_free_rate=item.risk_free_rate,
        basis="polynomial",
        degree=2,
    )
    result = evaluate_longstaff_schwartz_policy(valuation, policy)
    assert result.n_paths == 7000
    assert np.isfinite(result.price)
    assert result.price >= max(item.strike - item.spot, 0.0)
    assert result.standard_error >= 0.0
    assert not diagnostics.empty
    assert all(np.isfinite(model.coefficients).all() for model in policy.regressions.values())


def test_lsm_price_is_reasonably_close_to_crr() -> None:
    item = contract()
    training = simulate_contract_paths(item, n_paths=15_000, n_steps=25, seed=13)
    valuation = simulate_contract_paths(item, n_paths=30_000, n_steps=25, seed=17)
    experiment = longstaff_schwartz_put_price(
        training,
        valuation,
        strike=item.strike,
        time_to_maturity=item.time_to_maturity,
        risk_free_rate=item.risk_free_rate,
        basis="polynomial",
        degree=2,
    )
    benchmark = crr_option_price(
        spot=item.spot,
        strike=item.strike,
        time_to_maturity=item.time_to_maturity,
        risk_free_rate=item.risk_free_rate,
        dividend_yield=item.dividend_yield,
        volatility=item.volatility,
        steps=2000,
        option_type="put",
        exercise_style="american",
    )
    assert experiment.valuation.price == pytest.approx(benchmark, abs=0.35)


def test_nonzero_time_zero_intrinsic_is_respected() -> None:
    item = GBMContract(
        contract_id="deep",
        spot=40.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.10,
        dividend_yield=0.0,
        volatility=0.01,
    )
    training = simulate_contract_paths(item, n_paths=2000, n_steps=10, seed=4)
    valuation = simulate_contract_paths(item, n_paths=3000, n_steps=10, seed=5)
    experiment = longstaff_schwartz_put_price(
        training,
        valuation,
        strike=item.strike,
        time_to_maturity=item.time_to_maturity,
        risk_free_rate=item.risk_free_rate,
    )
    assert experiment.valuation.price >= 60.0
