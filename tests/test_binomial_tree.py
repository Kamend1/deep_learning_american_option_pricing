"""Tests for the Cox–Ross–Rubinstein pricing engine and diagnostics."""

import pytest

from src.pricing.black_scholes import (
    black_scholes_call_price,
    black_scholes_put_price,
)
from src.pricing.binomial_tree import (
    CRRPriceResult,
    crr_option_diagnostics,
    crr_option_price,
)


BASE_CASE = {
    "spot": 100.0,
    "strike": 100.0,
    "time_to_maturity": 1.0,
    "risk_free_rate": 0.05,
    "volatility": 0.20,
    "dividend_yield": 0.0,
}


def test_european_put_tree_converges_to_black_scholes() -> None:
    tree = crr_option_price(
        **BASE_CASE,
        steps=2_000,
        option_type="put",
        exercise_style="european",
    )
    analytical = black_scholes_put_price(**BASE_CASE)
    assert tree == pytest.approx(analytical, abs=0.01)


def test_european_call_tree_converges_to_black_scholes() -> None:
    tree = crr_option_price(
        **BASE_CASE,
        steps=2_000,
        option_type="call",
        exercise_style="european",
    )
    analytical = black_scholes_call_price(**BASE_CASE)
    assert tree == pytest.approx(analytical, abs=0.01)


def test_american_put_is_not_below_european_or_intrinsic() -> None:
    params = {**BASE_CASE, "spot": 85.0, "risk_free_rate": 0.07}
    european = crr_option_price(
        **params,
        steps=1_000,
        option_type="put",
        exercise_style="european",
    )
    american = crr_option_diagnostics(
        **params,
        steps=1_000,
        option_type="put",
        exercise_style="american",
    )
    assert american.price >= european
    assert american.price >= american.intrinsic_value


def test_non_dividend_american_call_matches_european_call() -> None:
    european = crr_option_price(
        **BASE_CASE,
        steps=1_000,
        option_type="call",
        exercise_style="european",
    )
    american = crr_option_price(
        **BASE_CASE,
        steps=1_000,
        option_type="call",
        exercise_style="american",
    )
    assert american == pytest.approx(european, abs=1e-10)


def test_deep_in_the_money_put_is_exercised_at_root() -> None:
    result = crr_option_diagnostics(
        spot=50.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.10,
        dividend_yield=0.0,
        volatility=0.10,
        steps=1_000,
        option_type="put",
        exercise_style="american",
    )
    assert isinstance(result, CRRPriceResult)
    assert result.exercise_now
    assert result.intrinsic_value >= result.continuation_value - 1e-12
    assert result.price == pytest.approx(result.intrinsic_value, abs=1e-10)


def test_out_of_the_money_put_continues_at_root() -> None:
    result = crr_option_diagnostics(
        **{**BASE_CASE, "spot": 120.0},
        steps=500,
        option_type="put",
        exercise_style="american",
    )
    assert not result.exercise_now
    assert result.continuation_value > result.intrinsic_value


def test_scalar_wrapper_matches_diagnostics_price() -> None:
    kwargs = {
        **BASE_CASE,
        "steps": 500,
        "option_type": "put",
        "exercise_style": "american",
    }
    assert crr_option_price(**kwargs) == pytest.approx(
        crr_option_diagnostics(**kwargs).price,
        abs=0.0,
    )


def test_tree_price_is_stable_as_steps_increase() -> None:
    price_1_000 = crr_option_price(
        **BASE_CASE,
        steps=1_000,
        option_type="put",
        exercise_style="american",
    )
    price_2_000 = crr_option_price(
        **BASE_CASE,
        steps=2_000,
        option_type="put",
        exercise_style="american",
    )
    assert price_1_000 == pytest.approx(price_2_000, abs=0.01)


@pytest.mark.parametrize("exercise_style", ["european", "american"])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_value_at_expiry_equals_intrinsic(
    exercise_style: str,
    option_type: str,
) -> None:
    params = {**BASE_CASE, "spot": 80.0, "time_to_maturity": 0.0}
    result = crr_option_diagnostics(
        **params,
        steps=10,
        option_type=option_type,
        exercise_style=exercise_style,
    )
    expected = (
        max(params["spot"] - params["strike"], 0.0)
        if option_type == "call"
        else max(params["strike"] - params["spot"], 0.0)
    )
    assert result.price == pytest.approx(expected)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("spot", 0.0),
        ("strike", 0.0),
        ("time_to_maturity", -0.01),
        ("volatility", -0.01),
        ("steps", 0),
    ],
)
def test_invalid_numeric_inputs_raise_value_error(
    field: str,
    invalid_value: float,
) -> None:
    kwargs = {
        **BASE_CASE,
        "steps": 100,
        "option_type": "put",
        "exercise_style": "american",
    }
    kwargs[field] = invalid_value
    with pytest.raises(ValueError):
        crr_option_price(**kwargs)


@pytest.mark.parametrize(
    ("option_type", "exercise_style"),
    [("straddle", "american"), ("put", "bermudan")],
)
def test_invalid_enum_inputs_raise_value_error(
    option_type: str,
    exercise_style: str,
) -> None:
    with pytest.raises(ValueError):
        crr_option_price(
            **BASE_CASE,
            steps=100,
            option_type=option_type,
            exercise_style=exercise_style,
        )
