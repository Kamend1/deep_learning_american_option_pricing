"""Black–Scholes–Merton pricing utilities for European vanilla options.

The module intentionally contains scalar pricing functions with explicit validation.
Vectorized dataset generation can be added later without changing this public API.

Notation
--------
spot
    Current underlying price S.
strike
    Option strike K.
time_to_maturity
    Remaining maturity T expressed in years.
risk_free_rate
    Continuously compounded annual risk-free rate r.
dividend_yield
    Continuously compounded annual dividend yield q.
volatility
    Annualized volatility sigma.
"""

from __future__ import annotations

import math
from typing import Final


_SQRT_TWO: Final[float] = math.sqrt(2.0)


def _standard_normal_cdf(value: float) -> float:
    """Return the standard normal cumulative distribution function."""

    return 0.5 * (1.0 + math.erf(value / _SQRT_TWO))


def _validate_inputs(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> None:
    """Validate common Black–Scholes inputs."""

    values = {
        "spot": spot,
        "strike": strike,
        "time_to_maturity": time_to_maturity,
        "risk_free_rate": risk_free_rate,
        "dividend_yield": dividend_yield,
        "volatility": volatility,
    }

    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite; received {value!r}.")

    if spot <= 0.0:
        raise ValueError(f"spot must be greater than zero; received {spot}.")
    if strike <= 0.0:
        raise ValueError(f"strike must be greater than zero; received {strike}.")
    if time_to_maturity < 0.0:
        raise ValueError(
            "time_to_maturity cannot be negative; "
            f"received {time_to_maturity}."
        )
    if volatility < 0.0:
        raise ValueError(f"volatility cannot be negative; received {volatility}.")


def _discounted_intrinsic_values(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> tuple[float, float]:
    """Return deterministic call and put values when volatility is zero."""

    discounted_spot = spot * math.exp(-dividend_yield * time_to_maturity)
    discounted_strike = strike * math.exp(
        -risk_free_rate * time_to_maturity
    )

    call_value = max(discounted_spot - discounted_strike, 0.0)
    put_value = max(discounted_strike - discounted_spot, 0.0)

    return call_value, put_value


def _d1_d2(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> tuple[float, float]:
    """Calculate the Black–Scholes d1 and d2 terms."""

    volatility_time = volatility * math.sqrt(time_to_maturity)

    d1 = (
        math.log(spot / strike)
        + (
            risk_free_rate
            - dividend_yield
            + 0.5 * volatility**2
        )
        * time_to_maturity
    ) / volatility_time

    d2 = d1 - volatility_time

    return d1, d2


def black_scholes_call_price(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European call under Black–Scholes–Merton.

    Returns the terminal intrinsic value when ``time_to_maturity == 0``.
    Uses the discounted deterministic payoff when ``volatility == 0``.
    """

    _validate_inputs(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
    )

    if time_to_maturity == 0.0:
        return max(spot - strike, 0.0)

    if volatility == 0.0:
        call_value, _ = _discounted_intrinsic_values(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        return call_value

    d1, d2 = _d1_d2(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
    )

    discounted_spot = spot * math.exp(
        -dividend_yield * time_to_maturity
    )
    discounted_strike = strike * math.exp(
        -risk_free_rate * time_to_maturity
    )

    price = (
        discounted_spot * _standard_normal_cdf(d1)
        - discounted_strike * _standard_normal_cdf(d2)
    )

    return max(float(price), 0.0)


def black_scholes_put_price(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European put under Black–Scholes–Merton.

    Returns the terminal intrinsic value when ``time_to_maturity == 0``.
    Uses the discounted deterministic payoff when ``volatility == 0``.
    """

    _validate_inputs(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
    )

    if time_to_maturity == 0.0:
        return max(strike - spot, 0.0)

    if volatility == 0.0:
        _, put_value = _discounted_intrinsic_values(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        return put_value

    d1, d2 = _d1_d2(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
    )

    discounted_spot = spot * math.exp(
        -dividend_yield * time_to_maturity
    )
    discounted_strike = strike * math.exp(
        -risk_free_rate * time_to_maturity
    )

    price = (
        discounted_strike * _standard_normal_cdf(-d2)
        - discounted_spot * _standard_normal_cdf(-d1)
    )

    return max(float(price), 0.0)


__all__ = [
    "black_scholes_call_price",
    "black_scholes_put_price",
]
