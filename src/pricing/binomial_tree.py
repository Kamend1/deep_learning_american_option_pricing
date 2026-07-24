"""Cox–Ross–Rubinstein lattice pricing for European and American options.

The implementation uses one-dimensional NumPy arrays during backward induction,
which keeps memory usage at O(steps) rather than constructing a full tree.

The tree supports:
- European calls and puts;
- American calls and puts;
- continuous dividend yield;
- zero maturity;
- zero volatility through a deterministic discrete-exercise calculation.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np


OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


def _validate_inputs(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    steps: int,
    option_type: str,
    exercise_style: str,
) -> tuple[OptionType, ExerciseStyle]:
    """Validate inputs and return normalized string parameters."""

    numeric_values = {
        "spot": spot,
        "strike": strike,
        "time_to_maturity": time_to_maturity,
        "risk_free_rate": risk_free_rate,
        "dividend_yield": dividend_yield,
        "volatility": volatility,
    }

    for name, value in numeric_values.items():
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

    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ValueError(f"steps must be a positive integer; received {steps!r}.")
    if steps <= 0:
        raise ValueError(f"steps must be greater than zero; received {steps}.")

    normalized_option_type = option_type.strip().lower()
    normalized_exercise_style = exercise_style.strip().lower()

    if normalized_option_type not in {"call", "put"}:
        raise ValueError(
            "option_type must be either 'call' or 'put'; "
            f"received {option_type!r}."
        )
    if normalized_exercise_style not in {"european", "american"}:
        raise ValueError(
            "exercise_style must be either 'european' or 'american'; "
            f"received {exercise_style!r}."
        )

    return (
        normalized_option_type,  # type: ignore[return-value]
        normalized_exercise_style,  # type: ignore[return-value]
    )


def _intrinsic_value(
    underlying_price: np.ndarray | float,
    *,
    strike: float,
    option_type: OptionType,
) -> np.ndarray | float:
    """Return call or put intrinsic value."""

    if option_type == "call":
        return np.maximum(np.asarray(underlying_price) - strike, 0.0)

    return np.maximum(strike - np.asarray(underlying_price), 0.0)


def _deterministic_option_price(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    steps: int,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
) -> float:
    """Price the zero-volatility case on the discrete exercise grid."""

    times = np.linspace(0.0, time_to_maturity, steps + 1)
    underlying_path = spot * np.exp(
        (risk_free_rate - dividend_yield) * times
    )
    intrinsic_values = _intrinsic_value(
        underlying_path,
        strike=strike,
        option_type=option_type,
    )
    discounted_values = np.exp(-risk_free_rate * times) * intrinsic_values

    if exercise_style == "european":
        return float(discounted_values[-1])

    return float(np.max(discounted_values))


def crr_option_price(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    steps: int,
    option_type: str,
    exercise_style: str,
    dividend_yield: float = 0.0,
) -> float:
    """Price a vanilla option with a Cox–Ross–Rubinstein tree.

    Parameters
    ----------
    spot:
        Current underlying price.
    strike:
        Option strike.
    time_to_maturity:
        Remaining maturity in years.
    risk_free_rate:
        Continuously compounded annual risk-free rate.
    volatility:
        Annualized volatility.
    steps:
        Number of CRR time steps.
    option_type:
        ``"call"`` or ``"put"``.
    exercise_style:
        ``"european"`` or ``"american"``.
    dividend_yield:
        Continuously compounded annual dividend yield.

    Returns
    -------
    float
        Present option value.

    Raises
    ------
    ValueError
        If parameters are invalid or the requested step count produces an
        invalid risk-neutral probability.
    """

    normalized_option_type, normalized_exercise_style = _validate_inputs(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        steps=steps,
        option_type=option_type,
        exercise_style=exercise_style,
    )

    if time_to_maturity == 0.0:
        intrinsic = _intrinsic_value(
            spot,
            strike=strike,
            option_type=normalized_option_type,
        )
        return float(intrinsic)

    if volatility == 0.0:
        return _deterministic_option_price(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            steps=steps,
            option_type=normalized_option_type,
            exercise_style=normalized_exercise_style,
        )

    delta_t = time_to_maturity / steps
    sqrt_delta_t = math.sqrt(delta_t)

    up_factor = math.exp(volatility * sqrt_delta_t)
    down_factor = 1.0 / up_factor

    growth_factor = math.exp(
        (risk_free_rate - dividend_yield) * delta_t
    )
    denominator = up_factor - down_factor
    risk_neutral_probability = (
        growth_factor - down_factor
    ) / denominator

    tolerance = 1e-14
    if (
        risk_neutral_probability < -tolerance
        or risk_neutral_probability > 1.0 + tolerance
    ):
        raise ValueError(
            "The selected parameters and step count produce an invalid "
            "CRR risk-neutral probability "
            f"p={risk_neutral_probability:.8f}. Increase 'steps' or revise "
            "the parameter combination."
        )

    risk_neutral_probability = min(
        max(risk_neutral_probability, 0.0),
        1.0,
    )
    discount_factor = math.exp(-risk_free_rate * delta_t)

    up_moves = np.arange(steps + 1, dtype=np.float64)
    down_moves = steps - up_moves

    terminal_underlying = (
        spot
        * np.power(up_factor, up_moves)
        * np.power(down_factor, down_moves)
    )

    option_values = np.asarray(
        _intrinsic_value(
            terminal_underlying,
            strike=strike,
            option_type=normalized_option_type,
        ),
        dtype=np.float64,
    )

    for current_step in range(steps - 1, -1, -1):
        option_values = discount_factor * (
            risk_neutral_probability * option_values[1:]
            + (1.0 - risk_neutral_probability) * option_values[:-1]
        )

        if normalized_exercise_style == "american":
            current_up_moves = np.arange(
                current_step + 1,
                dtype=np.float64,
            )
            current_down_moves = current_step - current_up_moves

            current_underlying = (
                spot
                * np.power(up_factor, current_up_moves)
                * np.power(down_factor, current_down_moves)
            )

            exercise_values = np.asarray(
                _intrinsic_value(
                    current_underlying,
                    strike=strike,
                    option_type=normalized_option_type,
                ),
                dtype=np.float64,
            )

            option_values = np.maximum(option_values, exercise_values)

    return max(float(option_values[0]), 0.0)


__all__ = ["crr_option_price"]
