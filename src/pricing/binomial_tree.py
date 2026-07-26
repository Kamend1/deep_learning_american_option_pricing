"""Cox–Ross–Rubinstein pricing for European and American vanilla options.

The implementation uses one-dimensional NumPy arrays during backward induction,
which keeps memory usage at O(steps). In addition to a scalar pricing function,
the module exposes root-node diagnostics needed to label the American exercise
decision in the synthetic dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np


OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


@dataclass(frozen=True, slots=True)
class CRRPriceResult:
    """Root-node diagnostics returned by the CRR pricing engine.

    Attributes
    ----------
    price:
        Present option value after applying the exercise rule.
    intrinsic_value:
        Immediate exercise payoff at the root node.
    continuation_value:
        Discounted expected value of continuing for one time step at the root.
    exercise_now:
        Whether immediate exercise is optimal at the root node. This is always
        ``False`` for European options before maturity.
    """

    price: float
    intrinsic_value: float
    continuation_value: float
    exercise_now: bool


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
    """Validate pricing parameters and normalize string arguments."""

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

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError(f"steps must be a positive integer; received {steps!r}.")

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

    return normalized_option_type, normalized_exercise_style  # type: ignore[return-value]


def _intrinsic_value(
    underlying_price: np.ndarray | float,
    *,
    strike: float,
    option_type: OptionType,
) -> np.ndarray | float:
    """Return call or put intrinsic value."""

    prices = np.asarray(underlying_price)
    if option_type == "call":
        values = np.maximum(prices - strike, 0.0)
    else:
        values = np.maximum(strike - prices, 0.0)

    if np.ndim(underlying_price) == 0:
        return float(values)
    return values


def _deterministic_result(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    steps: int,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
) -> CRRPriceResult:
    """Handle the zero-volatility case by deterministic backward induction."""

    delta_t = time_to_maturity / steps
    discount_factor = math.exp(-risk_free_rate * delta_t)
    times = np.linspace(0.0, time_to_maturity, steps + 1)
    path = spot * np.exp((risk_free_rate - dividend_yield) * times)

    option_value = float(
        _intrinsic_value(path[-1], strike=strike, option_type=option_type)
    )
    root_continuation = option_value

    for current_step in range(steps - 1, -1, -1):
        continuation = discount_factor * option_value
        exercise = float(
            _intrinsic_value(
                path[current_step],
                strike=strike,
                option_type=option_type,
            )
        )
        if current_step == 0:
            root_continuation = continuation
        option_value = (
            max(exercise, continuation)
            if exercise_style == "american"
            else continuation
        )

    root_intrinsic = float(
        _intrinsic_value(spot, strike=strike, option_type=option_type)
    )
    exercise_now = (
        exercise_style == "american"
        and root_intrinsic >= root_continuation - 1e-12
    )

    return CRRPriceResult(
        price=max(float(option_value), 0.0),
        intrinsic_value=root_intrinsic,
        continuation_value=max(float(root_continuation), 0.0),
        exercise_now=exercise_now,
    )


def crr_option_diagnostics(
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
) -> CRRPriceResult:
    """Price an option and return root-node exercise diagnostics.

    The American ``exercise_now`` label compares root intrinsic value with the
    discounted continuation value before the maximum operator is applied.
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

    root_intrinsic = float(
        _intrinsic_value(
            spot,
            strike=strike,
            option_type=normalized_option_type,
        )
    )

    if time_to_maturity == 0.0:
        return CRRPriceResult(
            price=root_intrinsic,
            intrinsic_value=root_intrinsic,
            continuation_value=0.0,
            exercise_now=(
                normalized_exercise_style == "american"
                and root_intrinsic > 0.0
            ),
        )

    if volatility == 0.0:
        return _deterministic_result(
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
    up_factor = math.exp(volatility * math.sqrt(delta_t))
    down_factor = 1.0 / up_factor
    growth_factor = math.exp((risk_free_rate - dividend_yield) * delta_t)
    probability = (growth_factor - down_factor) / (up_factor - down_factor)

    tolerance = 1e-14
    if probability < -tolerance or probability > 1.0 + tolerance:
        raise ValueError(
            "The selected parameters and step count produce an invalid CRR "
            f"risk-neutral probability p={probability:.8f}. Increase 'steps' "
            "or revise the parameter combination."
        )

    probability = min(max(probability, 0.0), 1.0)
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

    root_continuation = float("nan")

    for current_step in range(steps - 1, -1, -1):
        continuation_values = discount_factor * (
            probability * option_values[1:]
            + (1.0 - probability) * option_values[:-1]
        )

        if current_step == 0:
            root_continuation = float(continuation_values[0])

        if normalized_exercise_style == "american":
            current_up_moves = np.arange(current_step + 1, dtype=np.float64)
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
            option_values = np.maximum(continuation_values, exercise_values)
        else:
            option_values = continuation_values

    price = max(float(option_values[0]), 0.0)
    exercise_now = (
        normalized_exercise_style == "american"
        and root_intrinsic >= root_continuation - 1e-12
    )

    return CRRPriceResult(
        price=price,
        intrinsic_value=root_intrinsic,
        continuation_value=max(root_continuation, 0.0),
        exercise_now=exercise_now,
    )


def crr_option_price(**kwargs: object) -> float:
    """Return only the scalar CRR option price.

    This wrapper preserves the simple public API used by Notebook 01. Use
    :func:`crr_option_diagnostics` when continuation value or the exercise label
    is required.
    """

    return crr_option_diagnostics(**kwargs).price  # type: ignore[arg-type]


__all__ = [
    "CRRPriceResult",
    "crr_option_diagnostics",
    "crr_option_price",
]
