"""Validation and benchmarking helpers for the classical pricing engines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter

import numpy as np
import pandas as pd

from .black_scholes import black_scholes_put_price
from .binomial_tree import crr_option_diagnostics, crr_option_price


REQUIRED_CONTRACT_FIELDS = {
    "spot",
    "strike",
    "time_to_maturity",
    "risk_free_rate",
    "dividend_yield",
    "volatility",
}


def _validated_contract(contract: Mapping[str, float]) -> dict[str, float]:
    """Return a plain contract dictionary after checking required fields."""

    missing = REQUIRED_CONTRACT_FIELDS.difference(contract)
    if missing:
        raise ValueError(f"Contract is missing required fields: {sorted(missing)}")
    return {name: float(contract[name]) for name in REQUIRED_CONTRACT_FIELDS}


def build_crr_convergence_table(
    contract: Mapping[str, float],
    *,
    steps_grid: Sequence[int],
    reference_steps: int = 2_000,
    timing_repeats: int = 3,
    portfolio_size: int = 100_000,
) -> pd.DataFrame:
    """Compare CRR accuracy and runtime across tree resolutions.

    European prices are compared with Black–Scholes. American prices are
    compared with a finer CRR reference tree because no closed-form benchmark is
    available under the project assumptions.
    """

    if not steps_grid:
        raise ValueError("steps_grid cannot be empty.")
    if timing_repeats <= 0:
        raise ValueError("timing_repeats must be positive.")
    if portfolio_size <= 0:
        raise ValueError("portfolio_size must be positive.")

    params = _validated_contract(contract)
    analytical_european = black_scholes_put_price(**params)
    american_reference = crr_option_price(
        **params,
        steps=int(reference_steps),
        option_type="put",
        exercise_style="american",
    )

    records: list[dict[str, float | int]] = []
    for steps in steps_grid:
        if steps <= 0:
            raise ValueError("All values in steps_grid must be positive.")

        european_tree = crr_option_price(
            **params,
            steps=int(steps),
            option_type="put",
            exercise_style="european",
        )
        american_tree = crr_option_price(
            **params,
            steps=int(steps),
            option_type="put",
            exercise_style="american",
        )

        durations = []
        for _ in range(timing_repeats):
            start = perf_counter()
            crr_option_price(
                **params,
                steps=int(steps),
                option_type="put",
                exercise_style="american",
            )
            durations.append(perf_counter() - start)

        seconds_per_option = float(np.median(durations))
        records.append(
            {
                "steps": int(steps),
                "black_scholes_european": analytical_european,
                "crr_european": european_tree,
                "european_abs_error": abs(european_tree - analytical_european),
                "crr_american": american_tree,
                "american_reference": american_reference,
                "american_reference_abs_error": abs(
                    american_tree - american_reference
                ),
                "seconds_per_option": seconds_per_option,
                "estimated_minutes_for_portfolio": (
                    seconds_per_option * portfolio_size / 60.0
                ),
            }
        )

    return pd.DataFrame.from_records(records).sort_values("steps").reset_index(drop=True)


def select_production_steps(
    convergence_table: pd.DataFrame,
    *,
    max_european_abs_error: float = 0.02,
    max_american_reference_abs_error: float = 0.02,
    minimum_steps: int = 1,
) -> int:
    """Select the smallest tree meeting accuracy and minimum-resolution rules."""

    required = {
        "steps",
        "european_abs_error",
        "american_reference_abs_error",
    }
    missing = required.difference(convergence_table.columns)
    if missing:
        raise ValueError(f"Convergence table is missing columns: {sorted(missing)}")

    if minimum_steps <= 0:
        raise ValueError("minimum_steps must be positive.")

    eligible = convergence_table.loc[
        (convergence_table["steps"] >= minimum_steps)
        & (convergence_table["european_abs_error"] <= max_european_abs_error)
        & (
            convergence_table["american_reference_abs_error"]
            <= max_american_reference_abs_error
        )
    ]
    if eligible.empty:
        raise ValueError(
            "No tested tree resolution satisfies the selected error tolerances."
        )
    return int(eligible.sort_values("steps").iloc[0]["steps"])


def build_financial_validation_grid(
    contract: Mapping[str, float],
    *,
    spot_values: Sequence[float],
    steps: int,
) -> pd.DataFrame:
    """Evaluate lower bounds and root exercise decisions across spot values."""

    params = _validated_contract(contract)
    records: list[dict[str, float | bool]] = []

    for spot in spot_values:
        row_params = {**params, "spot": float(spot)}
        european = black_scholes_put_price(**row_params)
        result = crr_option_diagnostics(
            **row_params,
            steps=steps,
            option_type="put",
            exercise_style="american",
        )
        records.append(
            {
                "spot": float(spot),
                "moneyness": float(spot) / row_params["strike"],
                "intrinsic_value": result.intrinsic_value,
                "continuation_value": result.continuation_value,
                "european_price": european,
                "american_price": result.price,
                "early_exercise_premium": result.price - european,
                "exercise_now": result.exercise_now,
                "american_ge_intrinsic": (
                    result.price + 1e-12 >= result.intrinsic_value
                ),
                "american_ge_european": result.price + 1e-12 >= european,
            }
        )

    return pd.DataFrame.from_records(records)


def summarize_dataset_financial_checks(
    dataset: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Summarize financial-bound violations in a generated dataset."""

    required = {
        "american_price",
        "european_price",
        "intrinsic_value",
        "early_exercise_premium",
        "continuation_value",
        "exercise_now",
    }
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if dataset.empty:
        raise ValueError("Dataset cannot be empty.")

    checks = {
        "American price is non-negative": dataset["american_price"] >= -tolerance,
        "American price >= intrinsic value": (
            dataset["american_price"] + tolerance >= dataset["intrinsic_value"]
        ),
        "American price >= European price": (
            dataset["american_price"] + tolerance >= dataset["european_price"]
        ),
        "Early-exercise premium is non-negative": (
            dataset["early_exercise_premium"] >= -tolerance
        ),
        "Exercise label matches root comparison": (
            dataset["exercise_now"].astype(bool)
            == (
                dataset["intrinsic_value"]
                >= dataset["continuation_value"] - 1e-12
            )
        ),
    }

    records = []
    for name, passed in checks.items():
        violations = int((~passed).sum())
        records.append(
            {
                "check": name,
                "observations": int(len(dataset)),
                "violations": violations,
                "violation_rate": violations / len(dataset),
                "passed": violations == 0,
            }
        )

    return pd.DataFrame.from_records(records)


def quantlib_american_put_price(
    contract: Mapping[str, float],
    *,
    time_steps: int = 400,
    grid_points: int = 400,
) -> float:
    """Price an American put with QuantLib finite differences.

    QuantLib is imported lazily so the rest of the project remains usable when
    the optional dependency is not installed. The maturity date is rounded to
    the nearest calendar day, making this an independent validation check rather
    than the source of the synthetic labels.
    """

    try:
        import QuantLib as ql
    except ImportError as error:
        raise ImportError(
            "QuantLib is required for the independent finite-difference check. "
            "Install it with: pip install QuantLib"
        ) from error

    if time_steps <= 0 or grid_points <= 0:
        raise ValueError("time_steps and grid_points must be positive.")

    params = _validated_contract(contract)
    evaluation_date = ql.Date(2, ql.January, 2024)
    maturity_days = max(1, int(round(params["time_to_maturity"] * 365.0)))
    maturity_date = evaluation_date + maturity_days
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()

    ql.Settings.instance().evaluationDate = evaluation_date

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(params["spot"]))
    risk_free_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(
            evaluation_date,
            params["risk_free_rate"],
            day_count,
        )
    )
    dividend_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(
            evaluation_date,
            params["dividend_yield"],
            day_count,
        )
    )
    volatility_surface = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(
            evaluation_date,
            calendar,
            params["volatility"],
            day_count,
        )
    )

    process = ql.BlackScholesMertonProcess(
        spot_handle,
        dividend_curve,
        risk_free_curve,
        volatility_surface,
    )
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, params["strike"])
    exercise = ql.AmericanExercise(evaluation_date, maturity_date)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(
        ql.FdBlackScholesVanillaEngine(process, time_steps, grid_points)
    )
    return float(option.NPV())


__all__ = [
    "build_crr_convergence_table",
    "select_production_steps",
    "build_financial_validation_grid",
    "summarize_dataset_financial_checks",
    "quantlib_american_put_price",
]
