from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.business_case_break_even import (
    build_business_case_scenarios,
    build_lifecycle_break_even,
    build_operational_crossover,
    build_upfront_cost_scenarios,
    fit_runtime_curves,
)


def _runtime_scaling() -> pd.DataFrame:
    definitions = {
        "project_numba_crr": (0.002, 4.0e-5, "numerical valuation", "price and root exercise decision"),
        "notebook05_constrained_residual": (0.010, 1.0e-6, "static neural inference", "price only"),
        "notebook08_warm_start_integrated": (0.012, 1.5e-6, "static neural inference", "price and exercise decision"),
    }
    rows = []
    for method_id, (fixed, marginal, family, scope) in definitions.items():
        for timing_mode in ("cold", "warm"):
            mode_fixed = fixed * (2.0 if timing_mode == "cold" else 1.0)
            for observations in (10, 100, 1_000, 10_000, 100_000):
                seconds = mode_fixed + marginal * observations
                rows.append(
                    {
                        "method_id": method_id,
                        "method": method_id,
                        "family": family,
                        "output_scope": scope,
                        "timing_mode": timing_mode,
                        "requested_observations": observations,
                        "median_seconds": seconds,
                        "measurement_type": "measured",
                        "status": "complete",
                    }
                )
    return pd.DataFrame(rows)


def _inventory() -> pd.DataFrame:
    seconds = {
        "production_label_generation": 3_600.0,
        "notebook04_direct_training": 600.0,
        "notebook05_training": 900.0,
        "notebook06_classifier_training": 600.0,
        "notebook06_multitask_training": 900.0,
        "notebook07_training": 1_200.0,
        "notebook08_scratch_training": 1_800.0,
        "notebook08_warm_start_training": 600.0,
        "deployment_preparation": 0.0,
    }
    return pd.DataFrame(
        [
            {
                "component_id": key,
                "seconds": value,
                "status": "complete",
            }
            for key, value in seconds.items()
        ]
    )


def test_operational_and_lifecycle_break_even_are_distinct() -> None:
    scaling = _runtime_scaling()
    curves = fit_runtime_curves(scaling)
    operational = build_operational_crossover(curves, scaling)

    warm = operational.loc[
        operational["timing_mode"].eq("warm")
        & operational["numerical_method_id"].eq("project_numba_crr")
    ]
    assert set(warm["neural_method_id"]) == {
        "notebook05_constrained_residual",
        "notebook08_warm_start_integrated",
    }
    assert warm["curve_crossover_observations"].notna().all()

    upfront = build_upfront_cost_scenarios(_inventory())
    lifecycle = build_lifecycle_break_even(curves, upfront)
    lifecycle = lifecycle.loc[
        lifecycle["numerical_method_id"].eq("project_numba_crr")
        & lifecycle["status"].eq("complete")
    ]
    assert not lifecycle.empty
    assert lifecycle["break_even_valuations"].gt(0).all()
    assert lifecycle["break_even_valuations"].min() > warm[
        "curve_crossover_observations"
    ].min()

    scenarios = build_business_case_scenarios(curves, upfront)
    assert not scenarios.empty
    assert scenarios["seconds_saved_per_run"].notna().all()
    assert scenarios.loc[
        scenarios["valuations_per_run"].ge(1_000_000),
        "neural_faster_for_workload",
    ].all()


def test_missing_historical_times_generate_explicit_scenarios() -> None:
    inventory = _inventory()
    inventory.loc[
        inventory["component_id"].isin(
            ["production_label_generation", "notebook05_training"]
        ),
        "seconds",
    ] = np.nan
    inventory.loc[inventory["seconds"].isna(), "status"] = "missing"

    scenarios = build_upfront_cost_scenarios(
        inventory,
        assumed_total_build_hours=(2.0, 8.0),
    )
    complete = scenarios.loc[
        scenarios["deployment_id"].eq("notebook05_price_only")
        & scenarios["status"].eq("complete")
    ]
    assert set(complete["scenario_id"]) == {
        "assumed_total_build_2h",
        "assumed_total_build_8h",
    }
    assert set(complete["evidence_type"]) == {"total_build_scenario"}
