"""Reporting helpers for the practical neural-pricing business case."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NEURAL_MODEL_IDS = {
    "notebook05_constrained_residual": "constrained_floor_residual_mlp",
    "notebook08_warm_start_integrated": "integrated_warm_start_constrained_price",
}


def _numeric(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def build_accuracy_speed_tradeoff(
    static_model_metrics: pd.DataFrame,
    runtime_curves: pd.DataFrame,
    accuracy_speed_sample: pd.DataFrame,
) -> pd.DataFrame:
    """Combine authoritative test accuracy with measured warm runtime curves."""

    rows: list[dict[str, Any]] = []
    for benchmark_id, static_id in NEURAL_MODEL_IDS.items():
        accuracy = static_model_metrics.loc[
            static_model_metrics["model_id"].astype(str).eq(static_id)
        ]
        runtime = runtime_curves.loc[
            runtime_curves["method_id"].astype(str).eq(benchmark_id)
            & runtime_curves["timing_mode"].astype(str).eq("warm")
            & runtime_curves["status"].astype(str).eq("complete")
        ]
        if accuracy.empty or runtime.empty:
            continue
        a = accuracy.iloc[0]
        r = runtime.iloc[0]
        rows.append(
            {
                "method_id": benchmark_id,
                "method": r["method"],
                "family": r["family"],
                "output_scope": r["output_scope"],
                "accuracy_source": "aligned held-out project test set",
                "price_mae": _numeric(a.get("price_mae")),
                "price_rmse": _numeric(a.get("price_rmse")),
                "maximum_absolute_error": _numeric(
                    a.get("price_max_absolute_error")
                ),
                "normalized_mae": _numeric(a.get("normalized_mae")),
                "financial_floor_violation_rate": 0.0,
                "warm_seconds_per_observation_at_scale": _numeric(
                    r.get("marginal_seconds_per_observation")
                ),
                "warm_observations_per_second_at_scale": _numeric(
                    r.get("observations_per_second_at_scale")
                ),
                "validated_domain_only": True,
                "status": "complete",
            }
        )

    numerical = accuracy_speed_sample.loc[
        accuracy_speed_sample["status"].astype(str).eq("complete")
    ]
    numerical = numerical.loc[
        numerical["family"].astype(str).eq("numerical valuation")
    ]
    for _, a in numerical.iterrows():
        runtime = runtime_curves.loc[
            runtime_curves["method_id"].astype(str).eq(str(a["method_id"]))
            & runtime_curves["timing_mode"].astype(str).eq("warm")
            & runtime_curves["status"].astype(str).eq("complete")
        ]
        if runtime.empty:
            continue
        r = runtime.iloc[0]
        rows.append(
            {
                "method_id": a["method_id"],
                "method": a["method"],
                "family": a["family"],
                "output_scope": a["output_scope"],
                "accuracy_source": (
                    "small deterministic comparison to project production CRR"
                ),
                "price_mae": _numeric(a.get("price_mae")),
                "price_rmse": _numeric(a.get("price_rmse")),
                "maximum_absolute_error": _numeric(
                    a.get("maximum_absolute_error")
                ),
                "normalized_mae": (
                    _numeric(a.get("price_mae")) / 100.0
                    if np.isfinite(_numeric(a.get("price_mae")))
                    else np.nan
                ),
                "financial_floor_violation_rate": np.nan,
                "warm_seconds_per_observation_at_scale": _numeric(
                    r.get("marginal_seconds_per_observation")
                ),
                "warm_observations_per_second_at_scale": _numeric(
                    r.get("observations_per_second_at_scale")
                ),
                "validated_domain_only": False,
                "status": "complete",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["family", "price_mae", "method"],
        kind="stable",
    ).reset_index(drop=True)


def build_business_case_recommendations(
    operational_crossover: pd.DataFrame,
    lifecycle_break_even: pd.DataFrame,
    business_case_scenarios: pd.DataFrame,
) -> pd.DataFrame:
    """Translate benchmark evidence into deployment choices."""

    warm = operational_crossover.loc[
        operational_crossover["timing_mode"].astype(str).eq("warm")
        & operational_crossover["numerical_method_id"].astype(str).eq(
            "project_numba_crr"
        )
        & operational_crossover["status"].astype(str).eq("complete")
    ]
    crossover_by_method = {
        str(row.neural_method_id): int(row.curve_crossover_observations)
        for row in warm.itertuples(index=False)
        if np.isfinite(_numeric(row.curve_crossover_observations))
    }

    lifecycle = lifecycle_break_even.loc[
        lifecycle_break_even["numerical_method_id"].astype(str).eq(
            "project_numba_crr"
        )
        & lifecycle_break_even["status"].astype(str).eq("complete")
    ]
    lifecycle_summary: dict[str, str] = {}
    for method_id, group in lifecycle.groupby("neural_method_id"):
        minimum = pd.to_numeric(
            group["break_even_valuations"], errors="coerce"
        ).min()
        maximum = pd.to_numeric(
            group["break_even_valuations"], errors="coerce"
        ).max()
        if np.isfinite(minimum) and np.isfinite(maximum):
            lifecycle_summary[str(method_id)] = (
                f"Scenario range: {int(minimum):,} to {int(maximum):,} "
                "cumulative valuations."
            )
        else:
            lifecycle_summary[str(method_id)] = "Lifecycle break-even unavailable."

    large_workloads = business_case_scenarios.loc[
        business_case_scenarios["valuations_per_run"].ge(1_000_000)
    ]
    neural_large_case = bool(
        not large_workloads.empty
        and large_workloads["neural_faster_for_workload"].astype(bool).all()
    )

    rows = [
        {
            "situation": "One-off valuation or small portfolio",
            "recommended_method": "Numerical CRR, finite difference, or QuantLib",
            "reason": (
                "No model-building cost is justified, and cold-start overhead can "
                "dominate a small job."
            ),
            "measured_threshold": None,
        },
        {
            "situation": "Repeated in-domain price-only workload",
            "recommended_method": "Notebook 05 constrained residual model",
            "reason": (
                "It is the most accurate static pricer and has the lowest neural "
                "marginal runtime for the price-only task. "
                + lifecycle_summary.get(
                    "notebook05_constrained_residual",
                    "Lifecycle break-even requires an upfront-cost scenario.",
                )
            ),
            "measured_threshold": crossover_by_method.get(
                "notebook05_constrained_residual"
            ),
        },
        {
            "situation": "Repeated in-domain price and exercise workload",
            "recommended_method": "Notebook 08 warm-start integrated model",
            "reason": (
                "It returns both outputs in one call and is faster than two separate "
                "neural specialists. "
                + lifecycle_summary.get(
                    "notebook08_warm_start_integrated",
                    "Lifecycle break-even requires an upfront-cost scenario.",
                )
            ),
            "measured_threshold": crossover_by_method.get(
                "notebook08_warm_start_integrated"
            ),
        },
        {
            "situation": "Contract outside the validated neural domain",
            "recommended_method": "High-resolution numerical method",
            "reason": (
                "All static neural models deteriorate materially outside the "
                "training range; speed does not justify unsupported extrapolation."
            ),
            "measured_threshold": None,
        },
        {
            "situation": "Large scenario grids or portfolio stress testing",
            "recommended_method": (
                "Neural surrogate inside the domain with numerical fallback"
                if neural_large_case
                else "Use the measured crossover table before deployment"
            ),
            "reason": (
                "This is the workload where a small marginal inference cost can be "
                "amortized over millions of repeated valuations."
            ),
            "measured_threshold": 1_000_000,
        },
        {
            "situation": "New payoff, exercise rule, or model assumptions",
            "recommended_method": "Numerical implementation first",
            "reason": (
                "A neural surrogate only reproduces the pricing map on which it was "
                "trained; a changed contract requires new labels and validation."
            ),
            "measured_threshold": None,
        },
    ]
    return pd.DataFrame(rows)


def build_research_question_7_summary(
    runtime_curves: pd.DataFrame,
    operational_crossover: pd.DataFrame,
    lifecycle_break_even: pd.DataFrame,
    business_case_scenarios: pd.DataFrame,
) -> dict[str, Any]:
    """Create a structured three-level answer to Research Question 7."""

    warm_crr = operational_crossover.loc[
        operational_crossover["timing_mode"].astype(str).eq("warm")
        & operational_crossover["numerical_method_id"].astype(str).eq(
            "project_numba_crr"
        )
        & operational_crossover["status"].astype(str).eq("complete")
    ]
    crossover = {
        str(row.neural_method_id): int(row.curve_crossover_observations)
        for row in warm_crr.itertuples(index=False)
        if np.isfinite(_numeric(row.curve_crossover_observations))
    }

    lifecycle_crr = lifecycle_break_even.loc[
        lifecycle_break_even["numerical_method_id"].astype(str).eq(
            "project_numba_crr"
        )
        & lifecycle_break_even["status"].astype(str).eq("complete")
    ]
    ranges: dict[str, dict[str, int]] = {}
    for method_id, group in lifecycle_crr.groupby("neural_method_id"):
        values = pd.to_numeric(group["break_even_valuations"], errors="coerce").dropna()
        if not values.empty:
            ranges[str(method_id)] = {
                "minimum": int(values.min()),
                "maximum": int(values.max()),
            }

    large = business_case_scenarios.loc[
        business_case_scenarios["valuations_per_run"].ge(1_000_000)
    ]
    return {
        "research_question": (
            "Does neural inference provide a meaningful speed advantage over "
            "numerical valuation, and when does that advantage justify deployment?"
        ),
        "marginal_speed_answer": (
            "At scale, both deployed neural surrogates have a lower measured "
            "marginal time per valuation than the project high-resolution CRR."
        ),
        "operational_crossover_by_method": crossover,
        "lifecycle_break_even_ranges": ranges,
        "large_workload_result": (
            "Neural inference is faster in the tested million-plus valuation "
            "workloads."
            if not large.empty and large["neural_faster_for_workload"].astype(bool).all()
            else "The large-workload result is mixed or incomplete."
        ),
        "business_answer": (
            "Deep learning is not justified for isolated valuations. It becomes a "
            "practical surrogate when the same in-domain pricing map is evaluated "
            "repeatedly at a volume above the measured operational crossover and "
            "the cumulative workload is large enough to recover label-generation "
            "and training cost. Numerical pricing remains the fallback outside the "
            "validated domain and whenever contract assumptions change."
        ),
    }


def render_business_case_markdown(
    rq7_summary: Mapping[str, Any],
    recommendations: pd.DataFrame,
) -> str:
    crossover = rq7_summary.get("operational_crossover_by_method", {})
    lifecycle = rq7_summary.get("lifecycle_break_even_ranges", {})

    def threshold(method_id: str) -> str:
        value = crossover.get(method_id)
        return f"{int(value):,}" if value is not None else "not resolved"

    def lifecycle_text(method_id: str) -> str:
        value = lifecycle.get(method_id)
        if not isinstance(value, Mapping):
            return "not resolved because the upfront-cost record is incomplete"
        return f"{int(value['minimum']):,} to {int(value['maximum']):,} valuations"

    return f"""## Does deep learning make business sense here?

The answer is conditional, not universal.

For a single American put or a small portfolio, the standard numerical method is the sensible choice. It is already available, does not require training data, adapts immediately when the contract changes, and remains the reference calculation.

The neural approach becomes useful when the same pricing problem must be solved repeatedly. On the measured machine, the warm operational crossover against the project CRR is approximately:

- **Notebook 05 price model:** {threshold('notebook05_constrained_residual')} valuations in one job;
- **Notebook 08 combined model:** {threshold('notebook08_warm_start_integrated')} valuations in one job.

The full lifecycle break-even also includes label generation and training. Under the recorded and scenario-based upfront-cost cases it is:

- **Notebook 05 price deployment:** {lifecycle_text('notebook05_constrained_residual')};
- **Notebook 08 combined deployment:** {lifecycle_text('notebook08_warm_start_integrated')}.

These figures do not mean that the neural model replaces CRR or QuantLib. They define where a surrogate earns a role. The numerical method remains necessary to generate labels, validate the model, handle contracts outside the learned range, and support new payoff or model assumptions.

The practical conclusion is therefore:

> Use numerical pricing for one-off, low-volume, changing, or out-of-domain work. Use a validated neural surrogate when large portfolios, parameter grids, or scenario calculations require the same in-domain pricing map to be evaluated repeatedly above the measured crossover.
"""


def create_business_case_charts(
    output_dir: Path,
    *,
    runtime_scaling: pd.DataFrame,
    operational_crossover: pd.DataFrame,
    lifecycle_break_even: pd.DataFrame,
    business_case_scenarios: pd.DataFrame,
) -> dict[str, Path]:
    """Create the four charts required for the final business-case section."""

    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    warm = runtime_scaling.loc[
        runtime_scaling["timing_mode"].astype(str).eq("warm")
        & runtime_scaling["status"].astype(str).eq("complete")
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in warm.groupby("method", sort=False):
        group = group.sort_values("requested_observations")
        ax.plot(
            group["requested_observations"],
            group["median_seconds"],
            marker="o",
            label=str(method),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Valuations in one job")
    ax.set_ylabel("Median warm runtime, seconds")
    ax.set_title("Runtime scaling by valuation method")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = output / "business_runtime_scaling.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["business_runtime_scaling"] = path

    crr = warm.loc[warm["method_id"].astype(str).eq("project_numba_crr")][
        ["requested_observations", "median_seconds"]
    ].rename(columns={"median_seconds": "crr_seconds"})
    neural = warm.loc[
        warm["method_id"].astype(str).isin(
            ["notebook05_constrained_residual", "notebook08_warm_start_integrated"]
        )
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in neural.groupby("method", sort=False):
        merged = group.merge(crr, on="requested_observations", how="inner")
        ax.plot(
            merged["requested_observations"],
            merged["crr_seconds"] / merged["median_seconds"],
            marker="o",
            label=str(method),
        )
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Valuations in one job")
    ax.set_ylabel("Speedup versus project CRR")
    ax.set_title("When neural inference becomes faster")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = output / "business_speedup_vs_crr.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["business_speedup_vs_crr"] = path

    lifecycle = lifecycle_break_even.loc[
        lifecycle_break_even["numerical_method_id"].astype(str).eq(
            "project_numba_crr"
        )
        & lifecycle_break_even["status"].astype(str).eq("complete")
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    if not lifecycle.empty:
        for method, group in lifecycle.groupby("neural_method", sort=False):
            group = group.sort_values("upfront_hours")
            ax.plot(
                group["upfront_hours"],
                group["break_even_valuations"],
                marker="o",
                label=str(method),
            )
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    ax.set_xlabel("Upfront build cost, hours")
    ax.set_ylabel("Lifecycle break-even valuations")
    ax.set_title("How upfront cost changes the neural break-even")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = output / "business_lifecycle_break_even.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["business_lifecycle_break_even"] = path

    scenario = business_case_scenarios.loc[
        business_case_scenarios["neural_method_id"].astype(str).eq(
            "notebook05_constrained_residual"
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(scenario["scenario"], scenario["annual_hours_saved"])
    ax.set_xlabel("Estimated annual computation hours saved")
    ax.set_title("Business workload impact of the price surrogate")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    path = output / "business_workload_scenarios.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["business_workload_scenarios"] = path

    return paths


__all__ = [
    "build_accuracy_speed_tradeoff",
    "build_business_case_recommendations",
    "build_research_question_7_summary",
    "create_business_case_charts",
    "render_business_case_markdown",
]
