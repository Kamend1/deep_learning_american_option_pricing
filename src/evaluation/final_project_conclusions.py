"""Task-specific conclusions for the final American-option pricing evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


MODEL_IDS = {
    "selected_price": "constrained_floor_residual_mlp",
    "specialist_exercise": "exercise_only_classifier",
    "integrated_price": "integrated_warm_start_constrained_price",
    "integrated_exercise": "integrated_warm_start_exercise_head",
    "integrated_continuation": "integrated_warm_start_continuation_path",
    "integrated_scratch_price": "integrated_scratch_constrained_price",
}


def _numeric(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _row(
    frame: pd.DataFrame,
    *,
    model_id: str | None = None,
    model: str | None = None,
) -> pd.Series:
    if frame.empty:
        raise ValueError("Required conclusion table is empty.")
    match = frame
    if model_id is not None and "model_id" in frame.columns:
        match = frame.loc[frame["model_id"].astype(str).eq(model_id)]
    elif model is not None and "model" in frame.columns:
        match = frame.loc[frame["model"].astype(str).eq(model)]
    else:
        raise KeyError("No usable model identifier was supplied.")
    if match.empty:
        identifier = model_id if model_id is not None else model
        raise KeyError(f"Required model row is missing: {identifier}")
    return match.iloc[0]


def _preferred_tie_break(
    frame: pd.DataFrame,
    metric: str,
    *,
    higher_is_better: bool = False,
) -> pd.Series:
    if frame.empty or metric not in frame.columns:
        raise ValueError(f"Cannot select a model from missing metric {metric!r}.")
    work = frame.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.loc[work[metric].notna()].copy()
    if work.empty:
        raise ValueError(f"No finite values are available for {metric!r}.")
    if "source_selected" not in work.columns:
        work["source_selected"] = False
    if "source_notebook" not in work.columns:
        work["source_notebook"] = 99
    return work.sort_values(
        [metric, "source_selected", "source_notebook"],
        ascending=[not higher_is_better, False, True],
    ).iloc[0]


def build_task_recommendations(
    static_model_metrics: pd.DataFrame,
    exercise_model_metrics: pd.DataFrame,
    static_ood_model_summary: pd.DataFrame,
    lsm_heldout_pricing: pd.DataFrame,
    runtime_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Return one recommendation for each genuinely different user task."""

    price = _preferred_tie_break(static_model_metrics, "normalized_mae")
    exercise = _preferred_tie_break(
        exercise_model_metrics,
        "f1",
        higher_is_better=True,
    )
    specialist_exercise = _row(
        exercise_model_metrics,
        model_id=MODEL_IDS["specialist_exercise"],
    )

    ood = static_ood_model_summary.loc[
        static_ood_model_summary.get("h6_eligible", False).astype(bool)
    ].copy()
    if "source_selected" not in ood.columns:
        selected_ids = set(
            static_model_metrics.loc[
                static_model_metrics.get("source_selected", False).astype(bool),
                "model_id",
            ].astype(str)
        ) if "model_id" in static_model_metrics.columns else set()
        if "model_id" in ood.columns:
            ood["source_selected"] = ood["model_id"].astype(str).isin(selected_ids)
        else:
            ood["source_selected"] = False
    ood_winner = _preferred_tie_break(ood, "aggregate_ood_normalized_mae")

    if lsm_heldout_pricing.empty or "mae" not in lsm_heldout_pricing.columns:
        raise ValueError("Longstaff–Schwartz held-out pricing evidence is unavailable.")
    lsm = lsm_heldout_pricing.copy()
    lsm["mae"] = pd.to_numeric(lsm["mae"], errors="coerce")
    lsm_winner = lsm.sort_values("mae").iloc[0]

    integrated_price = _row(
        static_model_metrics,
        model_id=MODEL_IDS["integrated_price"],
    )
    integrated_exercise = _row(
        exercise_model_metrics,
        model_id=MODEL_IDS["integrated_exercise"],
    )

    static_runtime = runtime_comparison.loc[
        runtime_comparison["benchmark_family"].astype(str).eq(
            "static neural inference"
        )
    ]
    selected_runtime = static_runtime.loc[
        static_runtime["method"].astype(str).eq(str(price["model"]))
    ]
    crr_runtime = runtime_comparison.loc[
        runtime_comparison["method"].astype(str).eq("High-resolution CRR")
    ]
    speed_text = "Runtime evidence unavailable."
    speed_value = float("nan")
    if not selected_runtime.empty and not crr_runtime.empty:
        static_seconds = _numeric(selected_runtime.iloc[0]["seconds_per_observation"])
        crr_seconds = _numeric(crr_runtime.iloc[0]["seconds_per_observation"])
        if static_seconds > 0.0 and crr_seconds > 0.0:
            speed_value = crr_seconds / static_seconds
            speed_text = (
                f"About {speed_value:,.0f} times faster than the high-resolution "
                "tree in the recorded large-batch CPU benchmark."
            )

    rows = [
        {
            "task": "Most accurate static price",
            "recommended_model": price["model"],
            "source_notebook": str(price["source_notebook"]).zfill(2),
            "primary_metric": "price_mae",
            "primary_value": _numeric(price.get("price_mae")),
            "reason": (
                "Lowest error on the aligned common test set and zero violations "
                "of the European, intrinsic, and combined lower bounds."
            ),
            "tradeoff": "Produces a price only; it does not provide an exercise recommendation.",
        },
        {
            "task": "Most accurate exercise decision",
            "recommended_model": exercise["model"],
            "source_notebook": str(exercise["source_notebook"]).zfill(2),
            "primary_metric": "f1",
            "primary_value": _numeric(exercise.get("f1")),
            "reason": "Best exercise-decision F1 score on the aligned common test set.",
            "tradeoff": (
                "This is the integrated warm-start model, so it carries more "
                "computation than an exercise-only classifier but also returns a price."
                if str(exercise.get("model_id")) == MODEL_IDS["integrated_exercise"]
                else "This is an exercise-only model and does not return an option price."
            ),
        },
        {
            "task": "Exercise-only deployment",
            "recommended_model": specialist_exercise["model"],
            "source_notebook": str(specialist_exercise["source_notebook"]).zfill(2),
            "primary_metric": "f1",
            "primary_value": _numeric(specialist_exercise.get("f1")),
            "reason": (
                "Use the dedicated Notebook 06 classifier when only the exercise "
                "decision is required and the additional price output has no value."
            ),
            "tradeoff": (
                "Its F1 may be marginally below the integrated warm-start model, "
                "but it has the narrower and cheaper deployment scope."
            ),
        },
        {
            "task": "One model for both price and exercise",
            "recommended_model": "Integrated warm-start deployment model",
            "source_notebook": "08",
            "primary_metric": "price_mae_and_exercise_f1",
            "primary_value": _numeric(integrated_exercise.get("f1")),
            "reason": (
                f"Returns a constrained price with MAE {_numeric(integrated_price.get('price_mae')):.6f} "
                f"and an exercise F1 score of {_numeric(integrated_exercise.get('f1')):.6f} "
                "from one shared model."
            ),
            "tradeoff": (
                "Its price is less accurate and its inference is slower than the best "
                "specialist pricing model."
            ),
        },
        {
            "task": "Path-based valuation",
            "recommended_model": str(lsm_winner["method"]),
            "source_notebook": str(lsm_winner.get("source_notebook", "07")).zfill(2),
            "primary_metric": "heldout_price_mae",
            "primary_value": _numeric(lsm_winner.get("mae")),
            "reason": "Lowest held-out contract-level pricing error in the path-based experiment.",
            "tradeoff": "Path simulation remains much slower than static neural inference.",
        },
        {
            "task": "Lowest absolute error outside the training range",
            "recommended_model": ood_winner["model"],
            "source_notebook": str(ood_winner["source_notebook"]).zfill(2),
            "primary_metric": "aggregate_ood_normalized_mae",
            "primary_value": _numeric(ood_winner.get("aggregate_ood_normalized_mae")),
            "reason": "Lowest average absolute pricing error across the four predefined difficult regimes.",
            "tradeoff": (
                "The error still rises materially relative to its very low in-domain error; "
                "no tested model is reliable outside the tested range without qualification."
            ),
        },
        {
            "task": "Repeated large-batch pricing",
            "recommended_model": price["model"],
            "source_notebook": str(price["source_notebook"]).zfill(2),
            "primary_metric": "speedup_vs_high_resolution_crr",
            "primary_value": speed_value,
            "reason": speed_text,
            "tradeoff": "The measured speedup is hardware- and batch-size-specific and excludes training.",
        },
    ]
    return pd.DataFrame(rows)


def build_integrated_model_tradeoff(
    static_model_metrics: pd.DataFrame,
    static_financial_consistency: pd.DataFrame,
    exercise_model_metrics: pd.DataFrame,
    runtime_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Compare Notebook 08 with the best specialist models on like-for-like measures."""

    specialist_price = _row(
        static_model_metrics,
        model_id=MODEL_IDS["selected_price"],
    )
    integrated_price = _row(
        static_model_metrics,
        model_id=MODEL_IDS["integrated_price"],
    )
    specialist_exercise = _row(
        exercise_model_metrics,
        model_id=MODEL_IDS["specialist_exercise"],
    )
    integrated_exercise = _row(
        exercise_model_metrics,
        model_id=MODEL_IDS["integrated_exercise"],
    )
    continuation = _row(
        exercise_model_metrics,
        model_id=MODEL_IDS["integrated_continuation"],
    )

    def violation(model_id: str) -> float:
        row = _row(static_financial_consistency, model_id=model_id)
        return _numeric(row.get("below_financial_floor_rate"))

    static_runtime = runtime_comparison.loc[
        runtime_comparison["benchmark_family"].astype(str).eq(
            "static neural inference"
        )
    ]
    specialist_runtime = static_runtime.loc[
        static_runtime["method"].astype(str).eq(str(specialist_price["model"]))
    ]
    integrated_runtime = static_runtime.loc[
        static_runtime["method"].astype(str).eq("Integrated warm-start deployment model")
    ]
    specialist_throughput = (
        _numeric(specialist_runtime.iloc[0].get("observations_per_second"))
        if not specialist_runtime.empty
        else float("nan")
    )
    integrated_throughput = (
        _numeric(integrated_runtime.iloc[0].get("observations_per_second"))
        if not integrated_runtime.empty
        else float("nan")
    )

    price_ratio = _numeric(integrated_price["price_mae"]) / _numeric(
        specialist_price["price_mae"]
    )
    throughput_ratio = (
        integrated_throughput / specialist_throughput
        if specialist_throughput > 0.0
        else float("nan")
    )

    return pd.DataFrame(
        [
            {
                "dimension": "Pricing error",
                "specialist_model": specialist_price["model"],
                "specialist_value": _numeric(specialist_price["price_mae"]),
                "integrated_value": _numeric(integrated_price["price_mae"]),
                "comparison": f"Integrated error is {price_ratio:.2f} times larger.",
                "interpretation": "The combined model does not replace the specialist price model.",
            },
            {
                "dimension": "Exercise F1",
                "specialist_model": specialist_exercise["model"],
                "specialist_value": _numeric(specialist_exercise["f1"]),
                "integrated_value": _numeric(integrated_exercise["f1"]),
                "comparison": (
                    "Integrated minus specialist = "
                    f"{_numeric(integrated_exercise['f1'])-_numeric(specialist_exercise['f1']):.6f}."
                ),
                "interpretation": "The direct integrated decision is extremely close to the specialist.",
            },
            {
                "dimension": "Financial lower-bound violations",
                "specialist_model": specialist_price["model"],
                "specialist_value": violation(MODEL_IDS["selected_price"]),
                "integrated_value": violation(MODEL_IDS["integrated_price"]),
                "comparison": "Both authoritative constrained prices have zero violations.",
                "interpretation": "The integrated price keeps the same structural protection.",
            },
            {
                "dimension": "Inference throughput",
                "specialist_model": specialist_price["model"],
                "specialist_value": specialist_throughput,
                "integrated_value": integrated_throughput,
                "comparison": f"Integrated throughput is {throughput_ratio:.2%} of the specialist.",
                "interpretation": "Several outputs from one network cost additional inference time.",
            },
            {
                "dimension": "Continuation-implied decision F1",
                "specialist_model": specialist_exercise["model"],
                "specialist_value": _numeric(specialist_exercise["f1"]),
                "integrated_value": _numeric(continuation["f1"]),
                "comparison": (
                    "Continuation-implied minus specialist = "
                    f"{_numeric(continuation['f1'])-_numeric(specialist_exercise['f1']):.6f}."
                ),
                "interpretation": (
                    "The value-of-waiting output is informative but is not as reliable as the "
                    "direct exercise head."
                ),
            },
        ]
    )


def build_project_findings(
    static_model_metrics: pd.DataFrame,
    static_financial_consistency: pd.DataFrame,
    exercise_model_metrics: pd.DataFrame,
    static_ood_model_summary: pd.DataFrame,
    lsm_heldout_pricing: pd.DataFrame,
    lsm_coverage: pd.DataFrame,
    runtime_comparison: pd.DataFrame,
    hypothesis_decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Create the short evidence-led findings used in the final write-up."""

    price = _row(static_model_metrics, model_id=MODEL_IDS["selected_price"])
    direct = _row(static_model_metrics, model_id="direct_mlp")
    black_scholes = _row(static_model_metrics, model_id="black_scholes_proxy")
    integrated_price = _row(
        static_model_metrics,
        model_id=MODEL_IDS["integrated_price"],
    )
    exercise = _row(
        exercise_model_metrics,
        model_id=MODEL_IDS["specialist_exercise"],
    )
    integrated_exercise = _row(
        exercise_model_metrics,
        model_id=MODEL_IDS["integrated_exercise"],
    )
    direct_consistency = _row(static_financial_consistency, model_id="direct_mlp")

    eligible_ood = static_ood_model_summary.loc[
        static_ood_model_summary["h6_eligible"].astype(bool)
    ]
    minimum_ood_ratio = pd.to_numeric(
        eligible_ood["aggregate_ood_to_in_domain_ratio"], errors="coerce"
    ).min()

    lsm = lsm_heldout_pricing.set_index("method")
    classical_mae = _numeric(lsm.loc["classical_lsm_price", "mae"])
    neural_mae = _numeric(lsm.loc["neural_lsm_price", "mae"])
    coverage_map = {
        str(row["metric"]): _numeric(row["coverage"])
        for _, row in lsm_coverage.iterrows()
    }

    selected_runtime = runtime_comparison.loc[
        runtime_comparison["method"].astype(str).eq(str(price["model"]))
    ]
    crr_runtime = runtime_comparison.loc[
        runtime_comparison["method"].astype(str).eq("High-resolution CRR")
    ]
    speedup = float("nan")
    if not selected_runtime.empty and not crr_runtime.empty:
        selected_seconds = _numeric(selected_runtime.iloc[0]["seconds_per_observation"])
        crr_seconds = _numeric(crr_runtime.iloc[0]["seconds_per_observation"])
        if selected_seconds > 0.0:
            speedup = crr_seconds / selected_seconds

    decision_map = dict(
        zip(
            hypothesis_decisions["hypothesis"].astype(str),
            hypothesis_decisions["decision"].astype(str),
        )
    )

    rows = [
        {
            "topic": "Direct approximation",
            "finding": (
                f"The direct neural model reduces normalized MAE from "
                f"{_numeric(black_scholes['normalized_mae']):.6f} to "
                f"{_numeric(direct['normalized_mae']):.6f}."
            ),
            "meaning": "A neural surrogate can learn the American early-exercise component that the European proxy omits.",
        },
        {
            "topic": "Best static price",
            "finding": (
                f"The constrained residual model has price MAE "
                f"{_numeric(price['price_mae']):.6f}, the lowest on the common test set."
            ),
            "meaning": "Learning only the value above a known lower bound is better than learning the full price directly.",
        },
        {
            "topic": "Financial consistency",
            "finding": (
                f"The selected constrained price has zero lower-bound violations; "
                f"the direct model violates the combined floor in "
                f"{_numeric(direct_consistency['below_financial_floor_rate']):.2%} of cases."
            ),
            "meaning": "Embedding a minimum valid price in the output construction solves a real weakness rather than merely improving an average score.",
        },
        {
            "topic": "Exercise decision",
            "finding": f"The specialist classifier reaches F1={_numeric(exercise['f1']):.6f}.",
            "meaning": "A dedicated decision model remains the best option when the exercise recommendation is the only required output.",
        },
        {
            "topic": "Combined model",
            "finding": (
                f"The integrated model reaches price MAE {_numeric(integrated_price['price_mae']):.6f} "
                f"and exercise F1 {_numeric(integrated_exercise['f1']):.6f}."
            ),
            "meaning": "It is a strong one-model compromise, but it does not beat the best specialist on either task.",
        },
        {
            "topic": "Outside the training range",
            "finding": (
                f"Every eligible static model has aggregate out-of-domain error at least "
                f"{minimum_ood_ratio:.2f} times its in-domain error."
            ),
            "meaning": "Excellent interpolation does not imply dependable extrapolation.",
        },
        {
            "topic": "Path-based experiment",
            "finding": (
                f"Classical Longstaff–Schwartz MAE is {classical_mae:.6f}, versus "
                f"{neural_mae:.6f} for the neural policy; 95% interval coverage is "
                f"{coverage_map.get('Classical LSM 95% CI coverage', float('nan')):.0%} versus "
                f"{coverage_map.get('Neural LSM 95% CI coverage', float('nan')):.0%}."
            ),
            "meaning": "The neural continuation policy did not justify its extra complexity in this experiment.",
        },
        {
            "topic": "Speed",
            "finding": f"The selected static model is about {speedup:,.0f} times faster than high-resolution CRR in the recorded batch benchmark.",
            "meaning": "The practical value of the surrogate appears when many already-specified contracts must be priced repeatedly.",
        },
        {
            "topic": "Hypotheses",
            "finding": "; ".join(f"{key}: {value}" for key, value in sorted(decision_map.items())),
            "meaning": "The evidence supports approximation, residual learning, constraints, speed, and OOD deterioration, but not superiority from multi-task learning.",
        },
    ]
    return pd.DataFrame(rows)


def build_project_limitations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "limitation": "Synthetic reference prices",
                "effect": "The models reproduce a high-resolution CRR pricing rule; they are not validated against traded market prices.",
            },
            {
                "limitation": "Fixed experimental domain",
                "effect": "The results apply to the selected American put parameter ranges, fixed strike normalization, and predefined test design.",
            },
            {
                "limitation": "Separate path-based experiment",
                "effect": "Longstaff–Schwartz results use a different contract sample and cannot be placed in the same leaderboard as the static models.",
            },
            {
                "limitation": "Boundary labels",
                "effect": "Exercise labels and distance to the boundary inherit the numerical resolution and assumptions of the CRR reference model.",
            },
            {
                "limitation": "Out-of-domain scope",
                "effect": "Only four predefined difficult regimes are tested; other extrapolation failures may remain unseen.",
            },
            {
                "limitation": "Runtime context",
                "effect": "Speed results depend on hardware, batch size, implementation, and exclude the cost of generating labels and training the static models.",
            },
            {
                "limitation": "Partial financial structure",
                "effect": "The constrained outputs guarantee lower bounds, but they do not guarantee every monotonicity or no-arbitrage relationship.",
            },
        ]
    )


def build_final_results_summary(
    task_recommendations: pd.DataFrame,
    project_findings: pd.DataFrame,
    hypothesis_decisions: pd.DataFrame,
) -> dict[str, Any]:
    recommendations = {
        str(row["task"]): {
            "model": str(row["recommended_model"]),
            "source_notebook": str(row["source_notebook"]),
            "primary_metric": str(row["primary_metric"]),
            "primary_value": _numeric(row["primary_value"]),
        }
        for _, row in task_recommendations.iterrows()
    }
    decisions = {
        str(row["hypothesis"]): str(row["decision"])
        for _, row in hypothesis_decisions.iterrows()
    }
    return {
        "status": "complete",
        "universal_preferred_model": None,
        "overall_answer": (
            "Financially structured neural surrogates can price American puts very "
            "accurately and much faster than repeated high-resolution tree valuation, "
            "but the preferred model depends on whether the user needs a price, an "
            "exercise decision, both outputs together, or a path-based valuation."
        ),
        "task_recommendations": recommendations,
        "hypothesis_decisions": decisions,
        "finding_count": int(len(project_findings)),
    }


def render_final_conclusion_markdown(
    task_recommendations: pd.DataFrame,
    integrated_tradeoff: pd.DataFrame,
    project_findings: pd.DataFrame,
    project_limitations: pd.DataFrame,
    hypothesis_decisions: pd.DataFrame,
) -> str:
    """Render the final project conclusion in direct, non-technical language."""

    rec = task_recommendations.set_index("task")
    price_model = rec.loc["Most accurate static price", "recommended_model"]
    exercise_model = rec.loc["Most accurate exercise decision", "recommended_model"]
    lsm_model = rec.loc["Path-based valuation", "recommended_model"]

    pricing_tradeoff = integrated_tradeoff.loc[
        integrated_tradeoff["dimension"].eq("Pricing error")
    ].iloc[0]
    exercise_tradeoff = integrated_tradeoff.loc[
        integrated_tradeoff["dimension"].eq("Exercise F1")
    ].iloc[0]

    decision_lines = "\n".join(
        f"- **{row['hypothesis']}: {row['decision']}** — {row['primary_evidence']}"
        for _, row in hypothesis_decisions.iterrows()
    )
    limitation_lines = "\n".join(
        f"- **{row['limitation']}:** {row['effect']}"
        for _, row in project_limitations.iterrows()
    )

    return f"""# Final project conclusion

## The main answer

The project does not produce one model that is best at everything. It produces a clear division of work.

- **{price_model}** is the preferred model when the goal is the most accurate static price.
- **{exercise_model}** is the preferred model when the goal is only the exercise decision.
- **The warm-start integrated deployment model from Notebook 08** is preferred when one model must return both a protected price and an exercise recommendation.
- **{lsm_model}** is preferred in the separate path-based experiment.

The warm-start integrated model is useful for in-domain combined deployment, but it is not the overall winner. {pricing_tradeoff['comparison']} {exercise_tradeoff['comparison']} Its value is that it combines several related answers while keeping the authoritative price above the required lower bounds.

## What the experiments establish

The direct neural model already improves substantially on using the European price as a proxy for an American option. The larger improvement comes from changing the problem: instead of asking the network to relearn the whole price, the best model learns only the amount above a known minimum valid value. That model is both more accurate and free of lower-bound violations on the common test set.

The exercise experiments show that joint learning is possible without a large loss of classification quality, but they do not show that joint learning is better. The specialist classifier remains first. The integrated exercise head is extremely close and is therefore useful when price and decision must come from one model. The decision inferred from the estimated value of waiting is weaker and should be treated as supporting explanation rather than the primary recommendation.

The path-based experiment gives a different result. Classical Longstaff–Schwartz is more accurate, has better interval coverage, and is faster than the neural policy in this implementation. The neural version therefore does not earn a preferred role in the final project.

All eligible static neural models deteriorate materially when tested outside the range used for training. The exact size varies by model and regime, and one isolated regime can look stable, but the aggregate conclusion is unchanged: strong performance inside the training range is not evidence of safe extrapolation.

The static neural models are dramatically faster than repeated high-resolution tree valuation after training. This is the strongest practical case for the surrogate approach: repeated valuation of large batches, not elimination of the numerical model that produced the reference prices.

## Hypothesis decisions

{decision_lines}

## Final answer to the research question

A financially structured neural network can serve as a highly accurate and computationally efficient surrogate for American put pricing inside a defined domain. The best result comes from combining known option structure with residual learning, not from using the largest or most complicated network. Separate specialist models remain preferable when only one output is needed. A combined model is justified when price and exercise information must be produced together. The neural path-based policy is not preferred, and none of the models should be presented as dependable outside the tested domain without further validation.

## Limitations

{limitation_lines}
"""


def run_phase_7_conclusions(
    *,
    static_model_metrics: pd.DataFrame,
    static_financial_consistency: pd.DataFrame,
    exercise_model_metrics: pd.DataFrame,
    static_ood_model_summary: pd.DataFrame,
    lsm_heldout_pricing: pd.DataFrame,
    lsm_coverage: pd.DataFrame,
    runtime_comparison: pd.DataFrame,
    hypothesis_decisions: pd.DataFrame,
) -> dict[str, Any]:
    recommendations = build_task_recommendations(
        static_model_metrics,
        exercise_model_metrics,
        static_ood_model_summary,
        lsm_heldout_pricing,
        runtime_comparison,
    )
    tradeoff = build_integrated_model_tradeoff(
        static_model_metrics,
        static_financial_consistency,
        exercise_model_metrics,
        runtime_comparison,
    )
    findings = build_project_findings(
        static_model_metrics,
        static_financial_consistency,
        exercise_model_metrics,
        static_ood_model_summary,
        lsm_heldout_pricing,
        lsm_coverage,
        runtime_comparison,
        hypothesis_decisions,
    )
    limitations = build_project_limitations()
    summary = build_final_results_summary(
        recommendations,
        findings,
        hypothesis_decisions,
    )
    narrative = render_final_conclusion_markdown(
        recommendations,
        tradeoff,
        findings,
        limitations,
        hypothesis_decisions,
    )
    return {
        "task_recommendations": recommendations,
        "integrated_model_tradeoff": tradeoff,
        "project_findings": findings,
        "project_limitations": limitations,
        "final_results_summary": summary,
        "final_conclusion_markdown": narrative,
    }


__all__ = [
    "build_final_results_summary",
    "build_integrated_model_tradeoff",
    "build_project_findings",
    "build_project_limitations",
    "build_task_recommendations",
    "render_final_conclusion_markdown",
    "run_phase_7_conclusions",
]
