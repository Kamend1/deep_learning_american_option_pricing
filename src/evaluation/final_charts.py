"""Final project charts for Notebook 09."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


CHART_FILENAMES = {
    "static_pricing_mae": "static_pricing_mae.png",
    "exercise_f1": "exercise_f1.png",
    "ood_deterioration": "ood_deterioration.png",
    "runtime_comparison": "runtime_comparison.png",
    "lsm_heldout_mae": "lsm_heldout_mae.png",
}


def _save(fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_final_charts(
    output_dir: Path,
    *,
    static_model_metrics: pd.DataFrame,
    exercise_model_metrics: pd.DataFrame,
    static_ood_model_summary: pd.DataFrame,
    runtime_comparison: pd.DataFrame,
    lsm_heldout_pricing: pd.DataFrame,
) -> dict[str, Path]:
    """Generate five separate figures used in the final project conclusion."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}

    pricing = static_model_metrics.sort_values("price_mae", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(pricing["model"], pricing["price_mae"])
    ax.set_xscale("log")
    ax.set_xlabel("Mean absolute price error, log scale")
    ax.set_title("Static pricing models on the common test set")
    ax.invert_yaxis()
    result["static_pricing_mae"] = _save(
        fig, output / CHART_FILENAMES["static_pricing_mae"]
    )

    exercise = exercise_model_metrics.sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(exercise["model"], exercise["f1"])
    minimum = max(0.0, float(exercise["f1"].min()) - 0.01)
    ax.set_xlim(minimum, 1.0)
    ax.set_xlabel("F1 score")
    ax.set_title("Exercise-decision models on the common test set")
    result["exercise_f1"] = _save(
        fig, output / CHART_FILENAMES["exercise_f1"]
    )

    ood = static_ood_model_summary.loc[
        static_ood_model_summary["h6_eligible"].astype(bool)
    ].sort_values("aggregate_ood_to_in_domain_ratio", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(ood["model"], ood["aggregate_ood_to_in_domain_ratio"])
    ax.set_xscale("log")
    ax.set_xlabel("Aggregate OOD MAE / in-domain MAE, log scale")
    ax.set_title("Error deterioration outside the training range")
    result["ood_deterioration"] = _save(
        fig, output / CHART_FILENAMES["ood_deterioration"]
    )

    selected_model = static_model_metrics.sort_values(
        ["normalized_mae", "source_selected"],
        ascending=[True, False],
    ).iloc[0]["model"]
    runtime = runtime_comparison.loc[
        runtime_comparison["method"].astype(str).isin(
            [str(selected_model), "High-resolution CRR"]
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(runtime["method"], runtime["seconds_per_observation"])
    ax.set_xscale("log")
    ax.set_xlabel("Seconds per option, log scale")
    ax.set_title("Selected static neural model versus high-resolution CRR")
    result["runtime_comparison"] = _save(
        fig, output / CHART_FILENAMES["runtime_comparison"]
    )

    lsm = lsm_heldout_pricing.sort_values("mae", ascending=True)
    labels = lsm["method"].astype(str).replace(
        {
            "classical_lsm_price": "Classical Longstaff–Schwartz",
            "neural_lsm_price": "Neural Longstaff–Schwartz",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, lsm["mae"])
    ax.set_xlabel("Held-out mean absolute price error")
    ax.set_title("Path-based pricing experiment")
    result["lsm_heldout_mae"] = _save(
        fig, output / CHART_FILENAMES["lsm_heldout_mae"]
    )

    return result


__all__ = ["CHART_FILENAMES", "generate_final_charts"]
