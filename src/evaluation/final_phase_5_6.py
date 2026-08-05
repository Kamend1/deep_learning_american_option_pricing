"""Orchestration for Notebook 09 Phases 5 and 6."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.evaluation.final_artifact_adapters import FinalNotebookPackage
from src.evaluation.final_cross_family_evaluation import (
    assert_cross_family_evidence_ready,
    run_phase_5_cross_family_evaluation,
)
from src.evaluation.final_exercise_comparison import (
    assert_exercise_evidence_ready,
    run_phase_5_exercise_comparison,
)
from src.evaluation.final_hypothesis_evidence import (
    assert_phase_6_ready,
    run_phase_6_hypothesis_decisions,
)


def run_phases_5_6(
    packages: Mapping[str, FinalNotebookPackage],
    *,
    static_model_metrics: pd.DataFrame,
    static_financial_consistency: pd.DataFrame,
) -> dict[str, Any]:
    """Build all remaining evidence families and deterministic H1-H6 decisions."""

    exercise = run_phase_5_exercise_comparison(packages)
    cross_family = run_phase_5_cross_family_evaluation(
        packages,
        static_model_metrics,
    )
    hypotheses = run_phase_6_hypothesis_decisions(
        packages,
        static_model_metrics=static_model_metrics,
        static_financial_consistency=static_financial_consistency,
        exercise_model_metrics=exercise["exercise_model_metrics"],
        static_ood_model_summary=cross_family["static_ood_model_summary"],
        runtime_comparison=cross_family["runtime_comparison"],
    )
    return {**exercise, **cross_family, **hypotheses}


def assert_phases_5_6_ready(results: Mapping[str, Any]) -> None:
    """Apply the exercise, cross-family, and hypothesis strict gates."""

    assert_exercise_evidence_ready(results)
    assert_cross_family_evidence_ready(results)
    assert_phase_6_ready(results)


__all__ = ["assert_phases_5_6_ready", "run_phases_5_6"]
