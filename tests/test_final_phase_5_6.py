from __future__ import annotations

from src.evaluation.final_phase_5_6 import (
    assert_phases_5_6_ready,
    run_phases_5_6,
)

from phase5_6_fixtures import build_phase5_6_packages, phase4_tables


def test_combined_phases_5_6_gate(tmp_path):
    packages = build_phase5_6_packages(tmp_path)
    static_metrics, consistency = phase4_tables()
    results = run_phases_5_6(
        packages,
        static_model_metrics=static_metrics,
        static_financial_consistency=consistency,
    )
    assert_phases_5_6_ready(results)
    assert set(results["hypothesis_decisions"]["hypothesis"]) == {
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
    }
