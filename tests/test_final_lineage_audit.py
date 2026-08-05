from pathlib import Path

import pytest

from conftest import build_final_project
from src.evaluation.final_artifact_adapters import load_all_final_packages
from src.evaluation.final_lineage_audit import (
    assert_phase_1_3_ready,
    audit_static_prediction_alignment,
    run_phase_1_3_audit,
)


def test_static_predictions_use_same_samples_and_targets(final_project):
    packages = load_all_final_packages(final_project)
    summary, fields = audit_static_prediction_alignment(packages)
    assert summary["same_sample_id_set"].all()
    assert summary["same_true_target"].all()
    assert summary["valid"].all()
    assert fields["matches"].all()



def test_float32_rounding_difference_is_accepted(tmp_path: Path):
    project = build_final_project(tmp_path, target_shift=3e-8)
    packages = load_all_final_packages(project)
    summary, _ = audit_static_prediction_alignment(packages)
    notebook08 = summary.loc[summary["notebook"].eq("08")].iloc[0]
    assert notebook08["target_max_absolute_difference"] > 0.0
    assert notebook08["same_true_target"]
    assert notebook08["valid"]

def test_target_mismatch_is_detected(tmp_path: Path):
    project = build_final_project(tmp_path, target_shift=0.01)
    packages = load_all_final_packages(project)
    summary, _ = audit_static_prediction_alignment(packages)
    notebook08 = summary.loc[summary["notebook"].eq("08")].iloc[0]
    assert not notebook08["same_true_target"]
    assert not notebook08["valid"]


def test_phase_1_3_audit_is_strict(final_project):
    results = run_phase_1_3_audit(final_project)
    assert_phase_1_3_ready(results)


def test_phase_1_3_ready_raises_for_target_drift(tmp_path: Path):
    project = build_final_project(tmp_path, target_shift=0.01)
    results = run_phase_1_3_audit(project)
    with pytest.raises(RuntimeError, match="Static prediction alignment failed"):
        assert_phase_1_3_ready(results)
