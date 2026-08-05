from src.evaluation.artifact_registry import (
    assert_required_artifacts_valid,
    audit_artifacts,
)


def test_current_artifact_contract_accepts_complete_project(final_project):
    audit = audit_artifacts(final_project)
    assert_required_artifacts_valid(audit)
    required = audit.loc[audit["required_for_final"]]
    assert required["found"].all()
    assert required["valid"].all()


def test_registry_reports_current_notebook08_boundary_and_runtime_schema(final_project):
    audit = audit_artifacts(final_project).set_index("name")
    assert audit.loc["nb08_boundary_analysis", "valid"]
    assert audit.loc["nb08_runtime", "valid"]
    assert audit.loc["nb08_boundary_analysis", "rows"] == 2
