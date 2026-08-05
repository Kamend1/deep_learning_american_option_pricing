import json

import pytest

from src.evaluation.final_artifact_adapters import (
    FinalPackageError,
    build_package_summary,
    load_all_final_packages,
    load_notebook08_package,
)


def test_explicit_adapters_load_all_final_packages(final_project):
    packages = load_all_final_packages(final_project)
    assert set(packages) == {"04", "05", "06", "07", "08"}
    summary = build_package_summary(packages).set_index("notebook")
    assert summary.loc["05", "selected_model"] == "Constrained floor residual"
    assert summary.loc["07", "training_profile"] == "final"
    assert summary.loc["08", "checkpoint"] == "best_integrated_multihead.pt"


def test_notebook08_adapter_rejects_selection_checkpoint_mismatch(final_project):
    path = final_project / "artifacts/final_multihead/selection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["canonical_checkpoint"] = "wrong_checkpoint.pt"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FinalPackageError, match="canonical checkpoint"):
        load_notebook08_package(final_project)
