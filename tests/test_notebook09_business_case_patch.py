from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_notebook09_contains_the_measured_business_case() -> None:
    notebook = nbformat.read(
        PROJECT_ROOT / "notebooks" / "09_final_evaluation.ipynb",
        as_version=4,
    )
    text = "\n".join(cell.source for cell in notebook.cells)
    assert "## 21. The central business question" in text
    assert "run_final_business_case(" in text
    assert "static_model_metrics=static_model_metrics" in text
    assert "operational_crossover" in text
    assert "lifecycle_break_even" in text
    assert "research_question_7_summary" in text
    assert "runtime_environment.json" in text
    assert "research_question_7.md" in text


def test_notebook08_handoff_does_not_prejudge_the_business_case() -> None:
    notebook = nbformat.read(
        PROJECT_ROOT / "notebooks" / "08_final_multihead_model.ipynb",
        as_version=4,
    )
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    assert "not the economic justification for deploying it" in markdown
    assert "measured workload crossover" in markdown
    assert "Notebook 09" in markdown
