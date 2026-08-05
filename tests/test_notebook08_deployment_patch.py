from pathlib import Path

import nbformat


def test_notebook_contains_separate_scratch_and_deployment_roles():
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "08_final_multihead_model.ipynb"
    )
    notebook = nbformat.read(path, as_version=4)
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )

    assert "best_integrated_scratch.pt" in code
    assert "best_integrated_deployment.pt" in code
    assert '"test_metrics_used_for_selection": False' in code
    assert '"ood_metrics_used_for_selection": False' in code
    assert "preferred in-domain integrated deployment model" in markdown
    assert "High-resolution CRR tree" in markdown
