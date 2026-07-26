# Final Write-up Checklist

## Every analytical section

- State the analytical objective before code.
- Show the relevant output, table, or figure.
- Add a section conclusion after every material result.
- Explain the financial meaning, not only the metric.
- Compare the result with the predefined expectation.
- Identify limitations and possible numerical artifacts.
- State the implication for the next section.

## Every notebook

- Add a substantial notebook conclusion.
- Summarize the principal numerical findings.
- State which hypotheses are affected.
- Compare findings with relevant papers.
- List generated artifacts.
- Add an explicit handoff to the next notebook.
- Regenerate the Markdown twin after the conclusion is finalized.
- Confirm that figures and tables render correctly on GitHub.

## Notebook-to-notebook handoff

Each handoff should state:

- inputs received from the previous notebook;
- artifacts created in the current notebook;
- model or configuration selected;
- unresolved risks;
- exact files expected by the next notebook.

## Notebook 09 final synthesis

- Distinguish what the data demonstrate, suggest, and do not establish.
- Compare every model on aligned observations.
- Separate in-domain interpolation from OOD robustness.
- Separate financial guarantees from empirical consistency.
- Separate up-front training cost from marginal inference speed.
- Discuss static neural surrogates separately from path-based neural LSM.
- Decide H1-H6 using the predefined rules and academic judgment.
- Explain disagreements with prior literature.
- State practical implications without overstating deployment readiness.
- Provide limitations and future research.
- Answer the central research question directly.

## Repository and release

- `python -m pytest -q` passes.
- `python -m pytest -q -m integration` passes.
- Slow tests are either completed or explicitly documented.
- `python scripts/validate_production_project.py --deep` passes.
- `python scripts/build_final_results.py --strict` passes.
- README contains actual results rather than planned claims.
- `requirements.txt` matches all active imports.
- Executed notebooks contain outputs.
- Markdown twins are refreshed.
- Large data and checkpoints remain excluded from Git.
- Small manifests, tables, and final figures are tracked.
