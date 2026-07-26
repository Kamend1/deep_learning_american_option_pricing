# Results Required Before Final Writing

Notebook 09 is allowed to run in placeholder mode, but the final academic
conclusion must not be written until the following evidence exists.

## Data foundation

- Production manifest confirms exactly **1,450,000 observations**.
- Every Parquet chunk is present and readable.
- Frozen train, validation, in-domain test, and OOD assignments are available.
- Feature scalers were fitted only on training observations.
- Test and OOD rows were excluded from all model selection.

## Notebook 04 — Direct MLP

- Best checkpoint and feature scaler.
- Training history and selected epoch.
- Black–Scholes proxy baseline.
- In-domain predictions and full regression metrics.
- Segmented error tables.
- Financial-violation report.
- OOD predictions and metrics.
- CPU and GPU inference benchmark where available.

## Notebook 05 — Premium and constrained models

- Zero-premium and mean-premium baselines.
- Unconstrained, non-negative-premium, and floor-residual checkpoints.
- Validation ablation for loss weighting.
- Premium metrics and reconstructed-price metrics.
- Financial-constraint comparisons.
- OOD results.
- H2 and H3 evidence.

## Notebook 06 — Exercise boundary

- Exercise-only classifier results.
- Multi-task checkpoint and selected loss weight.
- Classification and calibration metrics.
- Near-boundary metrics.
- Reconstructed boundary curves.
- Boundary-location error.
- H4 evidence.

## Notebook 07 — Longstaff–Schwartz

- GBM moment validation.
- Classical basis-function comparison.
- Path-count and exercise-date convergence.
- Independent policy-training and valuation paths.
- Neural continuation-policy checkpoint.
- Held-out contract pricing results.
- Monte Carlo confidence intervals.
- Stopping-policy and boundary metrics.
- OOD and multi-seed robustness.
- Runtime decomposition.

## Notebook 08 — Integrated multi-head model

- Scratch and warm-start training results.
- Balanced, pricing-focused, and decision-focused validation ablation.
- Final checkpoint and feature scaler.
- Constrained-price, direct-price, continuation, and exercise metrics.
- Internal consistency and contradiction metrics.
- Boundary curves.
- OOD predictions.
- Runtime summary.

## Final cross-model tables

- Common in-domain pricing table.
- Financial-consistency table.
- Exercise-boundary table.
- OOD deterioration table.
- Runtime and throughput table.
- Static-model ablation table.
- Classical versus neural LSM table.
- H1-H6 decision table.

## Literature

- All ten supplied papers are identified in `references.bib`.
- Each paper is discussed substantively in at least one notebook.
- Foundational Black–Scholes, Merton, CRR, and Longstaff–Schwartz sources are cited.
- Every final literature comparison is supported by the actual reported result.
