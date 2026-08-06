# Training and Artifact Management

## Purpose

This document explains how models are optimized, selected, saved, reused, and validated as complete experimental packages.

## Shared regression training loop

File: `src/training/loops.py`

Configuration: `TrainingConfig`

Default controls:

| Control | Default |
|---|---:|
| Maximum epochs | 100 |
| Learning rate | $10^{-3}$ |
| Weight decay | $10^{-5}$ |
| Early-stopping patience | 10 |
| Scheduler patience | 4 |
| Scheduler factor | 0.5 |
| Minimum learning rate | $10^{-6}$ |
| Gradient clipping | 1.0 |
| Mixed precision | Enabled on CUDA |
| Seed | 42 |
| Minimum improvement | $10^{-8}$ |

### Reproducibility

`set_global_seed` seeds:

- Python;
- NumPy;
- PyTorch CPU;
- all CUDA devices.

When deterministic mode is enabled:

- cuDNN deterministic behavior is requested;
- cuDNN benchmarking is disabled.

Exact cross-hardware bitwise equality is not guaranteed, but the configuration removes uncontrolled seed variation.

### Optimizer and scheduler

Static regression models use AdamW:

$$
\theta_{t+1}
=
\operatorname{AdamW}
\left(
\theta_t,
\nabla_\theta \mathcal L
\right).
$$

`ReduceLROnPlateau` monitors validation loss and reduces the learning rate after the configured patience.

### Mixed precision

Automatic mixed precision is enabled only when:

- the configuration permits it; and
- the device type is CUDA.

CPU training stays in standard precision.

### Gradient clipping

Before the optimizer step, gradients are clipped by global norm when configured. This limits unstable updates without changing the forward architecture.

### Observation-weighted metrics

Epoch loss and MAE are aggregated by observation count rather than averaging batch averages equally. This avoids bias from a smaller final batch.

### Early stopping

The best checkpoint is the epoch with the lowest validation loss after `min_delta`. Training stops after the configured number of non-improving epochs.

The final epoch is not automatically the selected model.

## Losses

### Direct model

Notebook 04 uses Smooth L1 loss for normalized price regression.

### Residual models

File: `src/training/losses.py`

`WeightedSmoothL1Loss` supports optional sample weights.

Weighting alternatives are fitted only on training data:

- unweighted;
- premium-magnitude weighting;
- boundary-aware weighting.

The selected Notebook 05 candidate is chosen by validation performance after reconstructing the economically relevant final price.

### Multi-task loss

File: `src/training/multitask_losses.py`

The price-and-exercise objective combines residual regression with class-weighted binary cross entropy:

$$
\mathcal L
=
\mathcal L_R
+
\lambda_E \mathcal L_E.
$$

The positive-class weight is derived from training class counts and may be capped.

### Integrated multi-head loss

File: `src/training/multihead_losses.py`

The integrated objective combines:

- constrained residual loss;
- direct-price loss;
- continuation-value loss;
- exercise-classification loss;
- price-head consistency;
- decision-path consistency.

Notebook 08 evaluates predefined balanced, pricing-focused, and decision-focused presets rather than unrestricted post-hoc weight search.

## Separate training loops

The project maintains task-specific loops because the output contracts differ:

- `loops.py`: direct and residual regression;
- `multitask_loops.py`: classifier and price/exercise model;
- `lsm_training.py`: backward neural continuation policy;
- `multihead_loops.py`: integrated four-head model.

All loops return structured training histories and write best checkpoints.

## Atomic checkpoints

File: `src/training/checkpointing.py`

`atomic_torch_save` writes to a temporary path and uses `os.replace` to publish the completed checkpoint.

This prevents a partially written checkpoint from being mistaken for a valid training result after interruption.

Typical checkpoint payloads include:

- selected epoch;
- model state;
- optimizer state;
- best validation loss;
- training configuration;
- model configuration;
- task-specific loss configuration;
- thresholds or metadata where applicable.

`load_checkpoint` accepts an explicit `map_location`, supporting CPU loading of GPU-trained models.

## Feature scaler persistence

The static scaler is serialized with Joblib.

Notebook 04 fits the baseline scaler. Later specialist models deliberately reuse it where comparison requires one preprocessing regime.

Notebook 08 saves its own integrated package scaler where the training workflow requires it.

## Completion manifests

File: `src/training/artifact_management.py`

A checkpoint is not treated as a complete experiment by itself.

`write_training_manifest` records:

- `status`;
- notebook identity;
- training profile;
- checkpoint name;
- candidate set;
- selected configuration;
- dependencies;
- execution metadata.

The write is atomic.

`inspect_training_artifacts` verifies:

- every required file exists and is non-empty;
- the manifest is readable;
- status is `complete`;
- notebook identity matches;
- profile matches;
- dependency fingerprints match.

This allows notebooks to load an existing full package without silently accepting a stale or smoke-mode result.

## Dependency fingerprints

File: `src/training/dependency_fingerprints.py`

Training packages fingerprint:

- feature scaler;
- production dataset manifest;
- ordered feature columns.

SHA-256 is calculated in chunks so large files are not loaded into memory.

The dependency record includes repository-relative paths and hashes.

If the scaler, manifest, or feature ordering changes, the package is incompatible even if its checkpoint file still exists.

## Legacy manifest handling

`backfill_manifest_dependencies` can add dependency metadata to an otherwise complete legacy manifest only when explicitly enabled and retraining is not forced.

The migration is recorded through `dependency_metadata_backfilled`.

This mechanism is compatibility support, not permission to ignore mismatched dependencies.

## Final artifact registry

File: `src/evaluation/artifact_registry.py`

Notebook 09 defines an `ArtifactSpec` for every required final artifact.

The registry validates:

- candidate paths;
- loader type;
- required JSON paths;
- required table columns;
- minimum rows;
- expected notebook;
- allowed profile;
- complete status.

Artifact families include:

- production manifest;
- final metrics;
- test predictions;
- checkpoints;
- training manifests;
- model selections;
- boundary analysis;
- runtime summaries;
- deployment policies;
- domain bounds.

## Canonical artifact layout

### Notebook 04

```text
artifacts/direct_mlp/
├── best_direct_mlp.pt
├── feature_scaler.joblib
├── training_history.csv
├── training_complete.json
├── test_predictions.parquet
└── final_metrics.json
```

### Notebook 05

```text
artifacts/premium_models/
├── best_premium_model.pt
├── candidate checkpoints and histories
├── training_complete.json
├── test_predictions.parquet
└── final_metrics.json
```

### Notebook 06

```text
artifacts/multitask_model/
├── best_exercise_classifier.pt
├── best_multitask_pricer.pt
├── exercise_classifier_complete.json
├── multitask_training_complete.json
├── test_predictions.parquet
└── final_metrics.json
```

### Notebook 07

```text
artifacts/neural_lsm/
├── neural_lsm_policy.pt
├── training_complete.json
├── heldout_pricing_results.parquet
└── final_metrics.json
```

### Notebook 08

```text
artifacts/final_multihead/
├── best_integrated_scratch.pt
├── best_integrated_deployment.pt
├── best_integrated_multihead.pt
├── feature_scaler.joblib
├── selection.json
├── deployment_policy.json
├── domain_bounds.json
├── test_predictions.parquet
├── boundary_analysis.csv
├── runtime_summary.json
└── final_metrics.json
```

## Final package coherence

File: `src/evaluation/final_lineage_audit.py`

Before metrics are compared, Notebook 09 verifies:

- final package status;
- checkpoint existence;
- declared checkpoint matching actual filename;
- manifest completion;
- manifest profile matching final package;
- prediction-table existence;
- dependency fingerprints;
- common-test sample alignment.

## Training profiles

The project distinguishes:

- development or smoke execution;
- full static-model execution;
- final Longstaff–Schwartz execution.

Smoke outputs prove that code paths run. They are not accepted as academic evidence by the final artifact registry.

## Failure modes

A package is rejected when:

- a required file is missing or empty;
- status is not complete;
- notebook identity differs;
- profile differs;
- dependency hashes differ;
- required JSON keys are missing;
- prediction schema is incomplete;
- row counts are below the contract;
- the selected checkpoint cannot be resolved.

## Related notebooks

- Notebook 04 establishes the shared regression workflow
- Notebook 05 adds candidate and weighted-loss management
- Notebook 06 adds threshold and multi-task selection
- Notebook 07 stores a time-indexed policy
- Notebook 08 adds scratch/warm-start and deployment selection
- Notebook 09 audits every package
