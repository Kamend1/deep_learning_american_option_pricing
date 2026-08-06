# Dataset Schema and Splits

## Purpose

This document defines the data contract used by every static neural experiment. It also documents the split and preprocessing controls that prevent leakage.

## Core feature vector

All static models receive the same ordered feature vector:

```python
FEATURE_COLUMNS = (
    "log_moneyness",
    "time_to_maturity",
    "risk_free_rate",
    "dividend_yield",
    "volatility",
)
```

Mathematically:

$$
x =
\left[
\log(S/K),
T,
r,
q,
\sigma
\right].
$$

Strike is fixed at 100 during generation, and price targets are normalized by strike. This removes an unnecessary absolute scale while preserving the economically relevant state.

## Canonical schema

| Column | Meaning | Unit or normalization | Source | Principal use |
|---|---|---|---|---|
| `sample_id` | Stable global row identifier | Integer | generation | alignment and lineage |
| `component` | Dataset component | category | generation | core/boundary/OOD analysis |
| `split` | Frozen allocation | category | generation | train/validation/test |
| `spot` | Underlying price | currency | sampled | diagnostics |
| `strike` | Option strike | currency | fixed at 100 | normalization |
| `moneyness` | $S/K$ | ratio | derived | segmentation |
| `log_moneyness` | $\log(S/K)$ | unitless | derived | model input |
| `time_to_maturity` | $T$ | years | sampled | model input |
| `risk_free_rate` | $r$ | annual continuous rate | sampled | model input |
| `dividend_yield` | $q$ | annual continuous yield | sampled | model input |
| `volatility` | $\sigma$ | annualized | sampled | model input |
| `intrinsic_value` | $\max(K-S,0)$ | currency | pricing | lower bound |
| `continuation_value` | Root continuation value | currency | CRR | labels and boundary |
| `european_price` | Black–Scholes put value | currency | pricing | benchmark and lower bound |
| `raw_american_price` | Unfloored CRR value | currency | CRR | audit |
| `american_price` | Validated target | currency | derived | final target |
| `pricing_floor_adjustment` | $V_A-V_A^{raw}$ | currency | derived | numerical audit |
| `early_exercise_premium` | $V_A-V_E$ | currency | derived | residual target |
| `normalized_european_price` | $V_E/K$ | ratio | derived | reconstruction |
| `normalized_american_price` | $V_A/K$ | ratio | derived | direct target |
| `normalized_early_exercise_premium` | $(V_A-V_E)/K$ | ratio | derived | premium target |
| `boundary_distance_normalized` | $(I-C)/K$ | ratio | derived | boundary evaluation |
| `exercise_now` | Root stopping decision | Boolean | CRR | classifier target |
| `tree_steps` | CRR resolution | integer | configuration | audit |

## Additional derived targets

Notebook-specific target builders add:

### Normalized intrinsic value

$$
\widetilde I=\frac{I}{K}.
$$

### Financial floor

$$
\widetilde F=
\max(\widetilde V_E,\widetilde I).
$$

### Floor residual

$$
R_F=
\widetilde V_A-\widetilde F.
$$

Numerical noise is clipped at zero when creating the training target.

### Normalized continuation value

$$
\widetilde C=\frac{C}{K}.
$$

These derived targets are added before constructing multi-task or integrated datasets.

## Production split assignment

The production core and boundary files contain fixed split labels.

Default shares:

- train: 70%;
- validation: 15%;
- test: 15%.

Only `core` and `boundary` are split eligible. OOD components remain separate.

The project also provides general split utilities in `src/data/splitting.py` for deterministic, optionally stratified allocation and split-manifest generation.

## Split integrity

The split layer validates:

- required split names;
- identifier presence;
- unique IDs;
- disjoint membership;
- stored labels matching their container;
- complete row representation.

Exercise-aware allocation preserves the exercise/continuation class distribution as closely as practical.

## Leakage controls

### Training-only scaler

`fit_feature_scaler` fits `sklearn.preprocessing.StandardScaler` only on training observations.

For feature $j$:

$$
z_j=
\frac{x_j-\mu_{j,\text{train}}}
{\sigma_{j,\text{train}}}.
$$

The same saved scaler is applied unchanged to validation, test, and OOD sets.

### Validation-only selection

Validation data are used for:

- best epoch;
- learning-rate scheduling;
- early stopping;
- residual candidate selection;
- multi-task loss-weight selection;
- exercise threshold selection;
- integrated configuration selection.

### Test isolation

The common static test set is not used to:

- fit scalers;
- train weights;
- stop training;
- select candidate models;
- choose classification thresholds.

### OOD isolation

OOD sets are never used for:

- training;
- preprocessing;
- hyperparameter selection;
- checkpoint selection.

They are used only after the in-domain selection is frozen.

### Path-based independence

The Longstaff–Schwartz branch uses separate contract samples and separate policy-training and policy-valuation paths. It is not aligned row by row with the static test set.

## PyTorch dataset interfaces

File: `src/data/torch_datasets.py`

### `AmericanOptionDataset`

Returns:

- `features`
- `target`
- `row_id`
- optional `sample_weight`

Used by direct and residual regression.

### `MultiTaskAmericanOptionDataset`

Returns:

- `features`
- `residual_target`
- `exercise_target`
- normalized European value
- normalized intrinsic value
- normalized American value
- `row_id`
- optional sample weight

Used by the exercise-only and joint price/exercise experiments.

### `IntegratedMultiHeadDataset`

Returns targets for:

- floor residual;
- direct price;
- continuation value;
- exercise decision;
- European and intrinsic values required for reconstruction.

### DataLoader reproducibility

`LoaderConfig` controls:

- batch size;
- worker count;
- pin memory;
- random seed.

A fixed `torch.Generator` controls shuffle order. Worker seeds are propagated to NumPy and Python when multiprocessing is enabled.

## Selective Parquet loading

`read_parquet_components` reads only requested columns and may apply a Parquet `split` filter. This avoids loading all 1.25 million in-domain rows when only one split and a subset of fields are required.

Optional row limits support development and smoke profiles. Such results are not accepted as final evidence.

## Common-test alignment

Notebook 09 treats Notebook 04 as the reference static prediction set and verifies for Notebooks 05, 06, 08, and the Notebook 08 scratch benchmark:

- no duplicate IDs;
- no missing reference IDs;
- no extra IDs;
- identical true normalized targets within tolerance;
- identical shared state fields where exported.

The final common test contains 187,811 observations.

## Failure modes

A dataset constructor rejects:

- missing required columns;
- NaN or infinite features;
- NaN or infinite targets;
- negative non-negative targets;
- non-binary exercise labels;
- negative sample weights;
- incompatible scaler objects.

These checks prevent invalid arrays from reaching the training loop.

## Related notebooks

- Notebook 03: production design and frozen splits
- Notebook 04: scaler fitting and baseline dataset
- Notebook 05: residual targets
- Notebook 06: classification targets
- Notebook 08: integrated targets
- Notebook 09: common-test alignment
