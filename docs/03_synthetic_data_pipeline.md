# Synthetic Data Pipeline

## Purpose

The project requires a large, controlled dataset containing not only American put prices but also continuation values, early-exercise premiums, exercise labels, and boundary diagnostics. Those fields are not available as a complete clean market dataset across the required parameter space.

The production data pipeline therefore samples contract states and labels them with the validated CRR engine.

## Production design

File: `src/data/production_generation.py`

Configuration class: `ProductionDatasetConfig`

The default design contains:

| Component | Observations | Split eligible | Purpose |
|---|---:|---:|---|
| `core` | 1,000,000 | Yes | Broad interpolation domain |
| `boundary` | 250,000 | Yes | Higher density near the exercise boundary |
| `ood_high_volatility` | 50,000 | No | Volatility above the core maximum |
| `ood_extreme_moneyness` | 50,000 | No | Deep ITM and deep OTM contracts |
| `ood_long_maturity` | 50,000 | No | Maturities above two years |
| `ood_rate_dividend` | 50,000 | No | Jointly elevated rates and dividends |
| **Total** | **1,450,000** |  |  |

Default generation controls:

- CRR steps: 250
- strike: 100
- chunk size: 25,000
- seed: 42
- Parquet compression: Zstandard
- compression level: 3

## Parameter domains

### Core domain

| Parameter | Minimum | Maximum |
|---|---:|---:|
| Moneyness $S/K$ | 0.50 | 1.50 |
| Maturity $T$ | $7/365$ | 2.00 |
| Volatility $\sigma$ | 0.05 | 0.80 |
| Risk-free rate $r$ | 0.00 | 0.10 |
| Dividend yield $q$ | 0.00 | 0.08 |

The core domain covers broad practical ranges rather than a narrow calibration sample.

### Boundary candidate domain

The boundary component samples from a domain emphasizing states where early exercise is more likely to change:

- moneyness: 0.45–1.10;
- maturity: $7/365$–1.50;
- volatility: 0.05–0.60;
- rate: 0.01–0.15;
- dividend yield: 0.00–0.06.

Candidates are priced and ranked by distance from the stopping boundary before the final boundary sample is retained.

### OOD domains

OOD sets are generated separately and receive no `train`, `validation`, or `test` assignment.

- High volatility: $\sigma \in [0.80,1.20]$
- Long maturity: $T \in [2,4]$
- Deep ITM: $S/K \in [0.25,0.50]$
- Deep OTM: $S/K \in [1.50,2.00]$
- High-rate/high-dividend: $r \in [0.10,0.20]$, $q \in [0.08,0.15]$

These are stress regimes, not a random tail of the training sample.

## Sampling method

Function: `sample_parameter_chunk`

Each chunk uses a deterministic randomized Latin hypercube across five dimensions.

For each dimension, the unit interval is divided into $n$ strata. One randomized point is sampled per stratum and then independently permuted. Values are scaled to the component bounds.

This improves marginal coverage relative to independent random sampling of the same size.

For standard components:

$$
S=K\cdot(S/K).
$$

The saved feature is:

$$
\log(S/K).
$$

For `ood_extreme_moneyness`, half the values are sampled from the deep-ITM range and half from the deep-OTM range, then permuted.

## Stable global identifiers

`build_component_specs` allocates a fixed, non-overlapping `sample_id` interval to every component.

This supports:

- deterministic regeneration;
- split stability;
- exact prediction alignment across notebooks;
- duplicate detection;
- contract-level audit trails.

## Label construction

For each sampled contract, the Numba pricing batch returns:

- European Black–Scholes value $V_E$;
- raw CRR American value $V_A^{raw}$;
- intrinsic value $I$;
- continuation value $C$;
- exercise decision.

The validated American target is:

$$
V_A=
\max(V_A^{raw},V_E,I).
$$

The early-exercise premium is:

$$
EEP=V_A-V_E.
$$

The normalized targets are:

$$
\widetilde V_E=\frac{V_E}{K},
$$

$$
\widetilde V_A=\frac{V_A}{K},
$$

$$
\widetilde{EEP}=\frac{V_A-V_E}{K}.
$$

The signed normalized boundary distance is based on intrinsic minus continuation:

$$
D=\frac{I-C}{K}.
$$

Positive $D$ indicates the exercise region; negative $D$ indicates continuation.

## Output schema

`REQUIRED_OUTPUT_COLUMNS` defines the canonical production schema:

- `sample_id`
- `component`
- `split`
- `spot`
- `strike`
- `moneyness`
- `log_moneyness`
- `time_to_maturity`
- `risk_free_rate`
- `dividend_yield`
- `volatility`
- `intrinsic_value`
- `continuation_value`
- `european_price`
- `raw_american_price`
- `american_price`
- `pricing_floor_adjustment`
- `early_exercise_premium`
- `normalized_european_price`
- `normalized_american_price`
- `normalized_early_exercise_premium`
- `boundary_distance_normalized`
- `exercise_now`
- `tree_steps`

## Chunked generation

The full dataset is never required in memory.

`generate_component`:

1. determines chunk boundaries;
2. samples deterministic parameters;
3. prices the chunk in parallel through Numba;
4. constructs all raw and derived fields;
5. validates the chunk;
6. writes it through a Parquet writer;
7. accumulates component statistics.

Benefits:

- bounded memory use;
- restartability;
- progress reporting;
- component-level verification;
- compressed columnar storage;
- selective downstream reads.

## Restartable script

Entry point:

```bash
python scripts/generate_production_dataset.py
```

Important options:

```bash
python scripts/generate_production_dataset.py \
  --tree-steps 250 \
  --chunk-size 25000 \
  --seed 42
```

Generate selected components:

```bash
python scripts/generate_production_dataset.py \
  --component core \
  --component boundary
```

Overwrite existing outputs:

```bash
python scripts/generate_production_dataset.py --overwrite
```

Existing components are inspected and reused unless overwrite is requested. The complete manifest is written only after all six components are present.

## Production manifest

Path:

```text
data/manifests/production_dataset_manifest.json
```

The manifest records:

- configuration;
- component purposes and ranges;
- expected and observed counts;
- identifier ranges;
- split counts;
- exercise counts;
- floor-adjustment counts;
- maximum adjustment;
- file paths;
- SHA-256 hashes;
- generation timestamp.

The manifest is later fingerprinted by model training packages.

## Validation and failure modes

Generation fails or is rejected when:

- parameter bounds are invalid;
- component counts do not match the design;
- sample IDs overlap;
- required columns are absent;
- values are non-finite;
- pricing identities fail;
- exercise labels are inconsistent;
- output row counts differ from specifications;
- a completed file hash changes unexpectedly.

The raw American value and floor adjustment are retained so numerical repairs cannot be hidden.

## Related notebooks

- Notebook 02: pilot engine validation and generation
- Notebook 03: schema, coverage, boundary density, and split design
- Notebooks 04–08: downstream consumers
