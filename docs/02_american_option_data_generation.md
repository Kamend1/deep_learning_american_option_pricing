<!--
Generated from notebooks/02_american_option_data_generation.ipynb.
Keep the notebook as the executable source of truth and refresh this file after material notebook changes.
-->

# American Option Pricing Engine Validation and Pilot Dataset Generation

## Notebook 02

This notebook validates the classical pricing engines before they are used to generate deep-learning labels. The analysis follows a strict sequence:

1. run unit tests;
2. verify European CRR convergence toward Black–Scholes;
3. measure American-price stability and runtime;
4. test financial lower bounds and root exercise decisions;
5. select a defensible production tree resolution;
6. generate and validate a pilot synthetic dataset.

The notebook does **not** train a neural network. Its purpose is to establish that the future supervised targets are numerically credible, financially coherent, and computationally feasible.

## 1. Environment and imports

Reusable pricing and generation logic is imported from `src/`. Generated data are written under `data/generated/`, which should remain excluded from Git through `.gitignore`.

```python
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NOTEBOOK_DIR = Path.cwd().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR
DATA_DIR = PROJECT_ROOT / "data" / "generated"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")
print(f"Generated data directory: {DATA_DIR}")
```

```python
from src.data.generation import (
    ParameterRanges,
    generate_american_put_dataset,
    save_generated_dataset,
)
from src.pricing.black_scholes import black_scholes_put_price
from src.pricing.binomial_tree import crr_option_diagnostics, crr_option_price
from src.pricing.validation import (
    build_crr_convergence_table,
    build_financial_validation_grid,
    select_production_steps,
    summarize_dataset_financial_checks,
    quantlib_american_put_price,
)
```

## 2. Unit-test gate

The source modules should not be used for large-scale label generation until the pricing tests pass. The following cell runs only the two pricing test files relevant to this stage.

```python
pytest_result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(PROJECT_ROOT / "tests" / "test_black_scholes.py"),
        str(PROJECT_ROOT / "tests" / "test_binomial_tree.py"),
    ],
    cwd=PROJECT_ROOT,
    capture_output=True,
    text=True,
)

print(pytest_result.stdout)
if pytest_result.returncode != 0:
    print(pytest_result.stderr)
    raise RuntimeError("Pricing tests failed. Stop before generating data.")
```

## 3. Representative validation contract

The convergence study uses one fixed contract so the impact of tree resolution is isolated. This contract is slightly in the money and has positive rates and dividends, making the American exercise feature relevant without using an extreme parameter combination.

```python
validation_contract = {
    "spot": 90.0,
    "strike": 100.0,
    "time_to_maturity": 1.0,
    "risk_free_rate": 0.05,
    "dividend_yield": 0.02,
    "volatility": 0.25,
}

pd.Series(validation_contract, name="Value").to_frame()
```

## 4. CRR convergence and runtime

European CRR values can be compared with the closed-form Black–Scholes price. American values have no closed-form benchmark under the selected assumptions, so each tested tree is compared with a finer 2,000-step reference.

The runtime estimate is deliberately included. A tree resolution that is marginally more accurate but makes hundreds of thousands of labels operationally impractical is not an efficient production choice.

```python
STEPS_GRID = [25, 50, 100, 250, 500, 1_000]
REFERENCE_STEPS = 2_000

convergence_table = build_crr_convergence_table(
    validation_contract,
    steps_grid=STEPS_GRID,
    reference_steps=REFERENCE_STEPS,
    timing_repeats=3,
    portfolio_size=100_000,
)

convergence_table
```

```python
plt.figure(figsize=(10, 5))
plt.plot(
    convergence_table["steps"],
    convergence_table["european_abs_error"],
    marker="o",
    label="European CRR error vs Black–Scholes",
)
plt.plot(
    convergence_table["steps"],
    convergence_table["american_reference_abs_error"],
    marker="o",
    label="American CRR error vs 2,000-step reference",
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("CRR time steps")
plt.ylabel("Absolute pricing error")
plt.title("Pricing Accuracy by Tree Resolution")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
```

```python
plt.figure(figsize=(10, 5))
plt.plot(
    convergence_table["steps"],
    convergence_table["seconds_per_option"],
    marker="o",
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("CRR time steps")
plt.ylabel("Median seconds per American option")
plt.title("CRR Runtime by Tree Resolution")
plt.grid(alpha=0.25)
plt.show()
```

### Production-step decision rule

The production resolution is selected mechanically as the smallest tested tree satisfying both error tolerances **and** a minimum resolution of 250 steps. The minimum is a conservative data-quality rule intended to reduce finite-tree oscillation across a broad synthetic domain, rather than optimize only for the representative contract:

- European absolute error no greater than 0.02 currency units;
- American difference from the 2,000-step reference no greater than 0.02 currency units.

These tolerances are research-design choices rather than universal market standards. They should be revisited if the convergence results reveal oscillation or unacceptable error in other parameter regimes.

```python
PRODUCTION_TREE_STEPS = select_production_steps(
    convergence_table,
    max_european_abs_error=0.02,
    max_american_reference_abs_error=0.02,
    minimum_steps=250,
)

print(f"Selected pilot production tree resolution: {PRODUCTION_TREE_STEPS} steps")
```

## 4.1 Optional independent QuantLib check

The project uses CRR labels, but a representative American put should also be compared with an independently implemented finite-difference engine. This cell is optional during early development and becomes a required validation gate before full production generation. Install the optional dependency with `pip install QuantLib`.

```python
try:
    quantlib_price = quantlib_american_put_price(
        validation_contract,
        time_steps=400,
        grid_points=400,
    )
    crr_reference_price = crr_option_price(
        **validation_contract,
        steps=REFERENCE_STEPS,
        option_type="put",
        exercise_style="american",
    )
    quantlib_comparison = pd.Series({
        "QuantLib finite-difference price": quantlib_price,
        "CRR reference price": crr_reference_price,
        "Absolute difference": abs(quantlib_price - crr_reference_price),
    }, name="Value")
    display(quantlib_comparison.to_frame())
except ImportError as error:
    print(error)
    print("QuantLib validation skipped for this run.")
```

## 5. Financial-bound and exercise-decision validation

The next grid varies spot while holding all other parameters fixed. It checks that:

- American price is not below intrinsic value;
- American price is not below the European benchmark;
- the early-exercise premium is non-negative;
- the root exercise label changes coherently with moneyness.

```python
spot_values = np.linspace(50.0, 150.0, 101)

financial_grid = build_financial_validation_grid(
    validation_contract,
    spot_values=spot_values,
    steps=PRODUCTION_TREE_STEPS,
)

financial_grid.head()
```

```python
financial_grid[[
    "american_ge_intrinsic",
    "american_ge_european",
]].all().to_frame("Passed")
```

```python
plt.figure(figsize=(11, 6))
plt.plot(financial_grid["spot"], financial_grid["intrinsic_value"], label="Intrinsic")
plt.plot(financial_grid["spot"], financial_grid["european_price"], label="European Black–Scholes")
plt.plot(financial_grid["spot"], financial_grid["american_price"], label="American CRR")
plt.axvline(validation_contract["strike"], linestyle="--", label="Strike")
plt.xlabel("Underlying price")
plt.ylabel("Put value")
plt.title("American Put Lower-Bound Validation")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
```

```python
exercise_points = financial_grid.loc[financial_grid["exercise_now"]]
continuation_points = financial_grid.loc[~financial_grid["exercise_now"]]

plt.figure(figsize=(11, 5))
plt.scatter(
    continuation_points["spot"],
    continuation_points["continuation_value"] - continuation_points["intrinsic_value"],
    label="Continue",
    s=22,
)
plt.scatter(
    exercise_points["spot"],
    exercise_points["continuation_value"] - exercise_points["intrinsic_value"],
    label="Exercise now",
    s=22,
)
plt.axhline(0.0, linestyle="--")
plt.xlabel("Underlying price")
plt.ylabel("Continuation value minus intrinsic value")
plt.title("Root Exercise Decision Across Spot Prices")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
```

## 6. Synthetic parameter domain

The first dataset fixes strike at 100 and samples moneyness directly. This uses option-price homogeneity and avoids adding a redundant absolute scale. The five-dimensional domain is sampled using a randomized Latin hypercube to improve coverage relative to independent random draws of the same size.

```python
parameter_ranges = ParameterRanges(
    moneyness=(0.50, 1.50),
    time_to_maturity=(7.0 / 365.0, 2.0),
    volatility=(0.05, 0.80),
    risk_free_rate=(0.00, 0.10),
    dividend_yield=(0.00, 0.08),
)

pd.DataFrame(
    {
        "Minimum": [
            parameter_ranges.moneyness[0],
            parameter_ranges.time_to_maturity[0],
            parameter_ranges.volatility[0],
            parameter_ranges.risk_free_rate[0],
            parameter_ranges.dividend_yield[0],
        ],
        "Maximum": [
            parameter_ranges.moneyness[1],
            parameter_ranges.time_to_maturity[1],
            parameter_ranges.volatility[1],
            parameter_ranges.risk_free_rate[1],
            parameter_ranges.dividend_yield[1],
        ],
    },
    index=[
        "S / K",
        "Time to maturity",
        "Volatility",
        "Risk-free rate",
        "Dividend yield",
    ],
)
```

## 7. Pilot dataset generation

The default pilot contains 2,000 observations. This is intentionally smaller than the planned production dataset. Its purpose is to expose schema, runtime, numerical, and class-balance problems before expensive generation begins.

Increase `PILOT_SIZE` to 10,000 only after the preceding runtime table is acceptable on the execution machine.

```python
PILOT_SIZE = 2_000

last_reported_percent = -1

def report_progress(completed: int, total: int) -> None:
    global last_reported_percent
    percent = int(100 * completed / total)
    if percent // 10 > last_reported_percent // 10:
        print(f"Generated {completed:,} / {total:,} observations ({percent}%)")
        last_reported_percent = percent

pilot_dataset = generate_american_put_dataset(
    n_samples=PILOT_SIZE,
    tree_steps=PRODUCTION_TREE_STEPS,
    seed=42,
    strike=100.0,
    ranges=parameter_ranges,
    progress_callback=report_progress,
)

pilot_dataset.head()
```

## 8. Pilot financial validation

No dataset should be accepted merely because generation completed without raising an exception. The next table counts lower-bound and exercise-label violations across every generated observation.

A finite CRR lattice may oscillate slightly below the analytical European value even though the theoretical American value cannot do so. The generator therefore retains `raw_american_price` and applies a transparent no-arbitrage floor to create `american_price`. The adjustment is stored in `pricing_floor_adjustment` and reviewed below rather than hidden.

```python
pilot_checks = summarize_dataset_financial_checks(pilot_dataset)
pilot_checks

repair_summary = pd.Series({
    "observations_repaired": int((pilot_dataset["pricing_floor_adjustment"] > 0).sum()),
    "repair_rate": float((pilot_dataset["pricing_floor_adjustment"] > 0).mean()),
    "maximum_repair": float(pilot_dataset["pricing_floor_adjustment"].max()),
    "mean_positive_repair": float(
        pilot_dataset.loc[
            pilot_dataset["pricing_floor_adjustment"] > 0,
            "pricing_floor_adjustment",
        ].mean()
    ) if (pilot_dataset["pricing_floor_adjustment"] > 0).any() else 0.0,
}, name="Value")

repair_summary.to_frame()
```

```python
assert pilot_checks["passed"].all(), "The pilot dataset contains financial violations."
```

## 9. Pilot coverage and target structure

The distributions below verify that the Latin hypercube covers the intended ranges and reveal the shape of the pricing targets. The exercise label is expected to be imbalanced because immediate exercise is optimal only in a subset of the parameter domain. That imbalance should be measured now because it affects the later multi-task classification loss.

```python
pilot_dataset[[
    "moneyness",
    "time_to_maturity",
    "volatility",
    "risk_free_rate",
    "dividend_yield",
    "normalized_american_price",
    "normalized_early_exercise_premium",
]].describe().T
```

```python
exercise_distribution = (
    pilot_dataset["exercise_now"]
    .value_counts(dropna=False)
    .rename_axis("exercise_now")
    .to_frame("count")
)
exercise_distribution["proportion"] = exercise_distribution["count"] / len(pilot_dataset)
exercise_distribution
```

```python
plt.figure(figsize=(10, 5))
plt.scatter(
    pilot_dataset["moneyness"],
    pilot_dataset["normalized_early_exercise_premium"],
    s=10,
    alpha=0.45,
)
plt.axhline(0.0, linestyle="--")
plt.axvline(1.0, linestyle="--")
plt.xlabel("Moneyness S / K")
plt.ylabel("Normalized early-exercise premium")
plt.title("Pilot Early-Exercise Premium by Moneyness")
plt.grid(alpha=0.25)
plt.show()
```

## 10. Save the pilot dataset

CSV is used for the first pilot because it has no optional storage-engine dependency. The full production dataset can later be stored as Parquet after `pyarrow` is added to the environment.

```python
pilot_path = save_generated_dataset(
    pilot_dataset,
    DATA_DIR / "american_put_pilot.csv",
)

print(f"Saved pilot dataset to: {pilot_path}")
print(f"File size: {pilot_path.stat().st_size / 1_000_000:.2f} MB")
```

# 11. Approval gate before full generation

Proceed to a full dataset only when all of the following conditions are met:

- both pricing test files pass;
- the selected CRR resolution satisfies the predefined convergence tolerances;
- the financial spot grid contains no lower-bound violations;
- the pilot dataset contains no financial-check violations;
- the exercise class is present and its imbalance is documented;
- estimated full-generation runtime is acceptable;
- the generated schema contains every input, target, normalization field, and diagnostic required by the planned models.

The next notebook will analyze the accepted dataset, establish train/validation/test and out-of-domain partitions, and fit preprocessing exclusively on the training partition.
