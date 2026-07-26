<!-- Generated from notebooks/04_direct_mlp_pricer.ipynb. The notebook is the executable source of truth. -->

# Direct MLP Baseline for American Put Pricing

## Production Dataset, Reproducible PyTorch Pipeline, and Baseline Evaluation

**SoftUni Deep Learning Final Project**  
**Notebook:** `04_direct_mlp_pricer.ipynb`

This notebook introduces the first deep-learning model in the project. The model receives only the five normalized pricing inputs and predicts the normalized American put value directly. It is intentionally conventional: its purpose is to establish a credible baseline that the later early-exercise-premium and financially constrained models must outperform.

The production design contains **1,450,000 observations**:

- 1,000,000 core-domain observations;
- 250,000 boundary-focused observations;
- four out-of-domain sets with 50,000 observations each.

Large-scale pricing is performed by `scripts/generate_production_dataset.py`, not inside this notebook. The notebook begins only after the Parquet files and production manifest have been generated.

## 1. Research objective

The direct model estimates

\[
\widehat y = f_\theta(x),
\qquad
x=\left[\log(S/K),T,r,q,\sigma\right],
\qquad
y=\frac{V_A}{K}.
\]

It does not receive European price, intrinsic value, continuation value, early-exercise premium, or the exercise label. Those variables are withheld because they are reserved for later financially structured models.

The direct model is evaluated against the simplest analytical proxy:

\[
\widehat V_A = V_E.
\]

The neural model must materially improve on this Black–Scholes proxy while its financial violations and out-of-domain weaknesses remain visible.

```python
from pathlib import Path
import json
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchinfo import summary

NOTEBOOK_DIR = Path.cwd().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.torch_datasets import (
    AmericanOptionDataset,
    DIRECT_TARGET_COLUMN,
    FEATURE_COLUMNS,
    LoaderConfig,
    create_regression_loader,
    fit_feature_scaler,
    read_parquet_components,
    save_feature_scaler,
)
from src.evaluation.financial_checks import financial_bound_report
from src.evaluation.regression_metrics import (
    compare_models,
    regression_metrics,
    segmented_regression_metrics,
)
from src.models.direct_pricer import DirectAmericanPutMLP, DirectMLPConfig
from src.training.checkpointing import load_checkpoint, save_json
from src.training.loops import (
    TrainingConfig,
    fit_regression_model,
    predict_regression_model,
    set_global_seed,
)
```

## 2. Paths, device, and reproducibility configuration

The full production files remain outside Git. The tracked manifest documents their counts, pricing parameters, split method, and hashes. Training artifacts are also excluded from Git because they can be recreated from the fixed configuration and data-generation process.

```python
DATA_DIR = PROJECT_ROOT / "data" / "generated"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "production_dataset_manifest.json"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "direct_mlp"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

CORE_PATHS = [
    DATA_DIR / "american_put_core.parquet",
    DATA_DIR / "american_put_boundary.parquet",
]
OOD_PATHS = {
    "high_volatility": DATA_DIR / "american_put_ood_high_volatility.parquet",
    "extreme_moneyness": DATA_DIR / "american_put_ood_extreme_moneyness.parquet",
    "long_maturity": DATA_DIR / "american_put_ood_long_maturity.parquet",
    "rate_dividend": DATA_DIR / "american_put_ood_rate_dividend.parquet",
}

SEED = 42
set_global_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

```python
required_paths = [*CORE_PATHS, *OOD_PATHS.values(), MANIFEST_PATH]
missing_paths = [path for path in required_paths if not path.exists()]
if missing_paths:
    missing_text = "\n".join(f"- {path}" for path in missing_paths)
    raise FileNotFoundError(
        "Production files are missing. Run "
        "`python scripts/generate_production_dataset.py` first.\n"
        f"{missing_text}"
    )

manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
assert manifest["observed_total_observations"] == 1_450_000
manifest["components"]
```

## 3. Frozen data splits

The core and boundary components contain deterministic class-aware split labels. The split is assigned from immutable sample identifiers and the exercise class using a fixed 64-bit hash. This avoids loading all 1.25 million in-domain observations merely to allocate them, while preserving the exercise-class proportions in expectation.

The feature scaler is fitted exclusively on the training set.

```python
MODEL_COLUMNS = [
    "sample_id",
    "component",
    "split",
    *FEATURE_COLUMNS,
    DIRECT_TARGET_COLUMN,
    "normalized_european_price",
    "strike",
    "intrinsic_value",
    "european_price",
    "moneyness",
    "exercise_now",
    "boundary_distance_normalized",
]

train_frame = read_parquet_components(
    CORE_PATHS,
    columns=MODEL_COLUMNS,
    split="train",
)
validation_frame = read_parquet_components(
    CORE_PATHS,
    columns=MODEL_COLUMNS,
    split="validation",
)
test_frame = read_parquet_components(
    CORE_PATHS,
    columns=MODEL_COLUMNS,
    split="test",
)

split_summary = pd.DataFrame(
    {
        "observations": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        "exercise_rate": {
            "train": train_frame["exercise_now"].mean(),
            "validation": validation_frame["exercise_now"].mean(),
            "test": test_frame["exercise_now"].mean(),
        },
    }
)
split_summary
```

The observed shares should be close to 70%, 15%, and 15%. Exact equality is not necessary because assignment is hash-based rather than based on a global in-memory permutation. What matters is that the split is deterministic, disjoint, and stable across every model notebook.

## 4. Black–Scholes proxy baseline

Before training, the European normalized price is evaluated as a proxy for the American target. Its error is exactly the normalized early-exercise premium. This is the economically relevant baseline: a neural model that does not improve materially on it adds little value.

```python
baseline_metrics = regression_metrics(
    test_frame[DIRECT_TARGET_COLUMN],
    test_frame["normalized_european_price"],
)
pd.Series(baseline_metrics, name="Black–Scholes proxy").to_frame()
```

## 5. Training-only feature scaling and tensor datasets

The five inputs have different units and numerical scales. Standardization improves optimization but is fitted only on training observations to prevent leakage.

```python
feature_scaler = fit_feature_scaler(
    train_frame,
    feature_columns=FEATURE_COLUMNS,
)
SCALER_PATH = save_feature_scaler(
    feature_scaler,
    ARTIFACT_DIR / "feature_scaler.joblib",
)

train_dataset = AmericanOptionDataset(train_frame, scaler=feature_scaler)
validation_dataset = AmericanOptionDataset(validation_frame, scaler=feature_scaler)
test_dataset = AmericanOptionDataset(test_frame, scaler=feature_scaler)

LOADER_CONFIG = LoaderConfig(
    batch_size=1024,
    num_workers=0,
    pin_memory=True,
    seed=SEED,
)

train_loader = create_regression_loader(
    train_dataset,
    config=LOADER_CONFIG,
    shuffle=True,
    drop_last=True,
)
validation_loader = create_regression_loader(
    validation_dataset,
    config=LOADER_CONFIG,
    shuffle=False,
)
test_loader = create_regression_loader(
    test_dataset,
    config=LOADER_CONFIG,
    shuffle=False,
)

first_batch = next(iter(train_loader))
{
    "feature_batch_shape": tuple(first_batch["features"].shape),
    "target_batch_shape": tuple(first_batch["target"].shape),
    "feature_dtype": str(first_batch["features"].dtype),
    "target_dtype": str(first_batch["target"].dtype),
}
```

## 6. Direct MLP architecture

The architecture is deliberately strong but conventional:

```text
Input(5)
→ Linear(128) → BatchNorm → SiLU
→ Linear(128) → BatchNorm → SiLU
→ Linear(64)  → SiLU
→ Linear(32)  → SiLU
→ Linear(1)   → Softplus
```

`Softplus` prevents negative option values. It does not enforce the intrinsic-value or European-value lower bounds; those violations remain part of the baseline evaluation.

```python
MODEL_CONFIG = DirectMLPConfig(
    input_features=len(FEATURE_COLUMNS),
    hidden_sizes=(128, 128, 64, 32),
    batch_norm_after=(0, 1),
    output_activation="softplus",
)
model = DirectAmericanPutMLP(MODEL_CONFIG).to(DEVICE)

print(f"Trainable parameters: {model.trainable_parameter_count:,}")
summary(model, input_size=(LOADER_CONFIG.batch_size, len(FEATURE_COLUMNS)))
```

## 7. Training protocol

Smooth L1 loss is used as the primary objective because the pricing domain includes near-zero values, deep-in-the-money contracts, and potentially larger errors around the stopping boundary. AdamW, validation-based learning-rate reduction, early stopping, gradient clipping, and mixed precision on CUDA are applied consistently.

```python
TRAINING_CONFIG = TrainingConfig(
    epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-5,
    early_stopping_patience=10,
    scheduler_patience=4,
    scheduler_factor=0.5,
    min_learning_rate=1e-6,
    gradient_clip_norm=1.0,
    mixed_precision=True,
    seed=SEED,
)

CHECKPOINT_PATH = ARTIFACT_DIR / "best_direct_mlp.pt"
loss_fn = nn.SmoothL1Loss()

history = fit_regression_model(
    model,
    train_loader,
    validation_loader,
    loss_fn=loss_fn,
    device=DEVICE,
    checkpoint_path=CHECKPOINT_PATH,
    config=TRAINING_CONFIG,
    model_config=MODEL_CONFIG.to_dict(),
)

history.to_csv(ARTIFACT_DIR / "training_history.csv", index=False)
save_json(MODEL_CONFIG.to_dict(), ARTIFACT_DIR / "model_config.json")
history.tail()
```

## 8. Learning curves

Training and validation curves are reviewed together. The selected checkpoint is the epoch with the lowest validation loss, not the final epoch.

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(history["epoch"], history["train_loss"], label="Train")
axes[0].plot(history["epoch"], history["validation_loss"], label="Validation")
axes[0].set_title("Smooth L1 Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(history["epoch"], history["train_mae"], label="Train")
axes[1].plot(history["epoch"], history["validation_mae"], label="Validation")
axes[1].set_title("Normalized MAE")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(alpha=0.25)
plt.tight_layout()
plt.show()
```

## 9. In-domain test performance

The best checkpoint is reloaded before test evaluation. The test set has not influenced scaling, optimization, checkpoint selection, or learning-rate decisions.

```python
checkpoint = load_checkpoint(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_predictions = predict_regression_model(
    model,
    test_loader,
    device=DEVICE,
)
test_evaluation = test_frame.merge(
    test_predictions,
    on="sample_id",
    validate="one_to_one",
)

comparison = compare_models(
    test_evaluation[DIRECT_TARGET_COLUMN],
    {
        "Black–Scholes proxy": test_evaluation["normalized_european_price"],
        "Direct MLP": test_evaluation["prediction"],
    },
)
comparison
```

```python
evaluation_summary = {
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "best_validation_loss": float(checkpoint["best_validation_loss"]),
    "black_scholes_proxy": baseline_metrics,
    "direct_mlp": regression_metrics(
        test_evaluation[DIRECT_TARGET_COLUMN],
        test_evaluation["prediction"],
    ),
}
save_json(evaluation_summary, ARTIFACT_DIR / "evaluation_summary.json")
```

## 10. Segmented error analysis

Aggregate accuracy can conceal concentrated weaknesses. Errors are therefore evaluated by moneyness, maturity, volatility, exercise status, and distance from the stopping boundary.

```python
test_evaluation["moneyness_bucket"] = pd.cut(
    test_evaluation["moneyness"],
    bins=[0.0, 0.75, 0.90, 1.10, 1.25, np.inf],
    labels=["deep ITM", "ITM", "near ATM", "OTM", "deep OTM"],
)
test_evaluation["maturity_bucket"] = pd.cut(
    test_evaluation["time_to_maturity"],
    bins=[0.0, 0.25, 0.75, 1.25, np.inf],
    labels=["short", "medium", "long", "very long"],
)
test_evaluation["volatility_bucket"] = pd.cut(
    test_evaluation["volatility"],
    bins=[0.0, 0.20, 0.40, 0.60, np.inf],
    labels=["low", "moderate", "high", "very high"],
)
test_evaluation["exercise_region"] = np.where(
    test_evaluation["exercise_now"], "exercise", "continue"
)

segmented_regression_metrics(
    test_evaluation,
    actual_column=DIRECT_TARGET_COLUMN,
    prediction_column="prediction",
    segment_column="moneyness_bucket",
)
```

## 11. Financial consistency

The `Softplus` output prevents negative values, but it does not guarantee that the direct model remains above intrinsic value or above the corresponding European price. These violations are measured rather than repaired after prediction.

```python
financial_report = financial_bound_report(
    test_evaluation,
    normalized_prediction_column="prediction",
)
financial_report
```

## 12. Out-of-domain evaluation

The four OOD datasets are never included in training, validation, or in-domain test allocation. Their purpose is to test the predefined expectation that neural interpolation will deteriorate when the parameter domain shifts.

```python
ood_rows = []
for name, path in OOD_PATHS.items():
    frame = read_parquet_components(
        [path],
        columns=MODEL_COLUMNS,
        split=None,
    )
    dataset = AmericanOptionDataset(frame, scaler=feature_scaler)
    loader = create_regression_loader(
        dataset,
        config=LOADER_CONFIG,
        shuffle=False,
    )
    prediction = predict_regression_model(model, loader, device=DEVICE)
    evaluated = frame.merge(prediction, on="sample_id", validate="one_to_one")
    ood_rows.append(
        {
            "ood_set": name,
            **regression_metrics(
                evaluated[DIRECT_TARGET_COLUMN],
                evaluated["prediction"],
            ),
        }
    )

ood_results = pd.DataFrame(ood_rows).set_index("ood_set")
ood_results
```

## 13. Neural inference benchmark

The benchmark measures only the post-training forward-pass cost. Data generation and model training are reported separately and are not hidden in the inference comparison.

```python
@torch.inference_mode()
def benchmark_inference(model, features, device, repeats=5):
    model.eval()
    features = features.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        _ = model(features)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
    return {
        "observations": len(features),
        "median_seconds": float(np.median(timings)),
        "observations_per_second": float(len(features) / np.median(timings)),
    }

benchmark_rows = []
for size in (1_000, 10_000, 100_000):
    source = test_dataset.features[: min(size, len(test_dataset))]
    benchmark_rows.append(benchmark_inference(model, source, DEVICE))

pd.DataFrame(benchmark_rows)
```

## 14. Interim hypothesis assessment

Notebook 04 provides the first formal evidence for **H1**. The hypothesis is supported only if the direct MLP materially improves on the Black–Scholes proxy across the full in-domain test set and not merely in one narrow segment.

The speed component of **H5** remains preliminary until the later models are benchmarked under identical conditions.

The expected financial violations and OOD deterioration documented here motivate the early-exercise-premium architecture developed in Notebook 05.

## 15. Reproducibility checkpoint

The following artifacts are produced:

```text
artifacts/direct_mlp/
├── best_direct_mlp.pt
├── feature_scaler.joblib
├── training_history.csv
├── model_config.json
└── evaluation_summary.json
```

They remain excluded from Git. The code, configuration, data-generation manifest, and notebook preserve the experiment definition.
