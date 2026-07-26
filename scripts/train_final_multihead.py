#!/usr/bin/env python
"""Train the final integrated static multi-head American put model.

The script is the non-interactive production counterpart of Notebook 08. It
uses the frozen production splits, compares one selected loss preset, supports a
Step 6 warm start, and writes auditable model artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.multihead_targets import add_integrated_targets
from src.data.production_generation import (
    CORE_RANGES,
    ProductionDatasetConfig,
    build_priced_frame,
    sample_parameter_chunk,
)
from src.data.torch_datasets import (
    FEATURE_COLUMNS,
    IntegratedMultiHeadDataset,
    LoaderConfig,
    create_integrated_multihead_loader,
    fit_feature_scaler,
    read_parquet_components,
    save_feature_scaler,
)
from src.evaluation.integrated_model_comparison import (
    evaluate_integrated_prediction_frame,
)
from src.models.integrated_multihead_pricer import (
    IntegratedAmericanPutMultiHeadMLP,
    IntegratedMultiHeadConfig,
    copy_compatible_backbone_weights,
)
from src.training.multihead_losses import (
    IntegratedMultiHeadLoss,
    multihead_loss_preset,
)
from src.training.multihead_loops import (
    IntegratedTrainingConfig,
    fit_integrated_multihead_model,
    predict_integrated_multihead_model,
)
from src.training.multitask_losses import calculate_positive_class_weight


RAW_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "split",
    *FEATURE_COLUMNS,
    "strike",
    "intrinsic_value",
    "continuation_value",
    "european_price",
    "american_price",
    "exercise_now",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=("balanced", "pricing_focused", "decision_focused"),
        default="balanced",
        help="Predefined multi-objective loss configuration.",
    )
    parser.add_argument(
        "--architecture",
        choices=("large", "step6_compatible", "auto"),
        default="auto",
        help="Use the larger scratch model or the Step 6-compatible backbone.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "generated",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "final_multihead",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--warm-start-checkpoint",
        type=Path,
        default=None,
        help="Optional Step 6 checkpoint containing model_state_dict.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use small row limits and three epochs; generate fallback data if needed.",
    )
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def _production_paths(data_dir: Path) -> list[Path]:
    candidates = [
        data_dir / "american_put_core.parquet",
        data_dir / "american_put_boundary.parquet",
    ]
    return [path for path in candidates if path.exists()]


def _fallback_smoke_frame(*, rows: int, seed: int) -> pd.DataFrame:
    config = ProductionDatasetConfig(
        core_observations=max(rows, 1),
        boundary_observations=max(rows // 4, 1),
        ood_observations_per_set=max(rows // 20, 1),
        tree_steps=50,
        chunk_size=max(rows, 1),
        seed=seed,
    )
    parameters = sample_parameter_chunk(
        n_samples=rows,
        ranges=CORE_RANGES,
        seed=seed,
        strike=config.strike,
    )
    frame = build_priced_frame(
        parameters=parameters,
        sample_ids=np.arange(rows, dtype=np.int64),
        component="smoke",
        tree_steps=config.tree_steps,
        split_eligible=True,
        config=config,
    )
    return frame


def _load_splits(
    *,
    data_dir: Path,
    smoke: bool,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = _production_paths(data_dir)
    limits = {
        "train": 8_000 if smoke else None,
        "validation": 2_000 if smoke else None,
        "test": 2_000 if smoke else None,
    }
    if paths:
        frames = {
            split: read_parquet_components(
                paths,
                columns=RAW_COLUMNS,
                split=split,
                row_limit=limit,
            )
            for split, limit in limits.items()
        }
    elif smoke:
        fallback = _fallback_smoke_frame(rows=12_000, seed=seed)
        frames = {
            split: fallback.loc[fallback["split"] == split].copy()
            for split in ("train", "validation", "test")
        }
    else:
        raise FileNotFoundError(
            "Production Parquet files were not found. Run "
            "scripts/generate_production_dataset.py first or use --smoke."
        )

    prepared = tuple(
        add_integrated_targets(frames[split], copy=False)
        for split in ("train", "validation", "test")
    )
    if any(frame.empty for frame in prepared):
        raise RuntimeError("At least one required split is empty.")
    return prepared  # type: ignore[return-value]


def _architecture(
    name: str,
    *,
    warm_start_checkpoint: Path | None,
) -> IntegratedMultiHeadConfig:
    resolved = name
    if name == "auto":
        resolved = "step6_compatible" if warm_start_checkpoint else "large"
    if resolved == "step6_compatible":
        return IntegratedMultiHeadConfig.step6_compatible()
    return IntegratedMultiHeadConfig()



def _save_frame(frame: pd.DataFrame, path: Path) -> Path:
    """Save Parquet when available and fall back to CSV for smoke environments."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return path
    except ImportError:
        fallback = path.with_suffix(".csv")
        frame.to_csv(fallback, index=False)
        return fallback


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _model_card(
    *,
    config_name: str,
    architecture: IntegratedMultiHeadConfig,
    metrics: dict[str, float],
    device: torch.device,
) -> str:
    return f"""# Final Integrated Static Multi-Head Model

## Intended use

Static surrogate pricing and exercise analysis for American put options inside
the parameter domain documented by this project.

## Authoritative output

`predicted_normalized_american_price`, reconstructed as the financial floor plus
a non-negative residual.

## Auxiliary heads

- direct normalized American price;
- normalized continuation value;
- exercise probability.

## Configuration

- Loss preset: `{config_name}`
- Shared layers: `{architecture.shared_hidden_sizes}`
- Training device: `{device}`

## Validation principles

The model must be evaluated on frozen in-domain and out-of-domain sets. The
constrained price is guaranteed not to fall below the supplied European or
intrinsic values, but other financial shape properties remain empirical tests.

## Test metrics

```json
{json.dumps(metrics, indent=2)}
```

## Limitations

The model learns CRR-generated labels under constant volatility, continuous
dividend yield, and a controlled synthetic parameter domain. It is not a market
quote predictor and does not replace independent model validation.
"""


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    artifact_dir: Path = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_frame, validation_frame, test_frame = _load_splits(
        data_dir=args.data_dir,
        smoke=args.smoke,
        seed=args.seed,
    )
    scaler = fit_feature_scaler(train_frame, feature_columns=FEATURE_COLUMNS)
    save_feature_scaler(scaler, artifact_dir / "feature_scaler.joblib")

    loader_config = LoaderConfig(
        batch_size=args.batch_size if not args.smoke else min(args.batch_size, 512),
        num_workers=args.num_workers,
        pin_memory=True,
        seed=args.seed,
    )
    datasets = {
        "train": IntegratedMultiHeadDataset(train_frame, scaler=scaler),
        "validation": IntegratedMultiHeadDataset(validation_frame, scaler=scaler),
        "test": IntegratedMultiHeadDataset(test_frame, scaler=scaler),
    }
    loaders = {
        "train": create_integrated_multihead_loader(
            datasets["train"],
            config=loader_config,
            shuffle=True,
            drop_last=len(datasets["train"]) > loader_config.batch_size,
        ),
        "validation": create_integrated_multihead_loader(
            datasets["validation"],
            config=loader_config,
            shuffle=False,
        ),
        "test": create_integrated_multihead_loader(
            datasets["test"],
            config=loader_config,
            shuffle=False,
        ),
    }

    model_config = _architecture(
        args.architecture,
        warm_start_checkpoint=args.warm_start_checkpoint,
    )
    model = IntegratedAmericanPutMultiHeadMLP(model_config)
    warm_start_report: dict[str, object] = {}
    if args.warm_start_checkpoint is not None:
        checkpoint = torch.load(
            args.warm_start_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        source_state = checkpoint.get("model_state_dict", checkpoint)
        warm_start_report = copy_compatible_backbone_weights(
            model,
            source_state,
        )
        if int(warm_start_report["copied_count"]) == 0:
            raise RuntimeError(
                "Warm-start checkpoint had no shape-compatible backbone weights. "
                "Use --architecture step6_compatible."
            )

    positive_weight = calculate_positive_class_weight(
        train_frame["exercise_now"],
        maximum_weight=50.0,
    )
    loss_config = multihead_loss_preset(args.config)
    loss_fn = IntegratedMultiHeadLoss(
        config=loss_config,
        positive_class_weight=positive_weight,
    )
    training_config = IntegratedTrainingConfig(
        epochs=3 if args.smoke else args.epochs,
        early_stopping_patience=3 if args.smoke else 12,
        scheduler_patience=2 if args.smoke else 4,
        seed=args.seed,
        mixed_precision=device.type == "cuda",
    )
    checkpoint_path = artifact_dir / "best_integrated_multihead.pt"
    history = fit_integrated_multihead_model(
        model,
        loaders["train"],
        loaders["validation"],
        loss_fn=loss_fn,
        config=training_config,
        device=device,
        checkpoint_path=checkpoint_path,
        model_config=model_config.to_dict(),
        warm_start_report=warm_start_report,
    )
    _save_frame(history, artifact_dir / "training_history.parquet")

    validation_predictions = predict_integrated_multihead_model(
        model,
        loaders["validation"],
        device=device,
        decision_sharpness=loss_config.decision_sharpness,
    )
    test_predictions = predict_integrated_multihead_model(
        model,
        loaders["test"],
        device=device,
        decision_sharpness=loss_config.decision_sharpness,
    )
    _save_frame(
        validation_predictions,
        artifact_dir / "validation_predictions.parquet",
    )
    _save_frame(
        test_predictions,
        artifact_dir / "test_predictions.parquet",
    )
    validation_metrics = evaluate_integrated_prediction_frame(validation_predictions)
    test_metrics = evaluate_integrated_prediction_frame(test_predictions)

    _write_json(artifact_dir / "model_config.json", model_config.to_dict())
    _write_json(artifact_dir / "loss_config.json", loss_config.to_dict())
    _write_json(artifact_dir / "validation_metrics.json", validation_metrics)
    _write_json(artifact_dir / "test_metrics.json", test_metrics)
    _write_json(artifact_dir / "warm_start_report.json", warm_start_report)
    (artifact_dir / "model_card.md").write_text(
        _model_card(
            config_name=args.config,
            architecture=model_config,
            metrics=test_metrics,
            device=device,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "device": str(device),
                "configuration": args.config,
                "train_rows": len(train_frame),
                "validation_rows": len(validation_frame),
                "test_rows": len(test_frame),
                "test_constrained_rmse": test_metrics["constrained_rmse"],
                "test_exercise_f1": test_metrics["exercise_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
