"""Generate the complete 1.45 million-row American put dataset.

Run from the repository root:

    python scripts/generate_production_dataset.py

The script is component-restartable. Existing component files are skipped unless
``--overwrite`` is supplied. A generation manifest is written after all six
components are available and verified.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import pandas as pd

from src.data.production_generation import (
    ComponentResult,
    ProductionDatasetConfig,
    build_component_specs,
    build_generation_manifest,
    generate_component,
    save_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the production American put dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated"),
        help="Directory for generated Parquet components.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/production_dataset_manifest.json"),
        help="Output generation manifest.",
    )
    parser.add_argument("--tree-steps", type=int, default=250)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--component",
        action="append",
        default=None,
        help="Generate only named components. May be supplied repeatedly.",
    )
    return parser.parse_args()


def progress(name: str, completed: int, total: int) -> None:
    percentage = completed / total * 100.0
    print(
        f"\r{name:<28} {completed:>10,}/{total:<10,} "
        f"({percentage:6.2f}%)",
        end="",
        flush=True,
    )
    if completed >= total:
        print()


def load_existing_result(path: Path, name: str) -> ComponentResult:
    columns = [
        "sample_id",
        "split",
        "exercise_now",
        "pricing_floor_adjustment",
    ]
    frame = pd.read_parquet(path, columns=columns)
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(2**20):
            digest.update(block)

    adjustment = frame["pricing_floor_adjustment"]
    return ComponentResult(
        name=name,
        path=str(path),
        observations=int(len(frame)),
        start_id=int(frame["sample_id"].min()),
        end_id=int(frame["sample_id"].max()),
        split_counts={
            str(key): int(value)
            for key, value in frame["split"].value_counts().to_dict().items()
        },
        exercise_count=int(frame["exercise_now"].sum()),
        floor_adjustment_count=int((adjustment > 1e-12).sum()),
        max_floor_adjustment=float(adjustment.max()),
        sha256=digest.hexdigest(),
    )


def main() -> None:
    args = parse_args()
    config = ProductionDatasetConfig(
        tree_steps=args.tree_steps,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    specs = build_component_specs(config)
    selected = set(args.component or [spec.name for spec in specs])
    unknown = selected - {spec.name for spec in specs}
    if unknown:
        raise ValueError(f"Unknown component names: {sorted(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(asdict(config), indent=2))
    print(f"Expected total observations: {config.total_observations:,}")

    results: list[ComponentResult] = []
    started = time.perf_counter()

    for spec in specs:
        path = args.output_dir / f"american_put_{spec.name}.parquet"
        if spec.name not in selected:
            if path.exists():
                results.append(load_existing_result(path, spec.name))
            continue

        if path.exists() and not args.overwrite:
            print(f"Skipping existing component: {path}")
            result = load_existing_result(path, spec.name)
        else:
            print(f"Generating {spec.name}: {spec.observations:,} rows")
            result = generate_component(
                spec=spec,
                output_dir=args.output_dir,
                config=config,
                overwrite=args.overwrite,
                progress_callback=progress,
            )
        results.append(result)
        print(
            f"Completed {result.name}: {result.observations:,} rows, "
            f"exercise={result.exercise_count:,}, "
            f"repairs={result.floor_adjustment_count:,}"
        )

    result_names = {result.name for result in results}
    for spec in specs:
        path = args.output_dir / f"american_put_{spec.name}.parquet"
        if spec.name not in result_names and path.exists():
            results.append(load_existing_result(path, spec.name))

    if len(results) == len(specs):
        results.sort(key=lambda result: next(
            index for index, spec in enumerate(specs) if spec.name == result.name
        ))
        manifest = build_generation_manifest(config=config, results=results)
        save_manifest(manifest, args.manifest)
        print(f"Manifest: {args.manifest}")
    else:
        completed = {result.name for result in results}
        pending = [spec.name for spec in specs if spec.name not in completed]
        print(f"Partial run complete. Pending components: {pending}")
        print("The full production manifest will be written after all components exist.")

    elapsed = time.perf_counter() - started
    print(f"Total runtime: {elapsed / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
