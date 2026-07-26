"""Scalable production-data generation for American put-option pricing.

The module is designed for the 1.45 million-observation production run:

- 1,000,000 core-domain observations;
- 250,000 boundary-focused observations;
- four 50,000-observation out-of-domain sets.

Pricing is accelerated with Numba and written incrementally to Parquet so the
full dataset does not need to be held in memory. The generated price remains
fully auditable: raw CRR value, analytical European value, intrinsic value,
continuation value, and any no-arbitrage floor adjustment are stored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.special import ndtr


ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class RangeSpec:
    """Continuous sampling bounds for one generated dataset component."""

    moneyness: tuple[float, float]
    time_to_maturity: tuple[float, float]
    volatility: tuple[float, float]
    risk_free_rate: tuple[float, float]
    dividend_yield: tuple[float, float]

    def __post_init__(self) -> None:
        for name, bounds in asdict(self).items():
            lower, upper = bounds
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"{name} bounds must be finite.")
            if lower >= upper:
                raise ValueError(f"{name} lower bound must be below upper bound.")
        if self.moneyness[0] <= 0.0:
            raise ValueError("Moneyness must remain positive.")
        if self.time_to_maturity[0] <= 0.0:
            raise ValueError("Time to maturity must remain positive.")
        if self.volatility[0] <= 0.0:
            raise ValueError("Volatility must remain positive.")


CORE_RANGES = RangeSpec(
    moneyness=(0.50, 1.50),
    time_to_maturity=(7.0 / 365.0, 2.0),
    volatility=(0.05, 0.80),
    risk_free_rate=(0.00, 0.10),
    dividend_yield=(0.00, 0.08),
)

BOUNDARY_CANDIDATE_RANGES = RangeSpec(
    moneyness=(0.45, 1.10),
    time_to_maturity=(7.0 / 365.0, 1.50),
    volatility=(0.05, 0.60),
    risk_free_rate=(0.01, 0.15),
    dividend_yield=(0.00, 0.06),
)

HIGH_VOLATILITY_RANGES = RangeSpec(
    moneyness=CORE_RANGES.moneyness,
    time_to_maturity=CORE_RANGES.time_to_maturity,
    volatility=(0.80, 1.20),
    risk_free_rate=CORE_RANGES.risk_free_rate,
    dividend_yield=CORE_RANGES.dividend_yield,
)

LONG_MATURITY_RANGES = RangeSpec(
    moneyness=CORE_RANGES.moneyness,
    time_to_maturity=(2.00, 4.00),
    volatility=CORE_RANGES.volatility,
    risk_free_rate=CORE_RANGES.risk_free_rate,
    dividend_yield=CORE_RANGES.dividend_yield,
)

RATE_DIVIDEND_RANGES = RangeSpec(
    moneyness=CORE_RANGES.moneyness,
    time_to_maturity=CORE_RANGES.time_to_maturity,
    volatility=CORE_RANGES.volatility,
    risk_free_rate=(0.10, 0.20),
    dividend_yield=(0.08, 0.15),
)

DEEP_ITM_RANGES = RangeSpec(
    moneyness=(0.25, 0.50),
    time_to_maturity=CORE_RANGES.time_to_maturity,
    volatility=CORE_RANGES.volatility,
    risk_free_rate=CORE_RANGES.risk_free_rate,
    dividend_yield=CORE_RANGES.dividend_yield,
)

DEEP_OTM_RANGES = RangeSpec(
    moneyness=(1.50, 2.00),
    time_to_maturity=CORE_RANGES.time_to_maturity,
    volatility=CORE_RANGES.volatility,
    risk_free_rate=CORE_RANGES.risk_free_rate,
    dividend_yield=CORE_RANGES.dividend_yield,
)


@dataclass(frozen=True, slots=True)
class ProductionDatasetConfig:
    """Configuration for the complete production generation run."""

    core_observations: int = 1_000_000
    boundary_observations: int = 250_000
    ood_observations_per_set: int = 50_000
    tree_steps: int = 250
    strike: float = 100.0
    chunk_size: int = 25_000
    seed: int = 42
    boundary_candidate_multiplier: float = 2.0
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    compression: str = "zstd"
    compression_level: int = 3

    def __post_init__(self) -> None:
        integer_fields = (
            "core_observations",
            "boundary_observations",
            "ood_observations_per_set",
            "tree_steps",
            "chunk_size",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.strike <= 0.0:
            raise ValueError("strike must be positive.")
        if self.boundary_candidate_multiplier < 1.0:
            raise ValueError("boundary_candidate_multiplier must be at least 1.")
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(value <= 0.0 for value in fractions):
            raise ValueError("All split fractions must be positive.")
        if not np.isclose(sum(fractions), 1.0, atol=1e-12):
            raise ValueError("Split fractions must sum to 1.0.")

    @property
    def ood_set_count(self) -> int:
        return 4

    @property
    def in_domain_observations(self) -> int:
        return self.core_observations + self.boundary_observations

    @property
    def ood_observations(self) -> int:
        return self.ood_set_count * self.ood_observations_per_set

    @property
    def total_observations(self) -> int:
        return self.in_domain_observations + self.ood_observations


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """One production component and its fixed global identifier range."""

    name: str
    observations: int
    start_id: int
    ranges: RangeSpec
    purpose: str
    split_eligible: bool


@dataclass(slots=True)
class ComponentResult:
    """Generation metadata returned after one component is written."""

    name: str
    path: str
    observations: int
    start_id: int
    end_id: int
    split_counts: dict[str, int]
    exercise_count: int
    floor_adjustment_count: int
    max_floor_adjustment: float
    sha256: str


REQUIRED_OUTPUT_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "component",
    "split",
    "spot",
    "strike",
    "moneyness",
    "log_moneyness",
    "time_to_maturity",
    "risk_free_rate",
    "dividend_yield",
    "volatility",
    "intrinsic_value",
    "continuation_value",
    "european_price",
    "raw_american_price",
    "american_price",
    "pricing_floor_adjustment",
    "early_exercise_premium",
    "normalized_european_price",
    "normalized_american_price",
    "normalized_early_exercise_premium",
    "boundary_distance_normalized",
    "exercise_now",
    "tree_steps",
)


def build_component_specs(
    config: ProductionDatasetConfig,
) -> tuple[ComponentSpec, ...]:
    """Build the exact 1.45 million-observation component layout."""

    start = 0
    specs: list[ComponentSpec] = []

    def add(
        name: str,
        observations: int,
        ranges: RangeSpec,
        purpose: str,
        split_eligible: bool,
    ) -> None:
        nonlocal start
        specs.append(
            ComponentSpec(
                name=name,
                observations=observations,
                start_id=start,
                ranges=ranges,
                purpose=purpose,
                split_eligible=split_eligible,
            )
        )
        start += observations

    add(
        "core",
        config.core_observations,
        CORE_RANGES,
        "Primary interpolation domain.",
        True,
    )
    add(
        "boundary",
        config.boundary_observations,
        BOUNDARY_CANDIDATE_RANGES,
        "Exercise-boundary-focused in-domain sample.",
        True,
    )
    add(
        "ood_high_volatility",
        config.ood_observations_per_set,
        HIGH_VOLATILITY_RANGES,
        "Volatility above the core-domain maximum.",
        False,
    )
    add(
        "ood_extreme_moneyness",
        config.ood_observations_per_set,
        CORE_RANGES,
        "Balanced deep-ITM and deep-OTM contracts.",
        False,
    )
    add(
        "ood_long_maturity",
        config.ood_observations_per_set,
        LONG_MATURITY_RANGES,
        "Maturities above the core-domain maximum.",
        False,
    )
    add(
        "ood_rate_dividend",
        config.ood_observations_per_set,
        RATE_DIVIDEND_RANGES,
        "Jointly elevated risk-free and dividend rates.",
        False,
    )

    if start != config.total_observations:
        raise RuntimeError("Internal component design does not match total size.")

    return tuple(specs)


def _latin_hypercube_chunk(
    *,
    n_samples: int,
    n_dimensions: int,
    seed: int,
) -> np.ndarray:
    """Create a deterministic randomized Latin hypercube for one chunk."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    rng = np.random.default_rng(seed)
    output = np.empty((n_samples, n_dimensions), dtype=np.float64)
    for column in range(n_dimensions):
        points = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        output[:, column] = rng.permutation(points)
    return output


def _scale(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lower, upper = bounds
    return lower + values * (upper - lower)


def sample_parameter_chunk(
    *,
    n_samples: int,
    ranges: RangeSpec,
    seed: int,
    strike: float,
    extreme_moneyness: bool = False,
) -> dict[str, np.ndarray]:
    """Sample one parameter chunk with deterministic Latin hypercube coverage."""

    unit = _latin_hypercube_chunk(
        n_samples=n_samples,
        n_dimensions=5,
        seed=seed,
    )

    if extreme_moneyness:
        half = n_samples // 2
        moneyness = np.empty(n_samples, dtype=np.float64)
        moneyness[:half] = _scale(unit[:half, 0], DEEP_ITM_RANGES.moneyness)
        moneyness[half:] = _scale(unit[half:, 0], DEEP_OTM_RANGES.moneyness)
        rng = np.random.default_rng(seed + 81_919)
        moneyness = rng.permutation(moneyness)
    else:
        moneyness = _scale(unit[:, 0], ranges.moneyness)

    maturity = _scale(unit[:, 1], ranges.time_to_maturity)
    volatility = _scale(unit[:, 2], ranges.volatility)
    rate = _scale(unit[:, 3], ranges.risk_free_rate)
    dividend = _scale(unit[:, 4], ranges.dividend_yield)
    spot = strike * moneyness

    return {
        "spot": spot,
        "strike": np.full(n_samples, strike, dtype=np.float64),
        "moneyness": moneyness,
        "log_moneyness": np.log(moneyness),
        "time_to_maturity": maturity,
        "risk_free_rate": rate,
        "dividend_yield": dividend,
        "volatility": volatility,
    }


@njit(cache=True)
def _normal_cdf_numba(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


@njit(cache=True)
def _black_scholes_put_numba(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
) -> float:
    if maturity <= 0.0:
        return max(strike - spot, 0.0)
    if volatility <= 0.0:
        discounted_strike = strike * math.exp(-rate * maturity)
        discounted_spot = spot * math.exp(-dividend * maturity)
        return max(discounted_strike - discounted_spot, 0.0)

    root_t = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend + 0.5 * volatility * volatility) * maturity
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    return max(
        strike * math.exp(-rate * maturity) * _normal_cdf_numba(-d2)
        - spot * math.exp(-dividend * maturity) * _normal_cdf_numba(-d1),
        0.0,
    )


@njit(cache=True)
def _crr_american_put_one(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
    steps: int,
) -> tuple[float, float, float, bool]:
    """Return price, intrinsic, root continuation, and exercise decision."""

    intrinsic = max(strike - spot, 0.0)
    if maturity <= 0.0:
        return intrinsic, intrinsic, intrinsic, True

    if volatility <= 0.0:
        dt = maturity / steps
        best = intrinsic
        terminal_discounted = intrinsic
        for step in range(1, steps + 1):
            time = step * dt
            future_spot = spot * math.exp((rate - dividend) * time)
            payoff = max(strike - future_spot, 0.0)
            discounted = math.exp(-rate * time) * payoff
            terminal_discounted = discounted
            if discounted > best:
                best = discounted
        continuation = terminal_discounted if steps == 1 else best
        return best, intrinsic, continuation, intrinsic >= continuation

    dt = maturity / steps
    up = math.exp(volatility * math.sqrt(dt))
    down = 1.0 / up
    probability = (math.exp((rate - dividend) * dt) - down) / (up - down)
    if probability < 0.0:
        probability = 0.0
    elif probability > 1.0:
        probability = 1.0
    discount = math.exp(-rate * dt)

    values = np.empty(steps + 1, dtype=np.float64)
    terminal_spot = spot * (down ** steps)
    up_down_ratio = up / down
    for node in range(steps + 1):
        values[node] = max(strike - terminal_spot, 0.0)
        terminal_spot *= up_down_ratio

    root_continuation = 0.0
    for current_step in range(steps - 1, -1, -1):
        node_spot = spot * (down ** current_step)
        for node in range(current_step + 1):
            continuation = discount * (
                probability * values[node + 1]
                + (1.0 - probability) * values[node]
            )
            exercise = max(strike - node_spot, 0.0)
            values[node] = max(continuation, exercise)
            if current_step == 0:
                root_continuation = continuation
            node_spot *= up_down_ratio

    exercise_now = intrinsic >= root_continuation - 1e-12
    return values[0], intrinsic, root_continuation, exercise_now


@njit(parallel=True, cache=True)
def price_american_put_batch(
    spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    rate: np.ndarray,
    dividend: np.ndarray,
    volatility: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, ...]:
    """Price one homogeneous batch in parallel with Numba."""

    size = len(spot)
    european = np.empty(size, dtype=np.float64)
    raw_american = np.empty(size, dtype=np.float64)
    intrinsic = np.empty(size, dtype=np.float64)
    continuation = np.empty(size, dtype=np.float64)
    exercise_now = np.empty(size, dtype=np.bool_)

    for index in prange(size):
        european[index] = _black_scholes_put_numba(
            spot[index],
            strike[index],
            maturity[index],
            rate[index],
            dividend[index],
            volatility[index],
        )
        (
            raw_american[index],
            intrinsic[index],
            continuation[index],
            exercise_now[index],
        ) = _crr_american_put_one(
            spot[index],
            strike[index],
            maturity[index],
            rate[index],
            dividend[index],
            volatility[index],
            steps,
        )

    return european, raw_american, intrinsic, continuation, exercise_now


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Deterministic 64-bit mixing used for stable split assignment."""

    x = values.astype(np.uint64, copy=True)
    x += np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def deterministic_split_labels(
    sample_ids: np.ndarray,
    exercise_now: np.ndarray,
    *,
    config: ProductionDatasetConfig,
) -> np.ndarray:
    """Assign stable class-aware train, validation, and test labels.

    Exercise classes receive different salts, preserving split proportions in
    expectation without loading the full 1.25 million-row in-domain dataset.
    """

    salted = sample_ids.astype(np.uint64) ^ np.where(
        exercise_now,
        np.uint64(config.seed + 0xA5A5A5A5),
        np.uint64(config.seed + 0x5A5A5A5A),
    )
    mixed = _splitmix64(salted)
    uniform = mixed.astype(np.float64) / np.float64(2**64 - 1)
    train_cutoff = config.train_fraction
    validation_cutoff = train_cutoff + config.validation_fraction
    return np.where(
        uniform < train_cutoff,
        "train",
        np.where(uniform < validation_cutoff, "validation", "test"),
    )


def build_priced_frame(
    *,
    parameters: Mapping[str, np.ndarray],
    sample_ids: np.ndarray,
    component: str,
    tree_steps: int,
    split_eligible: bool,
    config: ProductionDatasetConfig,
) -> pd.DataFrame:
    """Price one parameter batch and return the canonical output schema."""

    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()}
    european, raw_american, intrinsic, continuation, exercise_now = price_american_put_batch(
        arrays["spot"],
        arrays["strike"],
        arrays["time_to_maturity"],
        arrays["risk_free_rate"],
        arrays["dividend_yield"],
        arrays["volatility"],
        int(tree_steps),
    )

    american = np.maximum.reduce((raw_american, european, intrinsic))
    floor_adjustment = american - raw_american
    premium = american - european
    boundary_distance = np.abs(intrinsic - continuation) / arrays["strike"]
    split = (
        deterministic_split_labels(
            sample_ids,
            exercise_now,
            config=config,
        )
        if split_eligible
        else np.full(len(sample_ids), "ood", dtype="<U10")
    )

    frame = pd.DataFrame(
        {
            "sample_id": sample_ids.astype(np.int64),
            "component": component,
            "split": split,
            **arrays,
            "intrinsic_value": intrinsic,
            "continuation_value": continuation,
            "european_price": european,
            "raw_american_price": raw_american,
            "american_price": american,
            "pricing_floor_adjustment": floor_adjustment,
            "early_exercise_premium": premium,
            "normalized_european_price": european / arrays["strike"],
            "normalized_american_price": american / arrays["strike"],
            "normalized_early_exercise_premium": premium / arrays["strike"],
            "boundary_distance_normalized": boundary_distance,
            "exercise_now": exercise_now,
            "tree_steps": np.full(len(sample_ids), tree_steps, dtype=np.int32),
        }
    )
    return frame.loc[:, REQUIRED_OUTPUT_COLUMNS]


def validate_generated_frame(frame: pd.DataFrame) -> None:
    """Fail fast when one generated chunk violates the production schema."""

    missing = [column for column in REQUIRED_OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Generated frame is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Generated frame cannot be empty.")
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Generated frame contains NaN or infinite numeric values.")
    tolerance = 1e-10
    if (frame["american_price"] < frame["intrinsic_value"] - tolerance).any():
        raise ValueError("American-price intrinsic-value bound failed.")
    if (frame["american_price"] < frame["european_price"] - tolerance).any():
        raise ValueError("American-price European-value bound failed.")
    if (frame["early_exercise_premium"] < -tolerance).any():
        raise ValueError("Negative early-exercise premium detected.")
    identity = frame["american_price"] - frame["european_price"]
    if not np.allclose(identity, frame["early_exercise_premium"], atol=1e-10):
        raise ValueError("Early-exercise-premium identity failed.")


def _require_pyarrow() -> tuple[object, object]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "Production Parquet generation requires pyarrow. "
            "Install the repository requirements before running the script."
        ) from error
    return pa, pq


def _sha256(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _write_frames_to_parquet(
    frames: Iterable[pd.DataFrame],
    *,
    output_path: Path,
    config: ProductionDatasetConfig,
) -> None:
    pa, pq = _require_pyarrow()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()

    writer = None
    try:
        for frame in frames:
            validate_generated_frame(frame)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression=config.compression,
                    compression_level=config.compression_level,
                    use_dictionary=["component", "split"],
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if not temporary.exists():
        raise RuntimeError("No Parquet output was written.")
    temporary.replace(output_path)


def _component_frames(
    *,
    spec: ComponentSpec,
    config: ProductionDatasetConfig,
    progress_callback: ProgressCallback | None,
) -> Iterable[pd.DataFrame]:
    generated = 0
    chunk_index = 0
    while generated < spec.observations:
        size = min(config.chunk_size, spec.observations - generated)
        chunk_seed = config.seed + spec.start_id + chunk_index * 104_729
        parameters = sample_parameter_chunk(
            n_samples=size,
            ranges=spec.ranges,
            seed=chunk_seed,
            strike=config.strike,
            extreme_moneyness=spec.name == "ood_extreme_moneyness",
        )
        sample_ids = np.arange(
            spec.start_id + generated,
            spec.start_id + generated + size,
            dtype=np.int64,
        )
        frame = build_priced_frame(
            parameters=parameters,
            sample_ids=sample_ids,
            component=spec.name,
            tree_steps=config.tree_steps,
            split_eligible=spec.split_eligible,
            config=config,
        )
        generated += size
        chunk_index += 1
        if progress_callback is not None:
            progress_callback(spec.name, generated, spec.observations)
        yield frame


def _boundary_frame(
    *,
    spec: ComponentSpec,
    config: ProductionDatasetConfig,
    progress_callback: ProgressCallback | None,
) -> pd.DataFrame:
    """Generate candidates and retain observations closest to the root boundary."""

    candidate_count = int(math.ceil(spec.observations * config.boundary_candidate_multiplier))
    candidate_frames: list[pd.DataFrame] = []
    generated = 0
    chunk_index = 0

    while generated < candidate_count:
        size = min(config.chunk_size, candidate_count - generated)
        seed = config.seed + 31_337 + chunk_index * 104_729
        parameters = sample_parameter_chunk(
            n_samples=size,
            ranges=spec.ranges,
            seed=seed,
            strike=config.strike,
        )
        temporary_ids = np.arange(generated, generated + size, dtype=np.int64)
        frame = build_priced_frame(
            parameters=parameters,
            sample_ids=temporary_ids,
            component=spec.name,
            tree_steps=config.tree_steps,
            split_eligible=False,
            config=config,
        )
        candidate_frames.append(frame)
        generated += size
        chunk_index += 1
        if progress_callback is not None:
            progress_callback("boundary_candidates", generated, candidate_count)

    candidates = pd.concat(candidate_frames, ignore_index=True)
    selected = candidates.nsmallest(
        spec.observations,
        "boundary_distance_normalized",
        keep="first",
    ).reset_index(drop=True)
    selected["sample_id"] = np.arange(
        spec.start_id,
        spec.start_id + spec.observations,
        dtype=np.int64,
    )
    selected["split"] = deterministic_split_labels(
        selected["sample_id"].to_numpy(dtype=np.int64),
        selected["exercise_now"].to_numpy(dtype=bool),
        config=config,
    )
    validate_generated_frame(selected)
    return selected


def generate_component(
    *,
    spec: ComponentSpec,
    output_dir: str | Path,
    config: ProductionDatasetConfig,
    overwrite: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> ComponentResult:
    """Generate one complete component and return its audit metadata."""

    output_path = Path(output_dir) / f"american_put_{spec.name}.parquet"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use overwrite=True to replace it."
        )

    if spec.name == "boundary":
        boundary = _boundary_frame(
            spec=spec,
            config=config,
            progress_callback=progress_callback,
        )
        _write_frames_to_parquet(
            [boundary],
            output_path=output_path,
            config=config,
        )
    else:
        _write_frames_to_parquet(
            _component_frames(
                spec=spec,
                config=config,
                progress_callback=progress_callback,
            ),
            output_path=output_path,
            config=config,
        )

    summary_columns = [
        "sample_id",
        "split",
        "exercise_now",
        "pricing_floor_adjustment",
    ]
    summary = pd.read_parquet(output_path, columns=summary_columns)
    split_counts = {
        str(key): int(value)
        for key, value in summary["split"].value_counts().to_dict().items()
    }
    adjustment = summary["pricing_floor_adjustment"]

    return ComponentResult(
        name=spec.name,
        path=str(output_path),
        observations=int(len(summary)),
        start_id=int(summary["sample_id"].min()),
        end_id=int(summary["sample_id"].max()),
        split_counts=split_counts,
        exercise_count=int(summary["exercise_now"].sum()),
        floor_adjustment_count=int((adjustment > 1e-12).sum()),
        max_floor_adjustment=float(adjustment.max()),
        sha256=_sha256(output_path),
    )


def build_generation_manifest(
    *,
    config: ProductionDatasetConfig,
    results: Iterable[ComponentResult],
) -> dict[str, object]:
    """Build the canonical production-data manifest."""

    result_list = list(results)
    observed_total = sum(result.observations for result in result_list)
    if observed_total != config.total_observations:
        raise ValueError(
            f"Observed {observed_total:,} rows; expected "
            f"{config.total_observations:,}."
        )
    return {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "deep_learning_american_option_pricing",
        "generation_config": asdict(config),
        "expected_total_observations": config.total_observations,
        "observed_total_observations": observed_total,
        "components": [asdict(result) for result in result_list],
        "feature_columns": [
            "log_moneyness",
            "time_to_maturity",
            "risk_free_rate",
            "dividend_yield",
            "volatility",
        ],
        "direct_target": "normalized_american_price",
        "split_method": "deterministic class-aware SplitMix64 hashing",
        "pricing_floor": "max(raw_american_price, european_price, intrinsic_value)",
    }


def save_manifest(manifest: Mapping[str, object], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "BOUNDARY_CANDIDATE_RANGES",
    "CORE_RANGES",
    "ComponentResult",
    "ComponentSpec",
    "ProductionDatasetConfig",
    "REQUIRED_OUTPUT_COLUMNS",
    "RangeSpec",
    "build_component_specs",
    "build_generation_manifest",
    "build_priced_frame",
    "deterministic_split_labels",
    "generate_component",
    "price_american_put_batch",
    "sample_parameter_chunk",
    "save_manifest",
    "validate_generated_frame",
]
