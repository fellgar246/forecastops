"""Frozen benchmark-v1 scores for naive, seasonal naive, and gradient boosting.

The pin records the seed, row counts, and WAPE bands measured on that
dataset. A later run fails when a measured WAPE leaves its band.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from forecastops_ml.baselines.naive import NaiveForecaster, weekly_seasonal_naive
from forecastops_ml.data.synthetic import DEFAULT_MAX_ROWS, DatasetSummary, generate_dataset
from forecastops_ml.evaluation.backtest import Backtester, Forecaster
from forecastops_ml.features import aggregate_store_sku_week
from forecastops_ml.training.gradient_boosting import GradientBoostingForecaster

BENCHMARK_VERSION = "benchmark-v1"
_PIN_PATH = Path(__file__).resolve().parent / "fixtures" / "benchmark-v1.json"
_FAMILIES = ("naive", "seasonal_naive", "gradient_boosting")
_GRAINS = ("day", "week")


@dataclass(frozen=True)
class PinnedModel:
    """Expected WAPE for one family on benchmark-v1."""

    family: str
    grain: str
    wape: float


@dataclass(frozen=True)
class BenchmarkPin:
    """Seed, row counts, and metric bands checked in for benchmark-v1."""

    benchmark_version: str
    seed: int
    profile: str
    daily_row_count: int
    weekly_row_count: int
    folds: int
    horizon: int
    tolerance: float
    models: tuple[PinnedModel, ...]


@dataclass(frozen=True)
class MeasuredModel:
    """WAPE measured for one family on this run."""

    family: str
    wape: float


@dataclass(frozen=True)
class ModelScore:
    """Measured WAPE and whether it stayed inside the pinned band."""

    family: str
    wape: float
    passed: bool


def load_benchmark_pin(path: Path | None = None) -> BenchmarkPin:
    """Read the checked-in benchmark-v1 pin."""

    source = _PIN_PATH if path is None else path
    payload = json.loads(source.read_text(encoding="utf-8"))
    return _parse_pin(payload)


def measure_benchmark(pin: BenchmarkPin | None = None) -> tuple[MeasuredModel, ...]:
    """Generate the pinned dataset and score each pinned family once."""

    selected = load_benchmark_pin() if pin is None else pin
    if selected.benchmark_version != BENCHMARK_VERSION:
        raise ValueError(f"This runner scores {BENCHMARK_VERSION}.")
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        summary = generate_dataset(
            directory,
            profile=selected.profile,
            seed=selected.seed,
            max_rows=DEFAULT_MAX_ROWS,
        )
        _require_identity(summary, selected)
        daily = pq.read_table(directory / "observations.parquet")
        weekly = aggregate_store_sku_week(daily)
        if daily.num_rows != selected.daily_row_count:
            raise ValueError(
                f"{BENCHMARK_VERSION} daily row count is {selected.daily_row_count}, "
                f"but this run produced {daily.num_rows}."
            )
        if weekly.num_rows != selected.weekly_row_count:
            raise ValueError(
                f"{BENCHMARK_VERSION} weekly row count is {selected.weekly_row_count}, "
                f"but this run produced {weekly.num_rows}."
            )
        return _score(daily, weekly, selected)


def compare_measurements(
    measured: Sequence[MeasuredModel],
    pin: BenchmarkPin,
) -> tuple[ModelScore, ...]:
    """Compare measured WAPE values with the pinned bands.

    ``ModelScore.wape`` is the measured value. ``passed`` is true when that
    value stays within the pin's absolute tolerance of the checked-in WAPE.
    """

    by_family = {item.family: item.wape for item in measured}
    expected = {item.family for item in pin.models}
    if set(by_family) != expected:
        found = ", ".join(sorted(by_family))
        raise ValueError(f"Measured families ({found}) do not match the benchmark pin.")
    if pin.tolerance < 0 or not math.isfinite(pin.tolerance):
        raise ValueError("WAPE tolerance must be a non-negative finite number.")
    scores: list[ModelScore] = []
    for pinned in pin.models:
        wape = by_family[pinned.family]
        passed = math.isfinite(wape) and abs(wape - pinned.wape) <= pin.tolerance
        scores.append(ModelScore(family=pinned.family, wape=wape, passed=passed))
    return tuple(scores)


def _score(daily: pa.Table, weekly: pa.Table, pin: BenchmarkPin) -> tuple[MeasuredModel, ...]:
    frames = {"day": daily, "week": weekly}
    backtester = Backtester()
    measured: list[MeasuredModel] = []
    for pinned in pin.models:
        report = backtester.evaluate(
            _forecaster(pinned.family, pin.seed),
            frames[pinned.grain],
            pin.folds,
            pin.horizon,
            dataset_version=pin.benchmark_version,
        )
        if report.model_family != pinned.family:
            raise ValueError(f"Expected family {pinned.family}, scored {report.model_family}.")
        measured.append(MeasuredModel(family=pinned.family, wape=report.global_metrics.wape))
    return tuple(measured)


def _forecaster(family: str, seed: int) -> Forecaster:
    if family == "naive":
        return NaiveForecaster()
    if family == "seasonal_naive":
        return weekly_seasonal_naive()
    if family == "gradient_boosting":
        return GradientBoostingForecaster(seed=seed)
    raise ValueError(f"Benchmark has no forecaster for {family}.")


def _require_identity(summary: DatasetSummary, pin: BenchmarkPin) -> None:
    if summary["profile"] != pin.profile or summary["seed"] != pin.seed:
        raise ValueError(f"Generated dataset does not match the {BENCHMARK_VERSION} pin.")
    if summary["row_count"] != pin.daily_row_count:
        raise ValueError(
            f"{BENCHMARK_VERSION} daily row count is {pin.daily_row_count}, "
            f"but this run produced {summary['row_count']}."
        )


def _parse_pin(payload: object) -> BenchmarkPin:
    data = _object(payload, "Benchmark pin")
    version = _text(data.get("benchmark_version"), "benchmark_version")
    if version != BENCHMARK_VERSION:
        raise ValueError(f"Benchmark pin version must be {BENCHMARK_VERSION}.")
    models_payload = data.get("models")
    if not isinstance(models_payload, list) or not models_payload:
        raise ValueError("Benchmark pin needs a models list.")
    models = tuple(_parse_model(item, index) for index, item in enumerate(models_payload))
    families = tuple(item.family for item in models)
    if families != _FAMILIES:
        raise ValueError("Benchmark pin models must be naive, seasonal_naive, gradient_boosting.")
    tolerance = _finite(data.get("tolerance"), "tolerance")
    if tolerance < 0:
        raise ValueError("WAPE tolerance must be non-negative.")
    daily_row_count = _whole(data.get("daily_row_count"), "daily_row_count")
    weekly_row_count = _whole(data.get("weekly_row_count"), "weekly_row_count")
    folds = _whole(data.get("folds"), "folds")
    horizon = _whole(data.get("horizon"), "horizon")
    if daily_row_count < 1 or weekly_row_count < 1:
        raise ValueError("Benchmark row counts must be positive.")
    if folds < 3:
        raise ValueError("Benchmark folds must be at least 3.")
    if horizon < 1:
        raise ValueError("Benchmark horizon must be positive.")
    return BenchmarkPin(
        benchmark_version=version,
        seed=_whole(data.get("seed"), "seed"),
        profile=_text(data.get("profile"), "profile"),
        daily_row_count=daily_row_count,
        weekly_row_count=weekly_row_count,
        folds=folds,
        horizon=horizon,
        tolerance=tolerance,
        models=models,
    )


def _parse_model(payload: object, index: int) -> PinnedModel:
    data = _object(payload, f"models[{index}]")
    family = _text(data.get("family"), f"models[{index}].family")
    grain = _text(data.get("grain"), f"models[{index}].grain")
    if grain not in _GRAINS:
        raise ValueError(f"models[{index}].grain must be day or week.")
    wape = _finite(data.get("wape"), f"models[{index}].wape")
    return PinnedModel(family=family, grain=grain, wape=wape)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    return cast(dict[str, object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _whole(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number
