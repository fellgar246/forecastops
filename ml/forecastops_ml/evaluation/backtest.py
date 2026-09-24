"""Rolling-origin backtesting.

:class:`Backtester` fits a :class:`Forecaster` once per fold and scores the
pooled predictions. Metrics are global and sliced by category, store, horizon
step, and SKU demand quartile. Quartiles use total units sold in that fold's
training rows only.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol, cast

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from forecastops_ml.evaluation.folds import MIN_PUBLISHED_FOLDS, Fold, build_folds
from forecastops_ml.evaluation.metrics import bias, mae, pinball_loss, rmse, smape, wape

F64 = NDArray[np.float64]
_REQUIRED = ("date", "store_id", "sku_id", "category_id", "units_sold")


class Forecaster(Protocol):
    """Model fitted on history before an origin and scored on one horizon."""

    model_family: str

    def fit(self, history: pa.Table, config: Mapping[str, object] | None) -> None:
        """Learn from observations strictly before the forecast origin."""

    def predict(self, horizon: int, known_future: pa.Table) -> "ForecastFrame":
        """Return one forecast row per known-future key.

        ``known_future`` identifies the series and dates to forecast. It does
        not include ``units_sold``.
        """


@dataclass(frozen=True)
class ForecastFrame:
    """Point forecasts for one or more series.

    ``series_id`` is ``{store_id}|{sku_id}`` from :func:`make_series_id`.
    ``p50`` is the point forecast. ``p10`` and ``p90`` are omitted until the
    model emits those quantiles.
    """

    series_id: tuple[str, ...]
    date: tuple[date, ...]
    p50: tuple[float, ...]
    p10: tuple[float, ...] | None = None
    p90: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        count = len(self.series_id)
        if count == 0:
            raise ValueError("Forecast frame has no rows.")
        if len(self.date) != count or len(self.p50) != count:
            raise ValueError("Forecast columns must have the same length.")
        if any(not isinstance(value, str) or value.strip() == "" for value in self.series_id):
            raise ValueError("Forecast series_id values must be non-empty strings.")
        if any(type(value) is not date for value in self.date):
            raise ValueError("Forecast dates must be calendar dates.")
        object.__setattr__(self, "p50", _finite_column(self.p50, "p50"))
        object.__setattr__(self, "p10", _optional_quantile(self.p10, count, "p10"))
        object.__setattr__(self, "p90", _optional_quantile(self.p90, count, "p90"))


@dataclass(frozen=True)
class MetricSummary:
    """One row of forecast metrics.

    Pinball fields are null when that quantile was not forecast. ``row_count``
    is the number of scored points in the group.
    """

    row_count: int
    mae: float
    rmse: float
    wape: float
    smape: float
    bias: float
    pinball_loss_p10: float | None
    pinball_loss_p90: float | None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for these metrics."""

        return {
            "bias": self.bias,
            "mae": self.mae,
            "pinball_loss_p10": self.pinball_loss_p10,
            "pinball_loss_p90": self.pinball_loss_p90,
            "rmse": self.rmse,
            "row_count": self.row_count,
            "smape": self.smape,
            "wape": self.wape,
        }


@dataclass(frozen=True)
class SliceMetrics:
    """Metrics for one slice value, such as a category or a horizon step."""

    key: str
    metrics: MetricSummary

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this slice."""

        return {"key": self.key, "metrics": self.metrics.to_dict()}


@dataclass(frozen=True)
class EvaluationReport:
    """JSON-serializable score for one model on one dataset.

    WAPE in ``global_metrics`` is the published business score. Slice tables
    repeat the same metrics by category, store, horizon step, and SKU demand
    quartile. ``wape_by_horizon`` is that same WAPE at each horizon step, so
    degradation across the horizon stays visible. ``runtime_ms`` is total fit
    time in milliseconds. ``skipped_series`` lists series left out of the
    score, each with an English reason.
    """

    model_family: str
    fold_count: int
    horizon: int
    dataset_version: str | None
    global_metrics: MetricSummary
    by_category: tuple[SliceMetrics, ...]
    by_store: tuple[SliceMetrics, ...]
    by_horizon_step: tuple[SliceMetrics, ...]
    by_demand_quartile: tuple[SliceMetrics, ...]
    runtime_ms: int
    wape_by_horizon: tuple[tuple[int, float], ...]
    skipped_series: tuple["SkippedSeries", ...]

    def to_dict(self) -> dict[str, object]:
        """Return the JSON document for this report."""

        return {
            "dataset_version": self.dataset_version,
            "fold_count": self.fold_count,
            "horizon": self.horizon,
            "metrics": {
                "by_category": [item.to_dict() for item in self.by_category],
                "by_demand_quartile": [item.to_dict() for item in self.by_demand_quartile],
                "by_horizon_step": [item.to_dict() for item in self.by_horizon_step],
                "by_store": [item.to_dict() for item in self.by_store],
                "global": self.global_metrics.to_dict(),
            },
            "model_family": self.model_family,
            "runtime_ms": self.runtime_ms,
            "skipped_series": [item.to_dict() for item in self.skipped_series],
            "wape_by_horizon": [
                {"step": step, "wape": value} for step, value in self.wape_by_horizon
            ],
        }


@dataclass(frozen=True)
class SkippedSeries:
    """A series left out of the score, with an English reason."""

    series_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        """Return the JSON object for this skip."""

        return {"reason": self.reason, "series_id": self.series_id}


class UnscoredModelError(ValueError):
    """Every series was skipped, so the score has no rows."""

    def __init__(self, skipped: tuple[SkippedSeries, ...]) -> None:
        self.skipped = skipped
        detail = "; ".join(f"{item.series_id}: {item.reason}" for item in skipped)
        super().__init__(f"Every series was skipped. {detail}")


@dataclass(frozen=True)
class _Point:
    actual: float
    predicted: float
    p10: float | None
    p90: float | None
    category_id: str
    store_id: str
    horizon_step: int
    quartile: str | None


class Backtester:
    """Score a forecaster with rolling-origin folds."""

    def evaluate(
        self,
        forecaster: Forecaster,
        frame: pa.Table,
        folds: int,
        horizon: int,
        *,
        dataset_version: str | None = None,
    ) -> EvaluationReport:
        """Fit and score ``forecaster`` on ``frame``.

        ``folds`` is at least 3 for a published score. Five folds is the
        preferred count for a final benchmark. ``horizon`` is a positive number
        of periods at the frame's grain. ``dataset_version`` is copied onto
        the report when the caller has one.
        """

        family = _model_family(forecaster)
        version = _dataset_version(dataset_version)
        _require_published_folds(folds)
        if type(horizon) is not int or horizon < 1:
            raise ValueError("Horizon must be a positive number of periods.")
        _require_frame(frame)
        _reject_duplicate_grain(frame)

        fold_list = build_folds(_distinct_dates(frame), folds=folds, horizon=horizon)
        points: list[_Point] = []
        quantiles: bool | None = None
        runtime_ms = 0
        skipped: list[SkippedSeries] = []
        for fold in fold_list:
            train, valid = split_fold(frame, fold)
            if train.num_rows == 0:
                raise ValueError(f"Fold {fold.index} has no training rows.")
            if valid.num_rows == 0:
                raise ValueError(f"Fold {fold.index} has no validation rows.")
            history = _with_series_id(train)
            labeled = _with_series_id(valid)
            config: dict[str, object] = {
                "fold": fold.index,
                "horizon": horizon,
                "origin": fold.origin.isoformat(),
            }
            started = time.perf_counter()
            forecaster.fit(history, config)
            measured_ms = int(round((time.perf_counter() - started) * 1000))
            runtime_ms += _fit_runtime_ms(forecaster, measured_ms)
            fold_skipped = _skipped_series(forecaster)
            skipped.extend(fold_skipped)
            labeled = _without_series(labeled, {item.series_id for item in fold_skipped})
            if labeled.num_rows == 0:
                continue
            known_future = labeled.drop(["units_sold"])
            forecast = forecaster.predict(horizon, known_future)
            if not isinstance(forecast, ForecastFrame):
                raise ValueError("Forecaster.predict must return a ForecastFrame.")
            quantiles = _merge_quantile_flag(quantiles, forecast)
            quartiles = demand_quartiles(
                _strings(train, "sku_id").tolist(),
                _units(train).tolist(),
            )
            points.extend(_align(labeled, forecast, fold, quartiles))
        if not points:
            unique = _unique_skipped(skipped)
            if unique:
                raise UnscoredModelError(unique)
            raise ValueError("Cannot score a forecast with no validation rows.")
        return _report(
            family,
            folds,
            horizon,
            version,
            points,
            runtime_ms=runtime_ms,
            skipped=_unique_skipped(skipped),
        )


def split_fold(frame: pa.Table, fold: Fold) -> tuple[pa.Table, pa.Table]:
    """Split ``frame`` into training rows and this fold's validation rows.

    Training rows have ``date`` strictly before ``fold.origin``. Validation
    rows fall on ``fold.validation_dates``.
    """

    _require_frame(frame)
    days = _dates(frame)
    validation = set(fold.validation_dates)
    train_mask = pa.array([day < fold.origin for day in days])
    valid_mask = pa.array([day in validation for day in days])
    return frame.filter(train_mask), frame.filter(valid_mask)


def make_series_id(store_id: str, sku_id: str) -> str:
    """Return the daily series key ``{store_id}|{sku_id}``."""

    if not isinstance(store_id, str) or not isinstance(sku_id, str):
        raise ValueError("store_id and sku_id must be strings.")
    if store_id == "" or sku_id == "":
        raise ValueError("store_id and sku_id must be non-empty.")
    if "|" in store_id or "|" in sku_id:
        raise ValueError("store_id and sku_id cannot contain '|'.")
    return f"{store_id}|{sku_id}"


def demand_quartiles(sku_ids: Sequence[str], units_sold: Sequence[float]) -> dict[str, str]:
    """Label each SKU ``q1`` through ``q4`` from total units sold.

    Edges are the 25th, 50th, and 75th percentiles of those totals. A total on
    an edge stays in the lower quartile. Pass training rows only; rows from
    the validation window would leak demand into the label.
    """

    if len(sku_ids) != len(units_sold):
        raise ValueError("sku_id and units_sold must have the same length.")
    if not sku_ids:
        return {}
    totals: dict[str, float] = {}
    for sku_id, units in zip(sku_ids, units_sold, strict=True):
        if not isinstance(sku_id, str) or sku_id == "":
            raise ValueError("sku_id must be a non-empty string.")
        totals[sku_id] = totals.get(sku_id, 0.0) + float(units)
    ordered = sorted(totals)
    values = np.asarray([totals[sku_id] for sku_id in ordered], dtype=np.float64)
    edges = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    return {sku_id: _quartile_label(total, edges) for sku_id, total in totals.items()}


def _report(
    family: str,
    folds: int,
    horizon: int,
    dataset_version: str | None,
    points: Sequence[_Point],
    *,
    runtime_ms: int,
    skipped: tuple[SkippedSeries, ...],
) -> EvaluationReport:
    by_horizon_step = tuple(
        sorted(
            _slices(
                points,
                lambda point: str(point.horizon_step),
                lambda key: f" for horizon step {key}",
            ),
            key=lambda item: int(item.key),
        )
    )
    return EvaluationReport(
        model_family=family,
        fold_count=folds,
        horizon=horizon,
        dataset_version=dataset_version,
        global_metrics=_summarize(points, ""),
        by_category=_slices(
            points,
            lambda point: point.category_id,
            lambda key: f" for category {key}",
        ),
        by_store=_slices(
            points,
            lambda point: point.store_id,
            lambda key: f" for store {key}",
        ),
        by_horizon_step=by_horizon_step,
        by_demand_quartile=_slices(
            points,
            lambda point: point.quartile,
            lambda key: f" for demand quartile {key}",
        ),
        runtime_ms=runtime_ms,
        wape_by_horizon=tuple((int(item.key), item.metrics.wape) for item in by_horizon_step),
        skipped_series=skipped,
    )


def _fit_runtime_ms(forecaster: Forecaster, measured_ms: int) -> int:
    recorded = getattr(forecaster, "fit_runtime_ms", None)
    runtime_ms = measured_ms if recorded is None else int(recorded)
    if runtime_ms < 0:
        raise ValueError("Fit runtime must be zero or a positive number of milliseconds.")
    return runtime_ms


def _skipped_series(forecaster: Forecaster) -> tuple[SkippedSeries, ...]:
    raw = getattr(forecaster, "skipped_series", ())
    if raw is None:
        return ()
    if not isinstance(raw, tuple) or any(not isinstance(item, SkippedSeries) for item in raw):
        raise ValueError("skipped_series must be a tuple of SkippedSeries.")
    if any(item.reason.strip() == "" or item.series_id.strip() == "" for item in raw):
        raise ValueError("A skipped series needs a series id and an English reason.")
    return raw


def _unique_skipped(skipped: Sequence[SkippedSeries]) -> tuple[SkippedSeries, ...]:
    ordered: dict[str, SkippedSeries] = {}
    for item in skipped:
        ordered.setdefault(item.series_id, item)
    return tuple(ordered[series_id] for series_id in sorted(ordered))


def _without_series(frame: pa.Table, skipped_ids: set[str]) -> pa.Table:
    if not skipped_ids:
        return frame
    series_ids = frame.column("series_id").to_pylist()
    keep = pa.array([series_id not in skipped_ids for series_id in series_ids])
    return frame.filter(keep)


def _slices(
    points: Sequence[_Point],
    key_of: Callable[[_Point], str | None],
    scope_of: Callable[[str], str],
) -> tuple[SliceMetrics, ...]:
    groups: dict[str, list[_Point]] = {}
    for point in points:
        key = key_of(point)
        if key is None:
            continue
        groups.setdefault(key, []).append(point)
    return tuple(
        SliceMetrics(key=key, metrics=_summarize(groups[key], scope_of(key)))
        for key in sorted(groups)
    )


def _summarize(points: Sequence[_Point], scope: str) -> MetricSummary:
    if not points:
        raise ValueError(f"Cannot score an empty forecast{scope}.")
    actual = [point.actual for point in points]
    predicted = [point.predicted for point in points]
    total = float(np.sum(np.asarray(actual, dtype=np.float64)))
    if total == 0.0:
        raise ValueError(f"WAPE is undefined because the sum of actual demand is zero{scope}.")
    p10 = _quantile_column(points, "p10")
    p90 = _quantile_column(points, "p90")
    return MetricSummary(
        row_count=len(points),
        mae=mae(actual, predicted),
        rmse=rmse(actual, predicted),
        wape=wape(actual, predicted),
        smape=smape(actual, predicted),
        bias=bias(actual, predicted),
        pinball_loss_p10=None if p10 is None else pinball_loss(actual, p10, 0.1),
        pinball_loss_p90=None if p90 is None else pinball_loss(actual, p90, 0.9),
    )


def _quantile_column(points: Sequence[_Point], name: str) -> list[float] | None:
    values = [getattr(point, name) for point in points]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Quantile forecasts must be present on every row or on none.")
    return [float(value) for value in values]


def _align(
    valid: pa.Table,
    forecast: ForecastFrame,
    fold: Fold,
    quartiles: Mapping[str, str],
) -> list[_Point]:
    index: dict[tuple[str, date], tuple[float, float | None, float | None]] = {}
    p10_values = forecast.p10
    p90_values = forecast.p90
    for position, series_id in enumerate(forecast.series_id):
        forecast_key = (series_id, forecast.date[position])
        if forecast_key in index:
            stamp = forecast.date[position].isoformat()
            raise ValueError(f"Forecast repeats series {series_id} on {stamp}.")
        p10 = None if p10_values is None else p10_values[position]
        p90 = None if p90_values is None else p90_values[position]
        index[forecast_key] = (forecast.p50[position], p10, p90)

    steps = {day: step for step, day in enumerate(fold.validation_dates, start=1)}
    stores = _strings(valid, "store_id")
    skus = _strings(valid, "sku_id")
    categories = _strings(valid, "category_id")
    units = _units(valid)
    days = _dates(valid)
    points: list[_Point] = []
    for position, day in enumerate(days):
        series_id = make_series_id(str(stores[position]), str(skus[position]))
        key = (series_id, day)
        found = index.pop(key, None)
        if found is None:
            raise ValueError(f"Forecast is missing series {series_id} on {day.isoformat()}.")
        predicted, p10, p90 = found
        step = steps.get(day)
        if step is None:
            raise ValueError(f"Validation row on {day.isoformat()} is outside fold {fold.index}.")
        sku_id = str(skus[position])
        points.append(
            _Point(
                actual=float(units[position]),
                predicted=predicted,
                p10=p10,
                p90=p90,
                category_id=str(categories[position]),
                store_id=str(stores[position]),
                horizon_step=step,
                quartile=quartiles.get(sku_id),
            )
        )
    if index:
        raise ValueError("Forecast contains rows that are not in the validation window.")
    return points


def _merge_quantile_flag(seen: bool | None, forecast: ForecastFrame) -> bool:
    present = forecast.p10 is not None or forecast.p90 is not None
    if seen is not None and present != seen:
        raise ValueError("Quantile forecasts must be present on every fold or on none.")
    return present


def _with_series_id(frame: pa.Table) -> pa.Table:
    stores = _strings(frame, "store_id")
    skus = _strings(frame, "sku_id")
    series_ids = [
        make_series_id(str(store_id), str(sku_id))
        for store_id, sku_id in zip(stores.tolist(), skus.tolist(), strict=True)
    ]
    column = pa.array(series_ids, type=pa.string())
    if "series_id" in frame.column_names:
        frame = frame.drop(["series_id"])
    return frame.append_column("series_id", column)


def _model_family(forecaster: Forecaster) -> str:
    family = getattr(forecaster, "model_family", None)
    if not isinstance(family, str) or family.strip() == "":
        raise ValueError("Forecaster model_family must be a non-empty string.")
    return family


def _dataset_version(dataset_version: str | None) -> str | None:
    if dataset_version is None:
        return None
    if not isinstance(dataset_version, str) or dataset_version.strip() == "":
        raise ValueError("dataset_version must be a non-empty string when it is provided.")
    return dataset_version


def _require_published_folds(folds: int) -> None:
    # bool is a subclass of int, so reject it by exact type.
    if type(folds) is not int or folds < MIN_PUBLISHED_FOLDS:
        raise ValueError(f"A published score needs at least {MIN_PUBLISHED_FOLDS} folds.")


def _require_frame(frame: pa.Table) -> None:
    if not isinstance(frame, pa.Table):
        raise ValueError("Observation frame must be a pyarrow table.")
    missing = [name for name in _REQUIRED if name not in frame.column_names]
    if missing:
        found = ", ".join(missing)
        raise ValueError(f"Observation table is missing {found}.")
    if frame.num_rows == 0:
        raise ValueError("Observation table has no rows.")


def _reject_duplicate_grain(frame: pa.Table) -> None:
    days = [day.toordinal() for day in _dates(frame)]
    stores = _strings(frame, "store_id").tolist()
    skus = _strings(frame, "sku_id").tolist()
    keys = list(zip(days, stores, skus, strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError("Observation table repeats a (date, store_id, sku_id) key.")


def _distinct_dates(frame: pa.Table) -> tuple[date, ...]:
    return tuple(sorted(set(_dates(frame))))


def _quartile_label(total: float, edges: F64) -> str:
    if total <= float(edges[0]):
        return "q1"
    if total <= float(edges[1]):
        return "q2"
    if total <= float(edges[2]):
        return "q3"
    return "q4"


def _finite_column(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not all(np.isfinite(value) for value in converted):
        raise ValueError(f"Forecast {name} contains a non-finite value.")
    return converted


def _optional_quantile(
    values: tuple[float, ...] | None,
    count: int,
    name: str,
) -> tuple[float, ...] | None:
    if values is None:
        return None
    if len(values) != count:
        raise ValueError(f"Forecast {name} must have one value per row.")
    return _finite_column(values, name)


def _dates(frame: pa.Table) -> list[date]:
    column = _as_date(frame.column("date"))
    if column.null_count:
        raise ValueError("Column date contains nulls.")
    values = column.to_pylist()
    if any(type(value) is not date for value in values):
        raise ValueError("Column date must contain calendar dates.")
    return cast(list[date], values)


def _as_date(column: pa.ChunkedArray) -> pa.ChunkedArray:
    if pa.types.is_date32(column.type):
        return column
    try:
        casted = column.cast(pa.date32())
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise ValueError(f"Column date has type {column.type}, expected a date.") from exc
    if isinstance(casted, pa.Array):
        return pa.chunked_array([casted])
    return casted


def _units(frame: pa.Table) -> np.ndarray:
    column = frame.column("units_sold")
    if not pa.types.is_integer(column.type):
        raise ValueError(f"Column units_sold has type {column.type}, expected an integer.")
    if column.null_count:
        raise ValueError("Column units_sold contains nulls.")
    values = column.cast(pa.int64()).to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.int64)


def _strings(frame: pa.Table, name: str) -> np.ndarray:
    column = _as_string(frame.column(name), name)
    if column.null_count:
        raise ValueError(f"Column {name} contains nulls.")
    values = column.to_pylist()
    if any(value == "" for value in values):
        raise ValueError(f"Column {name} contains a blank value.")
    return np.asarray(values, dtype=object)


def _as_string(column: pa.ChunkedArray, name: str) -> pa.ChunkedArray:
    if pa.types.is_dictionary(column.type):
        casted = column.cast(pa.string())
        if isinstance(casted, pa.Array):
            return pa.chunked_array([casted])
        column = casted
    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        return column
    raise ValueError(f"Column {name} has type {column.type}, expected a string.")
