"""Additive Holt-Winters forecasts for category-week series.

One model is fit per category on a weekly aggregate. The seasonal period is
52 weeks. A category with fewer than two full seasons is skipped. A fit that
does not converge is skipped with an English reason. Forecasts are point
values in ``p50``. ``p10`` and ``p90`` stay null.
"""

import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from forecastops_ml.evaluation.backtest import ForecastFrame, SkippedSeries, make_series_id

SEASON_LENGTH = 52
MIN_HISTORY_WEEKS = 2 * SEASON_LENGTH
_WEEK_DAYS = 7
F64 = NDArray[np.float64]


@dataclass(frozen=True)
class _Row:
    category_id: str
    week: date
    units: float
    series_id: str


class HoltWintersForecaster:
    """Fit additive Holt-Winters on one row per category and week.

    ``fit_runtime_ms`` is the duration of the latest ``fit`` call in
    milliseconds. ``skipped_series`` lists categories that were too short, not
    weekly, or did not converge. Each entry carries an English reason.
    """

    model_family = "holt_winters"

    def __init__(self) -> None:
        self.fit_runtime_ms = 0
        self.skipped_series: tuple[SkippedSeries, ...] = ()
        self._fitted: dict[str, object] = {}
        self._last_week: dict[str, date] = {}
        self._skip_reasons: dict[str, str] = {}

    def fit(self, history: pa.Table, config: Mapping[str, object] | None) -> None:
        """Fit one additive model per category that has two seasonal cycles."""

        _ = config
        started = time.perf_counter()
        try:
            self._fitted = {}
            self._last_week = {}
            self._skip_reasons = {}
            rows = _history_rows(history)
            if _has_duplicate_week(rows):
                grain_reason = (
                    "Holt-Winters fits category-week aggregates with one row per category and week."
                )
                self.skipped_series = _skip_all(rows, grain_reason)
                self._skip_reasons = {item.series_id: item.reason for item in self.skipped_series}
                return
            skipped: list[SkippedSeries] = []
            grouped = _by_category(rows)
            for category_id in sorted(grouped):
                observations = grouped[category_id]
                series_ids = _series_ids(observations)
                reason = _unfittable_reason(category_id, observations)
                if reason is None:
                    values = np.asarray([row.units for row in observations], dtype=np.float64)
                    try:
                        self._fitted[category_id] = _fit_category(category_id, values)
                    except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                        reason = _failure_reason(category_id, exc)
                if reason is not None:
                    skipped.extend(SkippedSeries(series_id, reason) for series_id in series_ids)
                    continue
                self._last_week[category_id] = observations[-1].week
            self.skipped_series = tuple(skipped)
            self._skip_reasons = {item.series_id: item.reason for item in self.skipped_series}
        finally:
            self.fit_runtime_ms = int(round((time.perf_counter() - started) * 1000))

    def predict(self, horizon: int, known_future: pa.Table) -> ForecastFrame:
        """Return one finite ``p50`` per requested category-week."""

        if type(horizon) is not int or horizon < 1:
            raise ValueError("Horizon must be a positive number of periods.")
        rows = _future_rows(known_future)
        predicted = [0.0] * len(rows)
        forecasts: dict[str, list[float]] = {}
        for index, row in enumerate(rows):
            skipped = self._skip_reasons.get(row.series_id)
            if skipped is not None:
                raise ValueError(skipped)
            fitted = self._fitted.get(row.category_id)
            last_week = self._last_week.get(row.category_id)
            if fitted is None or last_week is None:
                raise ValueError(f"No training observation for {row.series_id}.")
            step = _steps_after(last_week, row.week)
            cached = forecasts.get(row.category_id)
            if cached is None or len(cached) < step:
                cached = _forecast_ahead(fitted, max(step, horizon))
                forecasts[row.category_id] = cached
            predicted[index] = cached[step - 1]
        return ForecastFrame(
            series_id=tuple(row.series_id for row in rows),
            date=tuple(row.week for row in rows),
            p50=tuple(predicted),
        )


def _fit_category(category_id: str, values: F64) -> object:
    """Fit additive exponential smoothing and reject a non-finite forecast."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        fitted = ExponentialSmoothing(
            values,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASON_LENGTH,
            initialization_method="estimated",
        ).fit(optimized=True)
    retvals = getattr(fitted, "mle_retvals", None)
    if isinstance(retvals, dict) and retvals.get("converged") is False:
        raise ValueError(f"Holt-Winters did not converge for category {category_id}.")
    preview = np.asarray(fitted.forecast(1), dtype=np.float64).reshape(-1)
    if preview.size != 1 or not bool(np.isfinite(preview[0])):
        raise ValueError(f"Holt-Winters returned a non-finite forecast for category {category_id}.")
    return fitted


def _forecast_ahead(fitted: object, steps: int) -> list[float]:
    forecast = getattr(fitted, "forecast", None)
    if not callable(forecast):
        raise ValueError("Holt-Winters fit did not return a forecast method.")
    values = np.asarray(forecast(steps), dtype=np.float64).reshape(-1)
    if values.size != steps or not bool(np.all(np.isfinite(values))):
        raise ValueError("Holt-Winters returned a non-finite forecast.")
    return [float(value) for value in values.tolist()]


def _failure_reason(category_id: str, exc: BaseException) -> str:
    text = str(exc).strip()
    if text.startswith("Holt-Winters"):
        return text
    detail = text or "the optimizer did not converge"
    return f"Holt-Winters did not converge for category {category_id}. {detail}"


def _unfittable_reason(category_id: str, rows: Sequence[_Row]) -> str | None:
    weeks = [row.week for row in rows]
    count = len(weeks)
    if count < MIN_HISTORY_WEEKS:
        return f"Category {category_id} has {count} weeks, which is fewer than {MIN_HISTORY_WEEKS}."
    for previous, current in pairwise(weeks):
        if (current - previous).days != _WEEK_DAYS:
            return (
                f"Category {category_id} is missing a week, so a "
                f"{SEASON_LENGTH}-week season cannot be fit."
            )
    return None


def _skip_all(rows: Sequence[_Row], reason: str) -> tuple[SkippedSeries, ...]:
    return tuple(SkippedSeries(series_id, reason) for series_id in _series_ids(rows))


def _has_duplicate_week(rows: Sequence[_Row]) -> bool:
    seen: set[tuple[str, date]] = set()
    for row in rows:
        key = (row.category_id, row.week)
        if key in seen:
            return True
        seen.add(key)
    return False


def _by_category(rows: Sequence[_Row]) -> dict[str, list[_Row]]:
    grouped: dict[str, list[_Row]] = {}
    for row in rows:
        grouped.setdefault(row.category_id, []).append(row)
    for category_id, observations in grouped.items():
        ordered = sorted(observations, key=lambda item: item.week)
        if len({item.week for item in ordered}) != len(ordered):
            raise ValueError(f"Category {category_id} repeats a week.")
        grouped[category_id] = ordered
    return grouped


def _series_ids(rows: Sequence[_Row]) -> tuple[str, ...]:
    return tuple(sorted({row.series_id for row in rows}))


def _history_rows(history: pa.Table) -> list[_Row]:
    if not isinstance(history, pa.Table):
        raise ValueError("History must be a pyarrow table.")
    if history.num_rows == 0:
        raise ValueError("History has no rows.")
    categories = _categories(history)
    weeks = _weeks(history)
    units = _units(history)
    series_ids = _resolved_series_ids(history, categories)
    return [
        _Row(category_id, week, sold, series_id)
        for category_id, week, sold, series_id in zip(
            categories, weeks, units, series_ids, strict=True
        )
    ]


def _future_rows(known_future: pa.Table) -> list[_Row]:
    if not isinstance(known_future, pa.Table):
        raise ValueError("Known future must be a pyarrow table.")
    if known_future.num_rows == 0:
        raise ValueError("Known future has no rows.")
    categories = _categories(known_future)
    weeks = _weeks(known_future)
    series_ids = _resolved_series_ids(known_future, categories)
    return [
        _Row(category_id, week, 0.0, series_id)
        for category_id, week, series_id in zip(categories, weeks, series_ids, strict=True)
    ]


def _categories(frame: pa.Table) -> list[str]:
    if "category_id" not in frame.column_names:
        raise ValueError("Holt-Winters history needs category_id.")
    column = frame.column("category_id")
    if column.null_count:
        raise ValueError("Column category_id contains nulls.")
    values = column.to_pylist()
    if any(not isinstance(value, str) or value.strip() == "" for value in values):
        raise ValueError("category_id values must be non-empty strings.")
    return [str(value) for value in values]


def _weeks(frame: pa.Table) -> list[date]:
    name = "week_start" if "week_start" in frame.column_names else "date"
    if name not in frame.column_names:
        raise ValueError("Holt-Winters history needs week_start or date.")
    column = frame.column(name)
    if pa.types.is_date32(column.type):
        values = column.to_pylist()
    else:
        try:
            casted = column.cast(pa.date32())
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            raise ValueError(f"Column {name} has type {column.type}, expected a date.") from exc
        values = casted.to_pylist()
    if any(value is None or type(value) is not date for value in values):
        raise ValueError(f"Column {name} must contain calendar dates.")
    return [value for value in values if type(value) is date]


def _units(frame: pa.Table) -> list[float]:
    if "units_sold" not in frame.column_names:
        raise ValueError("History is missing units_sold.")
    column = frame.column("units_sold")
    if column.null_count:
        raise ValueError("Column units_sold contains nulls.")
    if not pa.types.is_integer(column.type) and not pa.types.is_floating(column.type):
        raise ValueError(f"Column units_sold has type {column.type}, expected a number.")
    return [float(value) for value in column.to_pylist()]


def _resolved_series_ids(frame: pa.Table, categories: Sequence[str]) -> list[str]:
    if "series_id" in frame.column_names:
        column = frame.column("series_id")
        if column.null_count:
            raise ValueError("Column series_id contains nulls.")
        values = column.to_pylist()
        if any(not isinstance(value, str) or value.strip() == "" for value in values):
            raise ValueError("series_id values must be non-empty strings.")
        return [str(value) for value in values]
    if "store_id" in frame.column_names and "sku_id" in frame.column_names:
        stores = frame.column("store_id").to_pylist()
        skus = frame.column("sku_id").to_pylist()
        return [
            make_series_id(str(store_id), str(sku_id))
            for store_id, sku_id in zip(stores, skus, strict=True)
        ]
    return list(categories)


def _steps_after(last_week: date, target: date) -> int:
    delta = (target - last_week).days
    if delta <= 0 or delta % _WEEK_DAYS != 0:
        raise ValueError(
            f"Week {target.isoformat()} is not a later 7-day step after {last_week.isoformat()}."
        )
    return delta // _WEEK_DAYS
