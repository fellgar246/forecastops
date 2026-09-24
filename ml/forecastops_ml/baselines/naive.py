"""Naive and seasonal-naive forecasters.

Naive repeats each series' last observation at every horizon step. Seasonal
naive copies the observation one season earlier. A missing lag uses that last
observation and increments ``fallback_count``. Neither model emits ``p10`` or
``p90``.
"""

from collections.abc import Mapping
from datetime import date, timedelta
from typing import cast

import pyarrow as pa

from forecastops_ml.evaluation.backtest import ForecastFrame, make_series_id

WEEKLY_SEASON_LENGTH = 7
YEARLY_SEASON_LENGTH = 52
_DAY = "day"
_WEEK = "week"


class NaiveForecaster:
    """Forecast ``value(t)`` at every step ``t + h``.

    ``fallback_count`` stays 0. A series with no history is an error. A last
    observation of zero is repeated as zero.
    """

    model_family = "naive"

    def __init__(self) -> None:
        self.fallback_count = 0
        self._last: dict[str, float] = {}

    def fit(self, history: pa.Table, config: Mapping[str, object] | None) -> None:
        """Remember the latest ``units_sold`` for each series."""

        _ = config
        self.fallback_count = 0
        self._last = _last_values(_history_points(history))

    def predict(self, horizon: int, known_future: pa.Table) -> ForecastFrame:
        """Repeat the last observation for every requested date."""

        _require_horizon(horizon)
        series_ids, days = _future_keys(known_future)
        missing = sorted({series_id for series_id in series_ids if series_id not in self._last})
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"No training observation for {joined}.")
        return ForecastFrame(
            series_id=series_ids,
            date=days,
            p50=tuple(self._last[series_id] for series_id in series_ids),
        )


class SeasonalNaiveForecaster:
    """Forecast the value one season before each requested date.

    ``season_length`` is a number of periods. ``period="day"`` lags that many
    calendar days, which is the weekly model on daily data when the length is
    7. ``period="week"`` lags that many weeks, which is the yearly model on
    weekly data when the length is 52.

    ``fallback_count`` counts forecast steps whose lag was absent. Fold 1, and
    a fit call that does not name a fold, starts the count again. Later folds
    on the same instance add to it.
    """

    model_family = "seasonal_naive"

    def __init__(self, season_length: int = WEEKLY_SEASON_LENGTH, *, period: str = _DAY) -> None:
        if type(season_length) is not int or season_length < 1:
            raise ValueError("Season length must be a positive number of periods.")
        if period not in {_DAY, _WEEK}:
            raise ValueError("Seasonal naive period must be day or week.")
        self.season_length = season_length
        self.period = period
        self.fallback_count = 0
        self._values: dict[tuple[str, date], float] = {}
        self._last: dict[str, float] = {}

    def fit(self, history: pa.Table, config: Mapping[str, object] | None) -> None:
        """Index history by series and date, and keep each series' last value."""

        if _starts_run(config):
            self.fallback_count = 0
        points = _history_points(history)
        self._values = points
        self._last = _last_values(points)

    def predict(self, horizon: int, known_future: pa.Table) -> ForecastFrame:
        """Copy the seasonal lag, or the last observation when the lag is missing."""

        _require_horizon(horizon)
        series_ids, days = _future_keys(known_future)
        predicted: list[float] = []
        for series_id, day in zip(series_ids, days, strict=True):
            last = self._last.get(series_id)
            if last is None:
                raise ValueError(f"No training observation for {series_id}.")
            lagged = self._values.get((series_id, _lag_date(day, self.season_length, self.period)))
            if lagged is None:
                self.fallback_count += 1
                predicted.append(last)
            else:
                predicted.append(lagged)
        return ForecastFrame(series_id=series_ids, date=days, p50=tuple(predicted))


def weekly_seasonal_naive() -> SeasonalNaiveForecaster:
    """Return seasonal naive with a 7-day lag for daily history."""

    return SeasonalNaiveForecaster(WEEKLY_SEASON_LENGTH, period=_DAY)


def yearly_seasonal_naive() -> SeasonalNaiveForecaster:
    """Return seasonal naive with a 52-week lag for weekly history."""

    return SeasonalNaiveForecaster(YEARLY_SEASON_LENGTH, period=_WEEK)


def _lag_date(day: date, season_length: int, period: str) -> date:
    if period == _WEEK:
        return day - timedelta(weeks=season_length)
    return day - timedelta(days=season_length)


def _starts_run(config: Mapping[str, object] | None) -> bool:
    if config is None:
        return True
    fold = config.get("fold")
    return fold is None or fold == 1


def _require_horizon(horizon: int) -> None:
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Horizon must be a positive number of periods.")


def _history_points(history: pa.Table) -> dict[tuple[str, date], float]:
    if not isinstance(history, pa.Table):
        raise ValueError("History must be a pyarrow table.")
    if history.num_rows == 0:
        raise ValueError("History has no rows.")
    series_ids = _series_ids(history)
    days = _dates(history)
    units = _units(history)
    points: dict[tuple[str, date], float] = {}
    for series_id, day, sold in zip(series_ids, days, units, strict=True):
        key = (series_id, day)
        if key in points:
            raise ValueError(f"History repeats series {series_id} on {day.isoformat()}.")
        points[key] = sold
    return points


def _last_values(points: Mapping[tuple[str, date], float]) -> dict[str, float]:
    last_on: dict[str, date] = {}
    last: dict[str, float] = {}
    for (series_id, day), sold in points.items():
        previous = last_on.get(series_id)
        if previous is None or day > previous:
            last_on[series_id] = day
            last[series_id] = sold
    return last


def _future_keys(known_future: pa.Table) -> tuple[tuple[str, ...], tuple[date, ...]]:
    if not isinstance(known_future, pa.Table):
        raise ValueError("Known future must be a pyarrow table.")
    if known_future.num_rows == 0:
        raise ValueError("Known future has no rows.")
    series_ids = tuple(_series_ids(known_future))
    days = tuple(_dates(known_future))
    return series_ids, days


def _series_ids(frame: pa.Table) -> list[str]:
    if "series_id" in frame.column_names:
        column = frame.column("series_id")
        if column.null_count:
            raise ValueError("Column series_id contains nulls.")
        values = column.to_pylist()
        if any(not isinstance(value, str) or value.strip() == "" for value in values):
            raise ValueError("series_id values must be non-empty strings.")
        return [str(value) for value in values]
    if "store_id" not in frame.column_names or "sku_id" not in frame.column_names:
        raise ValueError("History needs series_id, or store_id and sku_id.")
    stores = frame.column("store_id").to_pylist()
    skus = frame.column("sku_id").to_pylist()
    return [
        make_series_id(str(store_id), str(sku_id))
        for store_id, sku_id in zip(stores, skus, strict=True)
    ]


def _dates(frame: pa.Table) -> list[date]:
    if "date" not in frame.column_names:
        raise ValueError("History is missing date.")
    column = frame.column("date")
    if pa.types.is_date32(column.type):
        values = column.to_pylist()
    else:
        try:
            casted = column.cast(pa.date32())
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            raise ValueError(f"Column date has type {column.type}, expected a date.") from exc
        values = casted.to_pylist()
    if any(value is None or type(value) is not date for value in values):
        raise ValueError("Column date must contain calendar dates.")
    return cast(list[date], values)


def _units(frame: pa.Table) -> list[float]:
    if "units_sold" not in frame.column_names:
        raise ValueError("History is missing units_sold.")
    column = frame.column("units_sold")
    if column.null_count:
        raise ValueError("Column units_sold contains nulls.")
    if not pa.types.is_integer(column.type) and not pa.types.is_floating(column.type):
        raise ValueError(f"Column units_sold has type {column.type}, expected a number.")
    values = [float(value) for value in column.to_pylist()]
    return values
