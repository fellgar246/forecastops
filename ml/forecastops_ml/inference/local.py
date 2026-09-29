"""Local forecasts for the same state machine the batch path uses.

``deepar`` without a cloud job uses :func:`quantile_forecast`, a fake quantile
model that emits P10, P50, and P90 from the last observation. Other families
use the local catalog.
"""

import math
from datetime import date, timedelta

import pyarrow as pa

from forecastops_ml.baselines.catalog import local_catalog
from forecastops_ml.evaluation.backtest import ForecastFrame, make_series_id
from forecastops_ml.inference.errors import InferenceError
from forecastops_ml.inference.points import StoredForecastPoint


def quantile_forecast(
    frame: pa.Table,
    *,
    horizon: int,
    granularity: str,
) -> tuple[StoredForecastPoint, ...]:
    """Forecast each series from its latest ``units_sold``.

    P50 is that latest value. P10 is 80 percent of it and P90 is 120 percent.
    The same series and horizon always produce the same points.
    """

    latest = _latest_demand(frame)
    step = _step(granularity)
    _require_horizon(horizon)
    points: list[StoredForecastPoint] = []
    for (store_id, sku_id), (origin, value) in sorted(latest.items()):
        p50 = value
        p10 = min(value * 0.8, value)
        p90 = max(value * 1.2, value)
        series_id = make_series_id(store_id, sku_id)
        for offset in range(1, horizon + 1):
            points.append(
                StoredForecastPoint(
                    series_id=series_id,
                    date=origin + step * offset,
                    p10=p10,
                    p50=p50,
                    p90=p90,
                )
            )
    return tuple(points)


def local_family_forecast(
    frame: pa.Table,
    family: str,
    *,
    horizon: int,
    granularity: str,
) -> tuple[StoredForecastPoint, ...]:
    """Fit a local catalog model and forecast the next ``horizon`` periods."""

    _require_horizon(horizon)
    try:
        forecaster = local_catalog().get(family)
    except ValueError as exc:
        raise InferenceError(str(exc)) from exc
    forecaster.fit(frame, None)
    future = future_frame(frame, horizon, granularity)
    predicted = forecaster.predict(horizon, future)
    if not isinstance(predicted, ForecastFrame):
        raise InferenceError("The local model did not return a forecast.")
    return points_from_forecast_frame(predicted)


def points_from_forecast_frame(predicted: ForecastFrame) -> tuple[StoredForecastPoint, ...]:
    """Copy a catalog forecast into stored points."""

    points: list[StoredForecastPoint] = []
    for index, (series_id, day) in enumerate(zip(predicted.series_id, predicted.date, strict=True)):
        p10 = None if predicted.p10 is None else predicted.p10[index]
        p90 = None if predicted.p90 is None else predicted.p90[index]
        points.append(
            StoredForecastPoint(
                series_id=series_id,
                date=day,
                p10=p10,
                p50=predicted.p50[index],
                p90=p90,
            )
        )
    return tuple(points)


def future_frame(frame: pa.Table, horizon: int, granularity: str) -> pa.Table:
    """Return the store, SKU, and dates a local model should score."""

    _require_horizon(horizon)
    step = _step(granularity)
    keys = _series_keys(frame)
    if not keys:
        raise InferenceError("The dataset has no observation dates to forecast from.")
    origin = max(day for day, _store, _sku in keys)
    series = sorted({(store, sku) for _day, store, sku in keys})
    stores: list[str] = []
    skus: list[str] = []
    dates: list[date] = []
    for store_id, sku_id in series:
        for offset in range(1, horizon + 1):
            stores.append(store_id)
            skus.append(sku_id)
            dates.append(origin + step * offset)
    return pa.table(
        {
            "store_id": pa.array(stores, type=pa.string()),
            "sku_id": pa.array(skus, type=pa.string()),
            "date": pa.array(dates, type=pa.date32()),
        }
    )


def _latest_demand(frame: pa.Table) -> dict[tuple[str, str], tuple[date, float]]:
    required = {"date", "store_id", "sku_id", "units_sold"}
    if not required.issubset(set(frame.column_names)):
        raise InferenceError("The dataset is missing columns required for a forecast.")
    latest: dict[tuple[str, str], tuple[date, float]] = {}
    rows = zip(
        frame.column("date").to_pylist(),
        frame.column("store_id").to_pylist(),
        frame.column("sku_id").to_pylist(),
        frame.column("units_sold").to_pylist(),
        strict=True,
    )
    for day, store_id, sku_id, sold in rows:
        if type(day) is not date or not isinstance(store_id, str) or not isinstance(sku_id, str):
            continue
        if store_id == "" or sku_id == "":
            continue
        value = _demand(sold)
        if value is None:
            continue
        key = (store_id, sku_id)
        current = latest.get(key)
        if current is None or day > current[0]:
            latest[key] = (day, value)
    if not latest:
        raise InferenceError("The dataset has no observation dates to forecast from.")
    return latest


def _series_keys(frame: pa.Table) -> list[tuple[date, str, str]]:
    if not {"date", "store_id", "sku_id"}.issubset(set(frame.column_names)):
        raise InferenceError("The dataset is missing columns required for a forecast.")
    keys: list[tuple[date, str, str]] = []
    for day, store_id, sku_id in zip(
        frame.column("date").to_pylist(),
        frame.column("store_id").to_pylist(),
        frame.column("sku_id").to_pylist(),
        strict=True,
    ):
        if type(day) is not date or not isinstance(store_id, str) or not isinstance(sku_id, str):
            continue
        if store_id == "" or sku_id == "":
            continue
        keys.append((day, store_id, sku_id))
    return keys


def _demand(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise InferenceError("Forecast quantiles must be finite numbers.")
    return number


def _step(granularity: str) -> timedelta:
    if granularity == "day":
        return timedelta(days=1)
    if granularity == "week":
        return timedelta(weeks=1)
    raise InferenceError("Forecast granularity must be day or week.")


def _require_horizon(horizon: int) -> None:
    if type(horizon) is not int or isinstance(horizon, bool) or horizon < 1:
        raise InferenceError("Forecast horizon must be a positive number of periods.")
