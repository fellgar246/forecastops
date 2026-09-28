"""Convert retail observations into DeepAR train and test channels.

The default grain is week. Each series is one JSON line with ``start``,
``target``, ``cat`` (store, then category), and ``dynamic_feat`` for
promotion and holiday. The train target stops ``prediction_length`` periods
before the test target. Promotion and holiday for that held-out window stay
on the train channel, because they are known for the forecast horizon.
"""

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import cast

import pyarrow as pa

from forecastops_ml.evaluation.backtest import make_series_id
from forecastops_ml.features.week import aggregate_store_sku_week

DEFAULT_GRAIN = "week"
DEFAULT_PREDICTION_LENGTH = 2
_GRAIN_STEP = {"day": timedelta(days=1), "week": timedelta(days=7)}
_FREQUENCIES = {"day": "1D", "week": "1W"}


@dataclass(frozen=True)
class DeepARRecord:
    """One series in a DeepAR channel."""

    series_id: str
    start: date
    target: tuple[float, ...]
    cat: tuple[int, int]
    dynamic_feat: tuple[tuple[float, ...], tuple[float, ...]]
    target_end: date

    def json_object(self) -> dict[str, object]:
        """Return the JSON line fields DeepAR reads."""

        return {
            "start": f"{self.start.isoformat()} 00:00:00",
            "target": list(self.target),
            "cat": [self.cat[0], self.cat[1]],
            "dynamic_feat": [list(self.dynamic_feat[0]), list(self.dynamic_feat[1])],
        }


@dataclass(frozen=True)
class DeepARChannels:
    """Train and test JSON-lines channels for one DeepAR job."""

    train: tuple[DeepARRecord, ...]
    test: tuple[DeepARRecord, ...]
    grain: str
    frequency: str
    prediction_length: int
    store_cardinality: int
    category_cardinality: int

    def jsonl(self, channel: str) -> str:
        """Return JSON lines for ``train`` or ``test``."""

        records = _channel_records(self, channel)
        lines = [
            json.dumps(record.json_object(), allow_nan=False, separators=(",", ":"))
            for record in records
        ]
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class _Series:
    series_id: str
    store_id: str
    category_id: str
    dates: tuple[date, ...]
    target: tuple[float, ...]
    promotion: tuple[float, ...]
    holiday: tuple[float, ...]


def export_deepar_channels(
    frame: pa.Table,
    *,
    grain: str = DEFAULT_GRAIN,
    prediction_length: int = DEFAULT_PREDICTION_LENGTH,
    horizon_plan: pa.Table | None = None,
) -> DeepARChannels:
    """Export ``frame`` as DeepAR train and test channels.

    ``grain`` is ``week`` or ``day``. Week rolls the daily contract up to
    Monday-start weeks. ``horizon_plan`` supplies promotion and holiday for
    ``prediction_length`` periods after the last observation. Those values
    are appended to the test channel only.
    """

    if grain not in _GRAIN_STEP:
        raise ValueError("grain must be day or week.")
    if type(prediction_length) is not int or prediction_length < 1:
        raise ValueError("prediction_length must be a positive integer.")
    history = _prepare_history(frame, grain)
    series = _series_from_frame(history, grain)
    if len(series) < 2:
        count = len(series)
        raise ValueError(f"DeepAR training needs at least two series. This slice has {count}.")
    minimum = 3 * prediction_length
    for item in series:
        if len(item.dates) < minimum:
            kind = "week" if grain == "week" else "day"
            raise ValueError(
                f"Series {item.series_id} has {len(item.dates)} {kind} periods. "
                f"A horizon of {prediction_length} needs at least {minimum}."
            )
    plan = (
        None
        if horizon_plan is None
        else _plan_by_series(horizon_plan, series, grain, prediction_length)
    )
    store_codes, category_codes = _codes(series)
    train = tuple(
        _train_record(item, prediction_length, store_codes, category_codes) for item in series
    )
    test = tuple(
        _test_record(item, prediction_length, store_codes, category_codes, plan) for item in series
    )
    for trained, held_out in zip(train, test, strict=True):
        if trained.target_end >= held_out.target_end:
            raise ValueError(
                f"Train channel for {trained.series_id} must end before the test channel."
            )
        _require_feature_length(trained, len(trained.target) + prediction_length)
        test_length = (
            len(held_out.target) if plan is None else len(held_out.target) + prediction_length
        )
        _require_feature_length(held_out, test_length)
    return DeepARChannels(
        train=train,
        test=test,
        grain=grain,
        frequency=_FREQUENCIES[grain],
        prediction_length=prediction_length,
        store_cardinality=len(store_codes),
        category_cardinality=len(category_codes),
    )


def period_step(grain: str) -> timedelta:
    """Return the spacing between periods for ``day`` or ``week``."""

    try:
        return _GRAIN_STEP[grain]
    except KeyError as exc:
        raise ValueError("grain must be day or week.") from exc


def _require_feature_length(record: DeepARRecord, expected: int) -> None:
    promotion, holiday = record.dynamic_feat
    if len(promotion) != expected or len(holiday) != expected:
        raise ValueError(
            f"Series {record.series_id} dynamic features must cover {expected} periods."
        )


def _channel_records(channels: DeepARChannels, channel: str) -> tuple[DeepARRecord, ...]:
    if channel == "train":
        return channels.train
    if channel == "test":
        return channels.test
    raise ValueError("channel must be train or test.")


def _prepare_history(frame: pa.Table, grain: str) -> pa.Table:
    if not isinstance(frame, pa.Table):
        raise ValueError("Observation frame must be a pyarrow table.")
    required = ("date", "store_id", "sku_id", "category_id", "units_sold", "promotion", "holiday")
    missing = [name for name in required if name not in frame.column_names]
    if missing:
        found = ", ".join(missing)
        raise ValueError(f"Observation table is missing {found}.")
    if frame.num_rows == 0:
        raise ValueError("Observation table has no rows.")
    _reject_duplicate_keys(frame)
    if grain == "week":
        return aggregate_store_sku_week(frame)
    return frame


def _prepare_plan(frame: pa.Table, grain: str) -> pa.Table:
    if not isinstance(frame, pa.Table):
        raise ValueError("Horizon plan must be a pyarrow table.")
    required = ("date", "store_id", "sku_id", "promotion", "holiday")
    missing = [name for name in required if name not in frame.column_names]
    if missing:
        found = ", ".join(missing)
        raise ValueError(f"Horizon plan is missing {found}.")
    if frame.num_rows == 0:
        raise ValueError("Horizon plan has no rows.")
    _reject_duplicate_keys(frame)
    padded = frame
    if "units_sold" not in padded.column_names:
        zeros = pa.array([0] * padded.num_rows, type=pa.int64())
        padded = padded.append_column("units_sold", zeros)
    if "price" not in padded.column_names:
        prices = pa.array([0.0] * padded.num_rows, type=pa.float64())
        padded = padded.append_column("price", prices)
    if "category_id" not in padded.column_names:
        labels = pa.array(["plan"] * padded.num_rows, type=pa.string())
        padded = padded.append_column("category_id", labels)
    if grain == "day":
        return padded
    return aggregate_store_sku_week(padded)


def _series_from_frame(frame: pa.Table, grain: str) -> tuple[_Series, ...]:
    dates = _dates(frame)
    stores = _strings(frame, "store_id")
    skus = _strings(frame, "sku_id")
    categories = _strings(frame, "category_id")
    units = _numbers(frame, "units_sold")
    promotions = _flags(frame, "promotion")
    holidays = _flags(frame, "holiday")
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, store_id in enumerate(stores):
        grouped[(store_id, skus[index])].append(index)
    series: list[_Series] = []
    for store_id, sku_id in sorted(grouped):
        indexes = sorted(grouped[(store_id, sku_id)], key=lambda index: dates[index])
        series_id = make_series_id(store_id, sku_id)
        category_ids = {categories[index] for index in indexes}
        if len(category_ids) != 1:
            raise ValueError(f"Series {series_id} has more than one category.")
        ordered_dates = tuple(dates[index] for index in indexes)
        _require_contiguous(series_id, ordered_dates, grain)
        series.append(
            _Series(
                series_id=series_id,
                store_id=store_id,
                category_id=next(iter(category_ids)),
                dates=ordered_dates,
                target=tuple(units[index] for index in indexes),
                promotion=tuple(promotions[index] for index in indexes),
                holiday=tuple(holidays[index] for index in indexes),
            )
        )
    return tuple(series)


def _plan_by_series(
    frame: pa.Table,
    series: Sequence[_Series],
    grain: str,
    prediction_length: int,
) -> dict[str, _Series]:
    prepared = _series_from_frame(_prepare_plan(frame, grain), grain)
    found = {item.series_id: item for item in prepared}
    expected = {item.series_id for item in series}
    missing = sorted(expected - set(found))
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"The horizon plan is missing series {names}.")
    extra = sorted(set(found) - expected)
    if extra:
        names = ", ".join(extra)
        raise ValueError(f"The horizon plan contains unknown series {names}.")
    step = _GRAIN_STEP[grain]
    for item in series:
        plan = found[item.series_id]
        expected_start = item.dates[-1] + step
        if plan.dates[0] != expected_start or len(plan.dates) != prediction_length:
            raise ValueError(
                f"The horizon plan for {item.series_id} must cover {prediction_length} "
                f"periods starting {expected_start.isoformat()}."
            )
    return found


def _codes(series: Sequence[_Series]) -> tuple[dict[str, int], dict[str, int]]:
    stores = {item.store_id for item in series}
    categories = {item.category_id for item in series}
    store_codes = {name: index for index, name in enumerate(sorted(stores))}
    category_codes = {name: index for index, name in enumerate(sorted(categories))}
    return store_codes, category_codes


def _train_record(
    series: _Series,
    prediction_length: int,
    store_codes: Mapping[str, int],
    category_codes: Mapping[str, int],
) -> DeepARRecord:
    target = series.target[:-prediction_length]
    end_index = len(target) - 1
    return DeepARRecord(
        series_id=series.series_id,
        start=series.dates[0],
        target=target,
        cat=(store_codes[series.store_id], category_codes[series.category_id]),
        dynamic_feat=(series.promotion, series.holiday),
        target_end=series.dates[end_index],
    )


def _test_record(
    series: _Series,
    prediction_length: int,
    store_codes: Mapping[str, int],
    category_codes: Mapping[str, int],
    plan: Mapping[str, _Series] | None,
) -> DeepARRecord:
    promotion = series.promotion
    holiday = series.holiday
    if plan is not None:
        extra = plan[series.series_id]
        if len(extra.promotion) != prediction_length:
            raise ValueError(
                f"The horizon plan for {series.series_id} must cover {prediction_length} periods."
            )
        promotion = series.promotion + extra.promotion
        holiday = series.holiday + extra.holiday
    return DeepARRecord(
        series_id=series.series_id,
        start=series.dates[0],
        target=series.target,
        cat=(store_codes[series.store_id], category_codes[series.category_id]),
        dynamic_feat=(promotion, holiday),
        target_end=series.dates[-1],
    )


def _require_contiguous(series_id: str, dates: tuple[date, ...], grain: str) -> None:
    step = _GRAIN_STEP[grain]
    for previous, current in zip(dates, dates[1:], strict=False):
        if current - previous != step:
            raise ValueError(
                f"Series {series_id} skips from {previous.isoformat()} to {current.isoformat()}."
            )


def _reject_duplicate_keys(frame: pa.Table) -> None:
    dates = _dates(frame)
    stores = _strings(frame, "store_id")
    skus = _strings(frame, "sku_id")
    keys = list(zip(dates, stores, skus, strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError("Observation table repeats a (date, store_id, sku_id) key.")


def _dates(frame: pa.Table) -> list[date]:
    column = frame.column("date")
    if not pa.types.is_date32(column.type):
        try:
            column = column.cast(pa.date32())
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            raise ValueError(
                f"Column date has type {frame.column('date').type}, expected a date."
            ) from exc
    if column.null_count:
        raise ValueError("Column date contains nulls.")
    values = column.to_pylist()
    if any(type(value) is not date for value in values):
        raise ValueError("Column date must contain calendar dates.")
    return cast(list[date], values)


def _strings(frame: pa.Table, name: str) -> list[str]:
    column = frame.column(name)
    if column.null_count:
        raise ValueError(f"Column {name} contains nulls.")
    values = [str(value) for value in column.to_pylist()]
    if any(value == "" for value in values):
        raise ValueError(f"Column {name} contains a blank value.")
    return values


def _numbers(frame: pa.Table, name: str) -> list[float]:
    column = frame.column(name)
    if not (pa.types.is_integer(column.type) or pa.types.is_floating(column.type)):
        raise ValueError(f"Column {name} has type {column.type}, expected a number.")
    if column.null_count:
        raise ValueError(f"Column {name} contains nulls.")
    values = [float(value) for value in column.to_pylist()]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"Column {name} contains a non-finite value.")
    return values


def _flags(frame: pa.Table, name: str) -> list[float]:
    column = frame.column(name)
    if not pa.types.is_boolean(column.type):
        raise ValueError(f"Column {name} has type {column.type}, expected a boolean.")
    if column.null_count:
        raise ValueError(f"Column {name} contains nulls.")
    return [1.0 if value else 0.0 for value in column.to_pylist()]
