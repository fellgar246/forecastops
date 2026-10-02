"""Demand profile and Monday-start weekly aggregations.

Aggregations roll the daily store-SKU grain up to SKU-week, category-week,
store-week, and region-week. ``units_sold`` is a sum. ``price`` is the
demand-weighted average, and it is null when the week's units sum to zero.
``promotion``, ``holiday``, and ``stockout`` are true when any day in the week
is true. Weeks start on Monday.

:func:`build_profile` summarizes a daily frame. Re-running it on the same
frame returns the same values. Pass a dataset directory to
:func:`profile_dataset` to write ``profile.json``.
"""

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
from numpy.typing import NDArray

from forecastops_contracts import (
    CATEGORY_COLUMNS,
    OBSERVATION_COLUMNS,
    SKU_COLUMNS,
    STORE_COLUMNS,
    ColumnSpec,
    LogicalType,
)
from forecastops_ml.data.quality import DatasetDimensions, stockout_rate

PROFILE_FILENAME = "profile.json"
_UNIX_EPOCH = date(1970, 1, 1)
# 1970-01-01 is Thursday, and Monday is weekday 0.
_EPOCH_WEEKDAY = 3
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

_ARROW: dict[LogicalType, pa.DataType] = {
    LogicalType.DATE: pa.date32(),
    LogicalType.STRING: pa.string(),
    LogicalType.INT64: pa.int64(),
    LogicalType.FLOAT64: pa.float64(),
    LogicalType.BOOLEAN: pa.bool_(),
}


@dataclass(frozen=True)
class DemandDistribution:
    """Distribution of daily ``units_sold``."""

    count: int
    mean: float
    median: float
    p90: float
    share_of_zeros: float

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this distribution."""

        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "p90": self.p90,
            "share_of_zeros": self.share_of_zeros,
        }


@dataclass(frozen=True)
class CategoryVolume:
    """Total ``units_sold`` for one category."""

    category_id: str
    category_name: str | None
    units_sold: int

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this category."""

        return {
            "category_id": self.category_id,
            "category_name": self.category_name,
            "units_sold": self.units_sold,
        }


@dataclass(frozen=True)
class WeekdaySeasonality:
    """Mean demand on one weekday divided by mean demand overall.

    ``weekday`` is Monday through Sunday. ``index`` is null when overall mean
    demand is zero.
    """

    weekday: str
    index: float | None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this weekday."""

        return {"index": self.index, "weekday": self.weekday}


@dataclass(frozen=True)
class MonthSeasonality:
    """Mean demand in one calendar month divided by mean demand overall.

    ``index`` is null when overall mean demand is zero.
    """

    month: int
    index: float | None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this month."""

        return {"index": self.index, "month": self.month}


@dataclass(frozen=True)
class PromotionLift:
    """Mean demand with a promotion and without one, for one category.

    ``lift`` is ``mean_with_promotion / mean_without_promotion``. It is null
    when either mean is missing or the unpromoted mean is zero.
    """

    category_id: str
    category_name: str | None
    mean_with_promotion: float | None
    mean_without_promotion: float | None
    lift: float | None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this category."""

        return {
            "category_id": self.category_id,
            "category_name": self.category_name,
            "lift": self.lift,
            "mean_with_promotion": self.mean_with_promotion,
            "mean_without_promotion": self.mean_without_promotion,
        }


@dataclass(frozen=True)
class SeriesLength:
    """How many daily observations each ``(store_id, sku_id)`` series has.

    ``median`` averages the two middle lengths when the series count is even.
    """

    minimum: int
    median: float
    maximum: int

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for these lengths."""

        return {"max": self.maximum, "median": self.median, "min": self.minimum}


@dataclass(frozen=True)
class DemandProfile:
    """Reproducible description of a daily demand table."""

    demand_distribution: DemandDistribution
    volume_by_category: tuple[CategoryVolume, ...]
    weekday_seasonality_index: tuple[WeekdaySeasonality, ...]
    month_seasonality_index: tuple[MonthSeasonality, ...]
    promotion_lift: tuple[PromotionLift, ...]
    stockout_prevalence: float
    sparsity: float
    series_length: SeriesLength

    def to_dict(self) -> dict[str, object]:
        """Return the JSON document for this profile."""

        return {
            "demand_distribution": self.demand_distribution.to_dict(),
            "month_seasonality_index": [item.to_dict() for item in self.month_seasonality_index],
            "promotion_lift": [item.to_dict() for item in self.promotion_lift],
            "series_length": self.series_length.to_dict(),
            "sparsity": self.sparsity,
            "stockout_prevalence": self.stockout_prevalence,
            "volume_by_category": [item.to_dict() for item in self.volume_by_category],
            "weekday_seasonality_index": [
                item.to_dict() for item in self.weekday_seasonality_index
            ],
        }


def aggregate_sku_week(frame: pa.Table) -> pa.Table:
    """Sum daily rows to one row per SKU and Monday-start week."""

    return _aggregate_week(frame, ("sku_id",))


def aggregate_category_week(frame: pa.Table) -> pa.Table:
    """Sum daily rows to one row per category and Monday-start week."""

    return _aggregate_week(frame, ("category_id",))


def aggregate_store_week(frame: pa.Table) -> pa.Table:
    """Sum daily rows to one row per store and Monday-start week."""

    return _aggregate_week(frame, ("store_id",))


def aggregate_region_week(frame: pa.Table, stores: pa.Table | None = None) -> pa.Table:
    """Sum daily rows to one row per region and Monday-start week.

    A complete ``store_region`` column on ``frame`` is the region. Otherwise
    ``stores`` supplies ``store_region`` for each ``store_id``.
    """

    return _aggregate_week(_with_store_region(frame, stores), ("store_region",))


def build_profile(frame: pa.Table, dimensions: DatasetDimensions) -> DemandProfile:
    """Describe daily demand, seasonality, promotions, stock-outs, and series length.

    ``share_of_zeros`` is the fraction of rows with ``units_sold`` equal to zero.
    ``sparsity`` is the fraction of ``(store_id, sku_id, date)`` series-days whose
    units sum to zero. ``stockout_prevalence`` is the fraction of rows with
    ``stockout`` true, or zero when that column is absent. Weekday indexes use
    Monday as the first day. The 90th percentile uses linear interpolation.
    """

    _require_columns(
        frame, ("date", "store_id", "sku_id", "category_id", "units_sold", "promotion")
    )
    if frame.num_rows == 0:
        raise ValueError("Cannot profile an observation table with no rows.")

    units = _int_column(frame, "units_sold")
    days = _epoch_days(frame)
    store_ids = _strings(frame, "store_id")
    sku_ids = _strings(frame, "sku_id")
    category_ids = _strings(frame, "category_id")
    promotion = _flags(frame, "promotion")
    names = _category_names(dimensions.categories)
    values = units.astype(np.float64)
    mean = float(np.mean(values))
    distribution = DemandDistribution(
        count=int(values.size),
        mean=mean,
        median=float(np.median(values)),
        p90=float(np.quantile(values, 0.9, method="linear")),
        share_of_zeros=float(np.sum(units == 0)) / float(values.size),
    )
    return DemandProfile(
        demand_distribution=distribution,
        volume_by_category=_category_volume(category_ids, units, names),
        weekday_seasonality_index=_weekday_seasonality(days, values, mean),
        month_seasonality_index=_month_seasonality(days, values, mean),
        promotion_lift=_promotion_lift(category_ids, units, promotion, names),
        stockout_prevalence=stockout_rate(frame),
        sparsity=_sparsity(store_ids, sku_ids, days, units),
        series_length=_series_length(store_ids, sku_ids),
    )


def load_dataset(directory: Path) -> tuple[pa.Table, DatasetDimensions]:
    """Read observations and dimensions from a generated dataset directory.

    Parquet is preferred when both Parquet and CSV are present.
    """

    if not directory.is_dir():
        raise ValueError(f"Dataset path {directory} is not a directory.")
    return (
        _read_table(directory, "observations", OBSERVATION_COLUMNS),
        DatasetDimensions(
            stores=_read_table(directory, "stores", STORE_COLUMNS),
            skus=_read_table(directory, "skus", SKU_COLUMNS),
            categories=_read_table(directory, "categories", CATEGORY_COLUMNS),
        ),
    )


def write_profile(profile: DemandProfile, directory: Path) -> Path:
    """Write ``profile.json`` in ``directory`` and return that path."""

    if directory.exists() and not directory.is_dir():
        raise ValueError(f"Profile path {directory} is a file. Pass a directory.")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / PROFILE_FILENAME
    destination.write_text(_profile_json(profile), encoding="utf-8")
    return destination


def profile_dataset(directory: Path) -> Path:
    """Read a generated dataset directory and write ``profile.json`` beside it."""

    frame, dimensions = load_dataset(directory)
    return write_profile(build_profile(frame, dimensions), directory)


def _profile_json(profile: DemandProfile) -> str:
    return json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n"


def _aggregate_week(frame: pa.Table, keys: tuple[str, ...]) -> pa.Table:
    required = ("date", "units_sold", "price", "promotion", "holiday", "stockout", *keys)
    _require_columns(frame, required)
    if frame.num_rows == 0:
        return _empty_week_table(keys)

    days = _epoch_days(frame)
    week_days = _week_start_days(days)
    units = _int_column(frame, "units_sold")
    price = _float_column(frame, "price")
    grain = [_strings(frame, key) for key in keys]
    order, starts = _group((*grain, week_days))
    ordered_units = units[order]
    unit_sums = np.add.reduceat(ordered_units, starts)
    weighted = np.add.reduceat(price[order] * ordered_units.astype(np.float64), starts)
    labels = [_labels_at(values, order, starts) for values in grain]
    prices = _weighted_prices(unit_sums, weighted)
    flag_columns = [
        _any_true(_flags(frame, name), order, starts)
        for name in ("promotion", "holiday", "stockout")
    ]
    week_values = week_days[order][starts]
    arrays: list[pa.Array] = [pa.array(column, type=pa.string()) for column in labels]
    arrays.extend(
        [
            pa.array(week_values.astype(np.int32), type=pa.date32()),
            pa.array([int(value) for value in unit_sums.tolist()], type=pa.int64()),
            pa.array(prices, type=pa.float64()),
            *(pa.array(column, type=pa.bool_()) for column in flag_columns),
        ]
    )
    return pa.Table.from_arrays(arrays, schema=_week_schema(keys))


def _weighted_prices(unit_sums: I64, weighted: F64) -> list[float | None]:
    prices: list[float | None] = []
    for total, weight in zip(unit_sums.tolist(), weighted.tolist(), strict=True):
        if total == 0:
            prices.append(None)
        else:
            prices.append(float(weight) / float(total))
    return prices


def _any_true(flags: BoolArray, order: I64, starts: I64) -> list[bool]:
    reduced = np.maximum.reduceat(flags[order].astype(np.int8), starts)
    return [bool(value) for value in reduced.tolist()]


def _labels_at(values: np.ndarray, order: I64, starts: I64) -> list[str]:
    picked = values[order][starts]
    return [str(value) for value in picked.tolist()]


def _with_store_region(frame: pa.Table, stores: pa.Table | None) -> pa.Table:
    if _region_column_is_complete(frame):
        return frame
    if stores is None:
        raise ValueError(
            "Region-week aggregation needs a complete store_region column or a store dimension."
        )
    regions = _store_regions(stores)
    resolved = _map_store_regions(_strings(frame, "store_id"), regions)
    column = pa.array(resolved, type=pa.string())
    if "store_region" in frame.column_names:
        frame = frame.drop(["store_region"])
    return frame.append_column("store_region", column)


def _region_column_is_complete(frame: pa.Table) -> bool:
    if "store_region" not in frame.column_names:
        return False
    column = _as_string(frame.column("store_region"), "store_region")
    if column.null_count:
        return False
    blanks = pc.equal(column, "")
    return not bool(pc.any(blanks).as_py())


def _map_store_regions(store_ids: np.ndarray, regions: dict[str, str]) -> list[str]:
    unique_ids, inverse = np.unique(store_ids, return_inverse=True)
    mapped: list[str] = []
    for store_id in unique_ids.tolist():
        region = regions.get(str(store_id))
        if region is None:
            raise ValueError(f"store_id {store_id} is missing from the store dimension.")
        mapped.append(region)
    looked_up = np.asarray(mapped, dtype=object)[np.asarray(inverse)]
    return [str(region) for region in looked_up.tolist()]


def _category_volume(
    category_ids: np.ndarray,
    units: I64,
    names: dict[str, str],
) -> tuple[CategoryVolume, ...]:
    labels, sums, _sizes = _reduce_groups((category_ids,), units)
    return tuple(
        CategoryVolume(
            category_id=category_id,
            category_name=names.get(category_id),
            units_sold=total,
        )
        for category_id, total in zip(labels[0], sums, strict=True)
    )


def _weekday_seasonality(
    days: I64, values: F64, overall_mean: float
) -> tuple[WeekdaySeasonality, ...]:
    weekday = (days + _EPOCH_WEEKDAY) % 7
    rows: list[WeekdaySeasonality] = []
    for label, total, size in _iter_groups((weekday,), values):
        rows.append(
            WeekdaySeasonality(
                weekday=_WEEKDAYS[int(label)],
                index=_index(float(total) / float(size), overall_mean),
            )
        )
    return tuple(rows)


def _month_seasonality(days: I64, values: F64, overall_mean: float) -> tuple[MonthSeasonality, ...]:
    rows: list[MonthSeasonality] = []
    for label, total, size in _iter_groups((_months(days),), values):
        rows.append(
            MonthSeasonality(
                month=int(label),
                index=_index(float(total) / float(size), overall_mean),
            )
        )
    return tuple(rows)


def _promotion_lift(
    category_ids: np.ndarray,
    units: I64,
    promotion: BoolArray,
    names: dict[str, str],
) -> tuple[PromotionLift, ...]:
    lifts: list[PromotionLift] = []
    for category_id in sorted({str(value) for value in category_ids.tolist()}):
        selected = np.asarray(category_ids == category_id)
        mean_with = _masked_mean(units, selected & promotion)
        mean_without = _masked_mean(units, selected & ~promotion)
        lifts.append(
            PromotionLift(
                category_id=category_id,
                category_name=names.get(category_id),
                mean_with_promotion=mean_with,
                mean_without_promotion=mean_without,
                lift=_ratio(mean_with, mean_without),
            )
        )
    return tuple(lifts)


def _sparsity(store_ids: np.ndarray, sku_ids: np.ndarray, days: I64, units: I64) -> float:
    _labels, sums, _sizes = _reduce_groups((store_ids, sku_ids, days), units)
    if len(sums) == 0:
        return 0.0
    return float(np.sum(np.asarray(sums) == 0)) / float(len(sums))


def _series_length(store_ids: np.ndarray, sku_ids: np.ndarray) -> SeriesLength:
    _labels, _sums, sizes = _reduce_groups(
        (store_ids, sku_ids), np.zeros(len(store_ids), dtype=np.int64)
    )
    if len(sizes) == 0:
        raise ValueError("Cannot profile an observation table with no series.")
    lengths = np.asarray(sizes, dtype=np.float64)
    return SeriesLength(
        minimum=int(np.min(lengths)),
        median=float(np.median(lengths)),
        maximum=int(np.max(lengths)),
    )


def _index(group_mean: float, overall_mean: float) -> float | None:
    if overall_mean == 0.0:
        return None
    return group_mean / overall_mean


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _masked_mean(units: I64, mask: BoolArray) -> float | None:
    if not bool(np.any(mask)):
        return None
    return float(np.mean(units[mask]))


def _iter_groups(
    keys: Sequence[np.ndarray],
    values: F64,
) -> list[tuple[str | int, float, int]]:
    order, starts, sizes = _group_bounds(keys)
    ordered = values[order]
    totals = np.add.reduceat(ordered, starts)
    grouped: list[tuple[str | int, float, int]] = []
    labels = keys[0][order][starts]
    for label, total, size in zip(labels.tolist(), totals.tolist(), sizes.tolist(), strict=True):
        rendered: str | int = int(label) if isinstance(label, (int, np.integer)) else str(label)
        grouped.append((rendered, float(total), int(size)))
    return grouped


def _reduce_groups(
    keys: Sequence[np.ndarray],
    units: I64,
) -> tuple[list[list[str]], list[int], list[int]]:
    order, starts, sizes = _group_bounds(keys)
    sums = np.add.reduceat(units[order], starts)
    labels = [_labels_at(values, order, starts) for values in keys]
    return labels, [int(value) for value in sums.tolist()], [int(value) for value in sizes.tolist()]


def _group(keys: Sequence[np.ndarray]) -> tuple[I64, I64]:
    order, starts, _sizes = _group_bounds(keys)
    return order, starts


def _group_bounds(keys: Sequence[np.ndarray]) -> tuple[I64, I64, I64]:
    if not keys:
        raise ValueError("Aggregation needs at least one key.")
    n = len(keys[0])
    if any(len(key) != n for key in keys):
        raise ValueError("Aggregation keys must have the same length.")
    if n == 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty, empty
    codes = [_factorize(key) for key in keys]
    order = np.asarray(np.lexsort(tuple(reversed(codes))), dtype=np.int64)
    ordered = [code[order] for code in codes]
    change = np.zeros(n, dtype=np.bool_)
    change[0] = True
    for code in ordered:
        change[1:] |= code[1:] != code[:-1]
    starts = np.asarray(np.flatnonzero(change), dtype=np.int64)
    sizes = np.empty(len(starts), dtype=np.int64)
    sizes[:-1] = starts[1:] - starts[:-1]
    sizes[-1] = n - int(starts[-1])
    return order, starts, sizes


def _factorize(values: np.ndarray) -> I64:
    _unique, codes = np.unique(values, return_inverse=True)
    return np.asarray(codes, dtype=np.int64)


def _week_start_days(epoch_days: I64) -> I64:
    weekday = (epoch_days + _EPOCH_WEEKDAY) % 7
    return epoch_days - weekday


def _months(epoch_days: I64) -> I64:
    dates = pa.array(epoch_days.astype(np.int32, copy=False), type=pa.date32())
    months = pc.month(dates)
    return np.asarray(months.to_numpy(zero_copy_only=False), dtype=np.int64)


def _week_schema(keys: tuple[str, ...]) -> pa.Schema:
    fields = [pa.field(key, pa.string()) for key in keys]
    fields.extend(
        [
            pa.field("week_start", pa.date32()),
            pa.field("units_sold", pa.int64()),
            pa.field("price", pa.float64()),
            pa.field("promotion", pa.bool_()),
            pa.field("holiday", pa.bool_()),
            pa.field("stockout", pa.bool_()),
        ]
    )
    return pa.schema(fields)


def _empty_week_table(keys: tuple[str, ...]) -> pa.Table:
    arrays: list[pa.Array] = [pa.array([], type=pa.string()) for _key in keys]
    arrays.extend(
        [
            pa.array([], type=pa.date32()),
            pa.array([], type=pa.int64()),
            pa.array([], type=pa.float64()),
            pa.array([], type=pa.bool_()),
            pa.array([], type=pa.bool_()),
            pa.array([], type=pa.bool_()),
        ]
    )
    return pa.Table.from_arrays(arrays, schema=_week_schema(keys))


def _category_names(categories: pa.Table) -> dict[str, str]:
    _require_columns(categories, ("category_id", "category_name"))
    if categories.num_rows == 0:
        return {}
    category_ids = _strings(categories, "category_id")
    category_names = _strings(categories, "category_name")
    if len(set(category_ids.tolist())) != len(category_ids):
        raise ValueError("Category dimension has duplicate category_id values.")
    return {
        str(category_id): str(name)
        for category_id, name in zip(category_ids.tolist(), category_names.tolist(), strict=True)
    }


def _store_regions(stores: pa.Table) -> dict[str, str]:
    _require_columns(stores, ("store_id", "store_region"))
    store_ids = _strings(stores, "store_id")
    regions = _strings(stores, "store_region")
    if len(set(store_ids.tolist())) != len(store_ids):
        raise ValueError("Store dimension has duplicate store_id values.")
    return {
        str(store_id): str(region)
        for store_id, region in zip(store_ids.tolist(), regions.tolist(), strict=True)
    }


def _read_table(directory: Path, stem: str, columns: Sequence[ColumnSpec]) -> pa.Table:
    parquet_path = directory / f"{stem}.parquet"
    csv_path = directory / f"{stem}.csv"
    if parquet_path.is_file():
        return pq.read_table(parquet_path)
    if csv_path.is_file():
        return _read_csv(csv_path, columns)
    raise ValueError(f"Dataset directory {directory} is missing {stem}.parquet and {stem}.csv.")


def _read_csv(path: Path, columns: Sequence[ColumnSpec]) -> pa.Table:
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    expected = {column.name: _ARROW[column.logical_type] for column in columns}
    unknown = [name for name in header if name not in expected]
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"{path.name} has unexpected columns: {joined}.")
    column_types = {name: expected[name] for name in header}
    return pa_csv.read_csv(
        path,
        convert_options=pa_csv.ConvertOptions(column_types=column_types, strings_can_be_null=True),
    )


def _require_columns(frame: pa.Table, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in frame.column_names]
    if missing:
        found = ", ".join(missing)
        raise ValueError(f"Table is missing {found}.")


def _epoch_days(frame: pa.Table) -> I64:
    column = _as_date(frame.column("date"))
    if column.null_count:
        raise ValueError("Column date contains nulls.")
    days = column.cast(pa.int32()).to_numpy(zero_copy_only=False)
    return np.asarray(days, dtype=np.int64)


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


def _int_column(frame: pa.Table, name: str) -> I64:
    column = frame.column(name)
    if not pa.types.is_integer(column.type):
        raise ValueError(f"Column {name} has type {column.type}, expected an integer.")
    if column.null_count:
        raise ValueError(f"Column {name} contains nulls.")
    values = column.cast(pa.int64()).to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.int64)


def _float_column(frame: pa.Table, name: str) -> F64:
    column = frame.column(name)
    if not (pa.types.is_floating(column.type) or pa.types.is_integer(column.type)):
        raise ValueError(f"Column {name} has type {column.type}, expected a number.")
    if column.null_count:
        raise ValueError(f"Column {name} contains nulls.")
    values = column.cast(pa.float64()).to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float64)


def _flags(frame: pa.Table, name: str) -> BoolArray:
    column = frame.column(name)
    if not pa.types.is_boolean(column.type):
        raise ValueError(f"Column {name} has type {column.type}, expected a boolean.")
    filled = column.fill_null(False)
    return np.asarray(filled.to_numpy(zero_copy_only=False), dtype=np.bool_)


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
