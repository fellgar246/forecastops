"""Store-SKU weeks for the gradient-boosted model.

Weeks start on Monday. ``units_sold`` is the sum of the daily values.
``price`` is the demand-weighted average and is null when the week's units
sum to zero. ``discount_pct`` uses the same weight when units are positive,
and the unweighted mean of known discounts when they are not. ``promotion``
and ``holiday`` are true when any day in the week is true. Stock columns are
left off the weekly frame.
"""

from collections import defaultdict
from datetime import date, timedelta
from typing import cast

import pyarrow as pa


def aggregate_store_sku_week(frame: pa.Table) -> pa.Table:
    """Roll daily store-SKU rows up to one Monday-start week row."""

    if not isinstance(frame, pa.Table):
        raise ValueError("Observation frame must be a pyarrow table.")
    required = ("date", "store_id", "sku_id", "category_id", "units_sold", "price")
    missing = [name for name in required if name not in frame.column_names]
    if missing:
        found = ", ".join(missing)
        raise ValueError(f"Observation table is missing {found}.")
    if frame.num_rows == 0:
        raise ValueError("Observation table has no rows.")

    days = _dates(frame)
    stores = _strings(frame, "store_id")
    skus = _strings(frame, "sku_id")
    categories = _strings(frame, "category_id")
    units = _ints(frame, "units_sold")
    prices = _floats(frame, "price")
    discounts = _optional_floats(frame, "discount_pct")
    promotions = _flags(frame, "promotion")
    holidays = _flags(frame, "holiday")

    grouped: dict[tuple[str, str, date], list[int]] = defaultdict(list)
    for index, day in enumerate(days):
        grouped[(stores[index], skus[index], day - timedelta(days=day.weekday()))].append(index)

    ordered = sorted(grouped)
    out_stores: list[str] = []
    out_skus: list[str] = []
    out_categories: list[str] = []
    out_days: list[date] = []
    out_units: list[int] = []
    out_prices: list[float | None] = []
    out_discounts: list[float | None] = []
    out_promotions: list[bool] = []
    out_holidays: list[bool] = []
    for store_id, sku_id, week_start in ordered:
        indexes = grouped[(store_id, sku_id, week_start)]
        category_ids = {categories[index] for index in indexes}
        if len(category_ids) != 1:
            stamp = week_start.isoformat()
            raise ValueError(
                f"Store {store_id} SKU {sku_id} has more than one category in week {stamp}."
            )
        total = sum(units[index] for index in indexes)
        out_stores.append(store_id)
        out_skus.append(sku_id)
        out_categories.append(next(iter(category_ids)))
        out_days.append(week_start)
        out_units.append(total)
        out_prices.append(_weighted_mean(indexes, total, prices, units))
        out_discounts.append(_discount_mean(indexes, total, discounts, units))
        out_promotions.append(any(promotions[index] for index in indexes))
        out_holidays.append(any(holidays[index] for index in indexes))

    return pa.table(
        {
            "date": pa.array(out_days, type=pa.date32()),
            "store_id": pa.array(out_stores, type=pa.string()),
            "sku_id": pa.array(out_skus, type=pa.string()),
            "category_id": pa.array(out_categories, type=pa.string()),
            "units_sold": pa.array(out_units, type=pa.int64()),
            "price": pa.array(out_prices, type=pa.float64()),
            "discount_pct": pa.array(out_discounts, type=pa.float64()),
            "promotion": pa.array(out_promotions, type=pa.bool_()),
            "holiday": pa.array(out_holidays, type=pa.bool_()),
        }
    )


def _weighted_mean(
    indexes: list[int],
    total: int,
    values: list[float],
    units: list[int],
) -> float | None:
    if total == 0:
        return None
    weighted = sum(values[index] * units[index] for index in indexes)
    return float(weighted) / float(total)


def _discount_mean(
    indexes: list[int],
    total: int,
    discounts: list[float | None],
    units: list[int],
) -> float | None:
    known: list[tuple[int, float]] = []
    for index in indexes:
        value = discounts[index]
        if value is not None:
            known.append((index, value))
    if not known:
        return None
    values = [float(value) for _index, value in known]
    if total == 0:
        return sum(values) / float(len(values))
    weighted = sum(value * units[index] for index, value in known)
    weight = sum(units[index] for index, _value in known)
    if weight == 0:
        return sum(values) / float(len(values))
    return weighted / float(weight)


def _dates(frame: pa.Table) -> list[date]:
    column = frame.column("date")
    if not pa.types.is_date32(column.type):
        try:
            column = column.cast(pa.date32())
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            found = frame.column("date").type
            raise ValueError(f"Column date has type {found}, expected a date.") from exc
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


def _ints(frame: pa.Table, name: str) -> list[int]:
    column = frame.column(name)
    if not pa.types.is_integer(column.type):
        raise ValueError(f"Column {name} has type {column.type}, expected an integer.")
    if column.null_count:
        raise ValueError(f"Column {name} contains nulls.")
    return [int(value) for value in column.to_pylist()]


def _floats(frame: pa.Table, name: str) -> list[float]:
    column = frame.column(name)
    if not (pa.types.is_floating(column.type) or pa.types.is_integer(column.type)):
        raise ValueError(f"Column {name} has type {column.type}, expected a number.")
    if column.null_count:
        raise ValueError(f"Column {name} contains nulls.")
    return [float(value) for value in column.to_pylist()]


def _optional_floats(frame: pa.Table, name: str) -> list[float | None]:
    if name not in frame.column_names:
        return [None] * frame.num_rows
    column = frame.column(name)
    values: list[float | None] = []
    for value in column.to_pylist():
        values.append(None if value is None else float(value))
    return values


def _flags(frame: pa.Table, name: str) -> list[bool]:
    if name not in frame.column_names:
        return [False] * frame.num_rows
    column = frame.column(name)
    if not pa.types.is_boolean(column.type):
        raise ValueError(f"Column {name} has type {column.type}, expected a boolean.")
    filled = column.fill_null(False)
    return [bool(value) for value in filled.to_pylist()]
