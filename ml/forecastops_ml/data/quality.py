"""Quality checks for a retail demand dataset.

Blocking checks reject a table that breaks the daily grain or contains values
training must not see. Advisory checks record stock-outs, unexpected
categories, and zero demand, and they leave the status ``valid``.

:func:`validate_dataset` returns the report. Pass a directory to also write
``quality_report.json`` beside the dataset.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from forecastops_contracts import REQUIRED_COLUMNS

QUALITY_REPORT_FILENAME = "quality_report.json"
_UNIX_EPOCH = date(1970, 1, 1)
_GRAIN = ("date", "store_id", "sku_id")
_STRING_IDS = ("store_id", "sku_id", "category_id")

I64 = NDArray[np.int64]
F64 = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class BlockingCode(StrEnum):
    """Machine-readable codes for findings that reject a dataset."""

    DUPLICATE_GRAIN = "duplicate_grain"
    MISSING_REQUIRED = "missing_required"
    NEGATIVE_UNITS_SOLD = "negative_units_sold"
    NEGATIVE_PRICE = "negative_price"
    INVALID_PROMOTION_DISCOUNT = "invalid_promotion_discount"
    UNKNOWN_STORE = "unknown_store"
    UNKNOWN_SKU = "unknown_sku"
    CALENDAR_GAP = "calendar_gap"
    STALE_DATASET = "stale_dataset"
    EXTREME_OUTLIER = "extreme_outlier"


class AdvisoryCode(StrEnum):
    """Machine-readable codes for findings that do not reject a dataset."""

    STOCKOUT_RATE = "stockout_rate"
    UNEXPECTED_CATEGORY = "unexpected_category"
    ZERO_DEMAND_FRACTION = "zero_demand_fraction"


@dataclass(frozen=True)
class Finding:
    """One quality finding. ``code`` is the stable name; ``message`` is for people."""

    code: str
    count: int
    message: str
    rate: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this finding."""

        document: dict[str, object] = {
            "code": self.code,
            "count": self.count,
            "message": self.message,
        }
        if self.rate is not None:
            document["rate"] = self.rate
        return document


@dataclass(frozen=True)
class QualityReport:
    """Validation result a later API can store without reading prose.

    ``missing_value_rate`` is the fraction of required cells that are null or
    blank. ``duplicate_rate`` is the fraction of rows that repeat an earlier
    ``(date, store_id, sku_id)`` key. ``stockout_rate`` is the fraction of rows
    with ``stockout`` true, or zero when that column is absent.
    """

    status: Literal["valid", "invalid"]
    blocking: tuple[Finding, ...]
    advisory: tuple[Finding, ...]
    missing_value_rate: float
    duplicate_rate: float
    stockout_rate: float

    def __post_init__(self) -> None:
        expected: Literal["valid", "invalid"] = "invalid" if self.blocking else "valid"
        if self.status != expected:
            raise ValueError("status must be invalid exactly when blocking findings exist.")

    def to_dict(self) -> dict[str, object]:
        """Return the JSON document for this report."""

        return {
            "advisory": [item.to_dict() for item in self.advisory],
            "blocking": [item.to_dict() for item in self.blocking],
            "rates": {
                "duplicates": self.duplicate_rate,
                "missing_values": self.missing_value_rate,
                "stockouts": self.stockout_rate,
            },
            "status": self.status,
        }


@dataclass(frozen=True)
class DatasetDimensions:
    """Dimension tables used for referential checks."""

    stores: pa.Table
    skus: pa.Table
    categories: pa.Table


@dataclass(frozen=True)
class ValidationConfig:
    """Reference date and thresholds for staleness and extreme demand."""

    as_of: date
    staleness_days: int = 14
    outlier_multiple: float = 20.0

    def __post_init__(self) -> None:
        if self.staleness_days < 0:
            raise ValueError("staleness_days must be zero or greater.")
        if self.outlier_multiple <= 0:
            raise ValueError("outlier_multiple must be greater than zero.")


def validate_dataset(
    frame: pa.Table,
    dimensions: DatasetDimensions,
    config: ValidationConfig,
    *,
    report_dir: Path | None = None,
) -> QualityReport:
    """Validate ``frame`` and optionally write ``quality_report.json`` in ``report_dir``."""

    blocking = tuple(
        finding
        for finding in (
            find_duplicate_grain(frame),
            find_missing_required(frame),
            find_negative_units_sold(frame),
            find_negative_price(frame),
            find_invalid_promotion_discount(frame),
            find_unknown_store(frame, dimensions.stores),
            find_unknown_sku(frame, dimensions.skus),
            find_calendar_gaps(frame),
            find_stale_observations(
                frame, as_of=config.as_of, staleness_days=config.staleness_days
            ),
            find_extreme_outliers(frame, multiple=config.outlier_multiple),
        )
        if finding is not None
    )
    advisory = tuple(
        finding
        for finding in (
            _stockout_finding(frame),
            _unexpected_category_finding(frame, dimensions.categories),
            _zero_demand_finding(frame),
        )
        if finding is not None
    )
    report = QualityReport(
        status="invalid" if blocking else "valid",
        blocking=blocking,
        advisory=advisory,
        missing_value_rate=missing_value_rate(frame),
        duplicate_rate=duplicate_rate(frame),
        stockout_rate=stockout_rate(frame),
    )
    if report_dir is not None:
        write_quality_report(report, report_dir)
    return report


def write_quality_report(report: QualityReport, directory: Path) -> Path:
    """Write ``quality_report.json`` in ``directory`` and return that path."""

    if directory.exists() and not directory.is_dir():
        raise ValueError(f"Report path {directory} is a file. Pass a directory.")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / QUALITY_REPORT_FILENAME
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def find_duplicate_grain(frame: pa.Table) -> Finding | None:
    """Flag ``(date, store_id, sku_id)`` keys that occur more than once."""

    duplicated, _surplus = _duplicate_stats(frame)
    return _blocking(
        BlockingCode.DUPLICATE_GRAIN,
        duplicated,
        f"{duplicated} duplicate grain keys for (date, store_id, sku_id).",
    )


def find_missing_required(frame: pa.Table) -> Finding | None:
    """Flag null or blank values in required observation columns."""

    count, columns = _missing_required(frame)
    names = ", ".join(columns)
    return _blocking(
        BlockingCode.MISSING_REQUIRED,
        count,
        f"{count} missing values in required columns ({names}).",
    )


def find_negative_units_sold(frame: pa.Table) -> Finding | None:
    """Flag rows whose ``units_sold`` is below zero."""

    count = _negative_count(frame, "units_sold")
    return _blocking(
        BlockingCode.NEGATIVE_UNITS_SOLD,
        count,
        f"{count} rows with negative units_sold.",
    )


def find_negative_price(frame: pa.Table) -> Finding | None:
    """Flag rows whose ``price`` is below zero. Zero is allowed."""

    count = _negative_count(frame, "price")
    return _blocking(
        BlockingCode.NEGATIVE_PRICE,
        count,
        f"{count} rows with negative price.",
    )


def find_invalid_promotion_discount(frame: pa.Table) -> Finding | None:
    """Flag a promotion without ``discount_pct``, or a discount outside 0-100.

    ``discount_pct`` of 0 and 100 are inside the range. A missing discount is a
    failure only when ``promotion`` is true, or when the discount column itself
    is absent on a promoted row.
    """

    count = _invalid_promotion_discount_count(frame)
    return _blocking(
        BlockingCode.INVALID_PROMOTION_DISCOUNT,
        count,
        (f"{count} rows with a promotion missing discount_pct or a discount_pct outside 0-100."),
    )


def find_unknown_store(frame: pa.Table, stores: pa.Table) -> Finding | None:
    """Flag observation rows whose ``store_id`` is not in the store dimension."""

    count = _unknown_id_count(frame, "store_id", stores, "store")
    return _blocking(
        BlockingCode.UNKNOWN_STORE,
        count,
        f"{count} rows with a store_id missing from the store dimension.",
    )


def find_unknown_sku(frame: pa.Table, skus: pa.Table) -> Finding | None:
    """Flag observation rows whose ``sku_id`` is not in the SKU dimension."""

    count = _unknown_id_count(frame, "sku_id", skus, "SKU")
    return _blocking(
        BlockingCode.UNKNOWN_SKU,
        count,
        f"{count} rows with a sku_id missing from the SKU dimension.",
    )


def find_calendar_gaps(frame: pa.Table) -> Finding | None:
    """Flag breaks longer than one day inside a ``(store_id, sku_id)`` series.

    The check compares consecutive observations after sorting. A series that
    starts later than another series is not a gap.
    """

    count = _calendar_gap_count(frame)
    return _blocking(
        BlockingCode.CALENDAR_GAP,
        count,
        f"{count} calendar gaps longer than 1 day inside a daily series.",
    )


def find_stale_observations(
    frame: pa.Table,
    *,
    as_of: date,
    staleness_days: int = 14,
) -> Finding | None:
    """Flag a dataset whose newest date is more than ``staleness_days`` before ``as_of``.

    A ``date_max`` exactly ``staleness_days`` before ``as_of`` is still fresh.
    """

    if staleness_days < 0:
        raise ValueError("staleness_days must be zero or greater.")
    newest = _date_max(frame)
    if newest is None:
        return None
    age_days = (as_of - newest).days
    if age_days <= staleness_days:
        return None
    return Finding(
        code=BlockingCode.STALE_DATASET,
        count=1,
        message=(
            f"date_max {newest.isoformat()} is {age_days} days before {as_of.isoformat()}, "
            f"past the {staleness_days}-day staleness window."
        ),
    )


def find_extreme_outliers(frame: pa.Table, *, multiple: float = 20.0) -> Finding | None:
    """Flag a day whose ``units_sold`` exceeds ``multiple`` times the series median.

    The median uses positive ``units_sold`` only and includes every positive day
    in that ``(store_id, sku_id)`` series. A value equal to the threshold passes.
    Series with no positive demand are skipped.
    """

    if multiple <= 0:
        raise ValueError("outlier_multiple must be greater than zero.")
    count = _extreme_outlier_count(frame, multiple)
    rendered = int(multiple) if multiple == int(multiple) else multiple
    return _blocking(
        BlockingCode.EXTREME_OUTLIER,
        count,
        (
            f"{count} rows with units_sold above {rendered} times "
            "the series median of positive demand."
        ),
    )


def missing_value_rate(frame: pa.Table) -> float:
    """Return the fraction of required cells that are null or blank."""

    rows = _row_count(frame)
    if rows == 0:
        return 0.0
    count, _columns = _missing_required(frame)
    return count / (rows * len(REQUIRED_COLUMNS))


def duplicate_rate(frame: pa.Table) -> float:
    """Return the fraction of rows that are extra copies of a grain key."""

    rows = _row_count(frame)
    if rows == 0:
        return 0.0
    _duplicated, surplus = _duplicate_stats(frame)
    return surplus / rows


def stockout_rate(frame: pa.Table) -> float:
    """Return the fraction of rows with ``stockout`` true."""

    if frame.num_rows == 0 or "stockout" not in frame.column_names:
        return 0.0
    values, valid = _bool_column(frame, "stockout")
    return float(np.sum(valid & values)) / float(frame.num_rows)


def unexpected_category_count(frame: pa.Table, categories: pa.Table) -> int:
    """Return how many distinct ``category_id`` values are absent from the dimension."""

    if frame.num_rows == 0 or "category_id" not in frame.column_names:
        return 0
    known = _dimension_ids(categories, "category_id", "category")
    observed = _present_strings(frame.column("category_id"), "category_id")
    return len(set(observed) - known)


def zero_demand_fraction(frame: pa.Table) -> float:
    """Return the fraction of rows whose ``units_sold`` is zero."""

    if frame.num_rows == 0 or "units_sold" not in frame.column_names:
        return 0.0
    values, valid = _number_column(frame, "units_sold")
    return float(np.sum(valid & (values == 0))) / float(frame.num_rows)


def _stockout_finding(frame: pa.Table) -> Finding | None:
    if frame.num_rows == 0 or "stockout" not in frame.column_names:
        return None
    values, valid = _bool_column(frame, "stockout")
    count = int(np.sum(valid & values))
    rate = count / frame.num_rows
    return _advisory(
        AdvisoryCode.STOCKOUT_RATE,
        count,
        f"{count} of {frame.num_rows} rows are stock-outs.",
        rate=rate,
    )


def _unexpected_category_finding(frame: pa.Table, categories: pa.Table) -> Finding | None:
    count = unexpected_category_count(frame, categories)
    return _advisory(
        AdvisoryCode.UNEXPECTED_CATEGORY,
        count,
        f"{count} categories are not in the category dimension.",
    )


def _zero_demand_finding(frame: pa.Table) -> Finding | None:
    if frame.num_rows == 0 or "units_sold" not in frame.column_names:
        return None
    values, valid = _number_column(frame, "units_sold")
    count = int(np.sum(valid & (values == 0)))
    rate = count / frame.num_rows
    return _advisory(
        AdvisoryCode.ZERO_DEMAND_FRACTION,
        count,
        f"{count} of {frame.num_rows} rows have zero units_sold.",
        rate=rate,
    )


def _blocking(code: BlockingCode, count: int, message: str) -> Finding | None:
    if count <= 0:
        return None
    return Finding(code=code, count=count, message=message)


def _advisory(
    code: AdvisoryCode,
    count: int,
    message: str,
    *,
    rate: float | None = None,
) -> Finding | None:
    if count <= 0:
        return None
    return Finding(code=code, count=count, message=message, rate=rate)


def _missing_required(frame: pa.Table) -> tuple[int, tuple[str, ...]]:
    columns: list[str] = []
    count = 0
    for name in REQUIRED_COLUMNS:
        missing = _missing_in_column(frame, name)
        if missing:
            columns.append(name)
            count += missing
    return count, tuple(columns)


def _missing_in_column(frame: pa.Table, name: str) -> int:
    if name not in frame.column_names:
        return _row_count(frame)
    column = frame.column(name)
    if name in _STRING_IDS:
        return _blank_string_count(column, name)
    return _null_count(column)


def _blank_string_count(column: pa.ChunkedArray, name: str) -> int:
    _require_string(column, name)
    blanks = pc.sum(pc.fill_null(pc.equal(column, ""), False)).as_py()
    return _null_count(column) + int(blanks or 0)


def _row_count(frame: pa.Table) -> int:
    return int(frame.num_rows)


def _null_count(column: pa.ChunkedArray) -> int:
    return int(column.null_count)


def _duplicate_stats(frame: pa.Table) -> tuple[int, int]:
    """Return ``(duplicated_key_count, surplus_row_count)``."""

    if frame.num_rows == 0 or any(name not in frame.column_names for name in _GRAIN):
        return 0, 0
    keys = _complete_rows(frame, _GRAIN)
    if keys is None:
        return 0, 0
    grouped = keys.group_by(list(_GRAIN)).aggregate([([], "count_all")])
    counts = np.asarray(grouped.column("count_all").to_numpy(zero_copy_only=False), dtype=np.int64)
    duplicated = int(np.sum(counts > 1))
    surplus = int(np.sum(counts[counts > 1] - 1))
    return duplicated, surplus


def _negative_count(frame: pa.Table, name: str) -> int:
    if frame.num_rows == 0 or name not in frame.column_names:
        return 0
    values, valid = _number_column(frame, name)
    return int(np.sum(valid & (values < 0)))


def _invalid_promotion_discount_count(frame: pa.Table) -> int:
    if frame.num_rows == 0:
        return 0
    has_promotion = "promotion" in frame.column_names
    has_discount = "discount_pct" in frame.column_names
    if not has_promotion and not has_discount:
        return 0

    invalid = np.zeros(frame.num_rows, dtype=np.bool_)
    if has_discount:
        discount, discount_valid = _number_column(frame, "discount_pct")
        invalid |= discount_valid & ((discount < 0) | (discount > 100))
    if has_promotion:
        promoted, promotion_valid = _bool_column(frame, "promotion")
        on_promotion = promotion_valid & promoted
        if has_discount:
            _discount, discount_valid = _number_column(frame, "discount_pct")
            invalid |= on_promotion & ~discount_valid
        else:
            invalid |= on_promotion
    return int(np.sum(invalid))


def _unknown_id_count(frame: pa.Table, name: str, dimension: pa.Table, label: str) -> int:
    if frame.num_rows == 0 or name not in frame.column_names:
        return 0
    known = _dimension_ids(dimension, name, label)
    column = frame.column(name)
    _require_string(column, name)
    present = pc.fill_null(column, "")
    member = pc.fill_null(
        pc.is_in(present, value_set=pa.array(sorted(known), type=pa.string())), False
    )
    unknown = pc.and_(_nonempty_string(column, name), pc.invert(member))
    total = pc.sum(unknown).as_py()
    return int(total or 0)


def _calendar_gap_count(frame: pa.Table) -> int:
    rows = _complete_rows(frame, _GRAIN)
    if rows is None or rows.num_rows < 2:
        return 0
    store_codes = _string_codes(rows.column("store_id"))
    sku_codes = _string_codes(rows.column("sku_id"))
    days = np.asarray(
        rows.column("date").cast(pa.int32()).to_numpy(zero_copy_only=False), dtype=np.int64
    )
    order = np.lexsort((days, sku_codes, store_codes))
    ordered_store = store_codes[order]
    ordered_sku = sku_codes[order]
    ordered_days = days[order]
    same_series = (ordered_store[1:] == ordered_store[:-1]) & (ordered_sku[1:] == ordered_sku[:-1])
    delta = np.diff(ordered_days)
    return int(np.sum(same_series & (delta > 1)))


def _date_max(frame: pa.Table) -> date | None:
    if frame.num_rows == 0 or "date" not in frame.column_names:
        return None
    days, valid = _date_days(frame)
    if not np.any(valid):
        return None
    return _UNIX_EPOCH + timedelta(days=int(np.max(days[valid])))


def _extreme_outlier_count(frame: pa.Table, multiple: float) -> int:
    if frame.num_rows == 0 or any(
        name not in frame.column_names for name in ("store_id", "sku_id", "units_sold")
    ):
        return 0
    rows = _complete_rows(frame, ("store_id", "sku_id", "units_sold"))
    if rows is None:
        return 0
    store_codes = _string_codes(rows.column("store_id"))
    sku_codes = _string_codes(rows.column("sku_id"))
    units = np.asarray(
        rows.column("units_sold").cast(pa.float64()).to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    order = np.lexsort((sku_codes, store_codes))
    ordered_store = store_codes[order]
    ordered_sku = sku_codes[order]
    ordered_units = units[order]
    count = 0
    for start, stop in _series_slices(ordered_store, ordered_sku):
        observed = ordered_units[start:stop]
        positive = observed[observed > 0]
        if positive.size == 0:
            continue
        threshold = multiple * float(np.median(positive))
        count += int(np.sum(observed > threshold))
    return count


def _series_slices(store_codes: I64, sku_codes: I64) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    start = 0
    total = len(store_codes)
    while start < total:
        stop = start + 1
        while (
            stop < total
            and store_codes[stop] == store_codes[start]
            and sku_codes[stop] == sku_codes[start]
        ):
            stop += 1
        slices.append((start, stop))
        start = stop
    return slices


def _complete_rows(frame: pa.Table, columns: Sequence[str]) -> pa.Table | None:
    for name in columns:
        if name not in frame.column_names:
            return None
    table = frame.select(list(columns))
    mask: pa.ChunkedArray | None = None
    for name in columns:
        column = table.column(name)
        if _is_string(column):
            present = _nonempty_string(column, name)
        elif name == "date":
            present = pc.is_valid(_as_date(column, name))
        else:
            present = pc.is_valid(column)
        mask = present if mask is None else pc.and_(mask, present)
    if mask is None:
        return None
    filtered = table.filter(mask)
    if filtered.num_rows == 0:
        return None
    return filtered


def _string_codes(column: pa.ChunkedArray) -> I64:
    _require_string(column, "series key")
    _uniques, codes = np.unique(column.to_numpy(zero_copy_only=False), return_inverse=True)
    return np.asarray(codes, dtype=np.int64)


def _present_strings(column: pa.ChunkedArray, name: str) -> list[str]:
    _require_string(column, name)
    values = column.to_pylist()
    return [value for value in values if isinstance(value, str) and value != ""]


def _dimension_ids(table: pa.Table, column: str, label: str) -> set[str]:
    if column not in table.column_names:
        raise ValueError(f"The {label} dimension is missing {column}.")
    return set(_present_strings(table.column(column), column))


def _nonempty_string(column: pa.ChunkedArray, name: str) -> pa.ChunkedArray:
    _require_string(column, name)
    filled = pc.fill_null(column, "")
    return pc.and_(pc.is_valid(column), pc.not_equal(filled, ""))


def _number_column(frame: pa.Table, name: str) -> tuple[F64, BoolArray]:
    column = frame.column(name)
    if not (pa.types.is_integer(column.type) or pa.types.is_floating(column.type)):
        raise ValueError(f"Column {name} has type {column.type}, expected a number.")
    values = np.asarray(
        column.cast(pa.float64()).fill_null(0).to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    return values, _valid_mask(column)


def _bool_column(frame: pa.Table, name: str) -> tuple[BoolArray, BoolArray]:
    column = frame.column(name)
    if not pa.types.is_boolean(column.type):
        raise ValueError(f"Column {name} has type {column.type}, expected a boolean.")
    values = np.asarray(column.fill_null(False).to_numpy(zero_copy_only=False), dtype=np.bool_)
    return values, _valid_mask(column)


def _date_days(frame: pa.Table) -> tuple[I64, BoolArray]:
    column = _as_date(frame.column("date"), "date")
    days = np.asarray(
        column.cast(pa.int32()).fill_null(0).to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    return days, _valid_mask(frame.column("date"))


def _as_date(column: pa.ChunkedArray, name: str) -> pa.ChunkedArray:
    if pa.types.is_date32(column.type):
        return column
    try:
        casted = column.cast(pa.date32())
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise ValueError(f"Column {name} has type {column.type}, expected a date.") from exc
    if isinstance(casted, pa.Array):
        return pa.chunked_array([casted])
    return casted


def _valid_mask(column: pa.ChunkedArray) -> BoolArray:
    return np.asarray(column.is_valid().to_numpy(zero_copy_only=False), dtype=np.bool_)


def _require_string(column: pa.ChunkedArray, name: str) -> None:
    if not _is_string(column):
        raise ValueError(f"Column {name} has type {column.type}, expected a string.")


def _is_string(column: pa.ChunkedArray) -> bool:
    return bool(pa.types.is_string(column.type) or pa.types.is_large_string(column.type))
