"""Feature rows shared by training, evaluation, and inference.

:func:`build_features` is the only place that computes lags and rolling
windows. A lag or rolling window uses observations on or before the cutoff
and strictly before the row date. Calendar fields, planned price, planned
discount, planned promotion, and planned holiday come from the row itself.
``units_sold``, ``stock_available``, and ``stockout`` are not model inputs.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import cast

import numpy as np
import pyarrow as pa

MODEL_FEATURES: tuple[str, ...] = (
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
    "day_of_week",
    "week_of_year",
    "month",
    "price",
    "discount_pct",
    "promotion",
    "holiday",
    "store_id",
    "sku_id",
    "category_id",
)

_LAGS = (1, 7, 14, 28)
_REQUIRED = ("date", "store_id", "sku_id", "category_id", "price")


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one model input or one rejected column."""

    name: str
    description: str
    dtype: str
    source: str
    available_at_prediction_time: bool


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "lag_1",
        "Target one period before the row date.",
        "float64",
        "units_sold",
        True,
    ),
    FeatureSpec(
        "lag_7",
        "Target seven periods before the row date.",
        "float64",
        "units_sold",
        True,
    ),
    FeatureSpec(
        "lag_14",
        "Target fourteen periods before the row date.",
        "float64",
        "units_sold",
        True,
    ),
    FeatureSpec(
        "lag_28",
        "Target twenty-eight periods before the row date.",
        "float64",
        "units_sold",
        True,
    ),
    FeatureSpec(
        "rolling_mean_7",
        "Mean of targets in the seven periods before the row date.",
        "float64",
        "units_sold",
        True,
    ),
    FeatureSpec(
        "rolling_mean_28",
        "Mean of targets in the twenty-eight periods before the row date.",
        "float64",
        "units_sold",
        True,
    ),
    FeatureSpec(
        "rolling_std_28",
        "Population standard deviation of targets in the twenty-eight periods before the row date.",
        "float64",
        "units_sold",
        True,
    ),
    FeatureSpec(
        "day_of_week",
        "Weekday of the row date. Monday is 0.",
        "int64",
        "date",
        True,
    ),
    FeatureSpec(
        "week_of_year",
        "ISO week number of the row date.",
        "int64",
        "date",
        True,
    ),
    FeatureSpec(
        "month",
        "Calendar month of the row date, from 1 to 12.",
        "int64",
        "date",
        True,
    ),
    FeatureSpec(
        "price",
        "Planned selling price on the row date.",
        "float64",
        "price",
        True,
    ),
    FeatureSpec(
        "discount_pct",
        "Planned discount percent on the row date.",
        "float64",
        "discount_pct",
        True,
    ),
    FeatureSpec(
        "promotion",
        "Planned promotion flag on the row date.",
        "bool",
        "promotion",
        True,
    ),
    FeatureSpec(
        "holiday",
        "Planned holiday flag on the row date.",
        "bool",
        "holiday",
        True,
    ),
    FeatureSpec(
        "store_id",
        "Store identifier. The encoding is fit on training rows only.",
        "string",
        "store_id",
        True,
    ),
    FeatureSpec(
        "sku_id",
        "SKU identifier. The encoding is fit on training rows only.",
        "string",
        "sku_id",
        True,
    ),
    FeatureSpec(
        "category_id",
        "Category identifier. The encoding is fit on training rows only.",
        "string",
        "category_id",
        True,
    ),
    FeatureSpec(
        "units_sold",
        "Target. It is not known for dates after the cutoff.",
        "float64",
        "units_sold",
        False,
    ),
    FeatureSpec(
        "stock_available",
        "Units on hand. Stock after the cutoff is not a known-future covariate.",
        "int64",
        "stock_available",
        False,
    ),
    FeatureSpec(
        "stockout",
        "Stock-out flag. It is not a known-future covariate.",
        "bool",
        "stockout",
        False,
    ),
)

_SPECS = {spec.name: spec for spec in FEATURE_SPECS}


@dataclass(frozen=True)
class FeatureFrame:
    """Feature rows and the metadata for the columns the model may use.

    ``frame`` includes ``date``, ``units_sold``, and every selected feature.
    ``units_sold`` is the target and is null when the row has no known demand.
    ``metadata`` has ``name``, ``description``, ``dtype``, ``source``, and
    ``available_at_prediction_time``.
    """

    frame: pa.Table
    metadata: pa.Table
    cutoff_date: date

    def __post_init__(self) -> None:
        if type(self.cutoff_date) is not date:
            raise ValueError("cutoff_date must be a calendar date.")
        require_prediction_time_features(self.metadata)


def build_features(
    df: pa.Table,
    cutoff_date: date,
    config: Mapping[str, object] | None = None,
) -> FeatureFrame:
    """Build one feature row per input row.

    ``cutoff_date`` is the last date whose target may enter a lag or a rolling
    window. ``config["grain"]`` is ``day`` or ``week`` and selects the period
    length. ``config["features"]`` selects model inputs. A feature with
    ``available_at_prediction_time`` false is rejected.
    """

    if not isinstance(df, pa.Table):
        raise ValueError("Feature input must be a pyarrow table.")
    if type(cutoff_date) is not date:
        raise ValueError("cutoff_date must be a calendar date.")
    selected = _selected_features(config)
    grain = _grain(config)
    period_days = 7 if grain == "week" else 1
    rows = _read_rows(df)
    history = {
        (row.store_id, row.sku_id, row.day): row.units_sold
        for row in rows
        if row.units_sold is not None and row.day <= cutoff_date
    }
    built = [_featurize(row, history, cutoff_date, period_days) for row in rows]
    return FeatureFrame(
        frame=_feature_table(built, selected),
        metadata=metadata_table(selected),
        cutoff_date=cutoff_date,
    )


def metadata_table(names: Sequence[str] = MODEL_FEATURES) -> pa.Table:
    """Return feature metadata for ``names``."""

    specs = [_SPECS[name] for name in names]
    return pa.table(
        {
            "name": pa.array([spec.name for spec in specs], type=pa.string()),
            "description": pa.array([spec.description for spec in specs], type=pa.string()),
            "dtype": pa.array([spec.dtype for spec in specs], type=pa.string()),
            "source": pa.array([spec.source for spec in specs], type=pa.string()),
            "available_at_prediction_time": pa.array(
                [spec.available_at_prediction_time for spec in specs],
                type=pa.bool_(),
            ),
        }
    )


def require_prediction_time_features(metadata: pa.Table) -> None:
    """Raise when metadata lists a feature that prediction time cannot know."""

    if not isinstance(metadata, pa.Table):
        raise ValueError("Feature metadata must be a pyarrow table.")
    required = ("name", "description", "dtype", "source", "available_at_prediction_time")
    missing = [name for name in required if name not in metadata.column_names]
    if missing:
        found = ", ".join(missing)
        raise ValueError(f"Feature metadata is missing {found}.")
    names = metadata.column("name").to_pylist()
    flags = metadata.column("available_at_prediction_time").to_pylist()
    blocked = [str(name) for name, flag in zip(names, flags, strict=True) if not flag]
    if blocked:
        found = ", ".join(blocked)
        raise ValueError(f"Feature {found} is not available at prediction time.")


@dataclass(frozen=True)
class _Row:
    day: date
    store_id: str
    sku_id: str
    category_id: str
    units_sold: float | None
    price: float | None
    discount_pct: float | None
    promotion: bool | None
    holiday: bool | None


@dataclass(frozen=True)
class _Built:
    row: _Row
    lags: dict[int, float | None]
    rolling_mean_7: float | None
    rolling_mean_28: float | None
    rolling_std_28: float | None


def _featurize(
    row: _Row,
    history: Mapping[tuple[str, str, date], float],
    cutoff_date: date,
    period_days: int,
) -> _Built:
    lags = {lag: _lag_value(history, row, cutoff_date, lag * period_days) for lag in _LAGS}
    mean_7 = _window_stat(history, row, cutoff_date, 7 * period_days, "mean")
    mean_28 = _window_stat(history, row, cutoff_date, 28 * period_days, "mean")
    std_28 = _window_stat(history, row, cutoff_date, 28 * period_days, "std")
    return _Built(
        row=row,
        lags=lags,
        rolling_mean_7=mean_7,
        rolling_mean_28=mean_28,
        rolling_std_28=std_28,
    )


def _lag_value(
    history: Mapping[tuple[str, str, date], float],
    row: _Row,
    cutoff_date: date,
    days_back: int,
) -> float | None:
    lagged = row.day - timedelta(days=days_back)
    if lagged > cutoff_date or lagged >= row.day:
        return None
    return history.get((row.store_id, row.sku_id, lagged))


def _window_stat(
    history: Mapping[tuple[str, str, date], float],
    row: _Row,
    cutoff_date: date,
    window_days: int,
    stat: str,
) -> float | None:
    start = row.day - timedelta(days=window_days)
    end = min(row.day - timedelta(days=1), cutoff_date)
    if end < start:
        return None
    values = [
        units
        for (store_id, sku_id, day), units in history.items()
        if store_id == row.store_id and sku_id == row.sku_id and start <= day <= end
    ]
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if stat == "mean":
        return float(np.mean(array))
    if len(values) < 2:
        return None
    return float(np.std(array, ddof=0))


def _feature_table(built: Sequence[_Built], selected: Sequence[str]) -> pa.Table:
    columns: dict[str, pa.Array] = {
        "date": pa.array([item.row.day for item in built], type=pa.date32()),
        "units_sold": pa.array([item.row.units_sold for item in built], type=pa.float64()),
    }
    for name in selected:
        columns[name] = _feature_column(built, name)
    return pa.table(columns)


def _feature_column(built: Sequence[_Built], name: str) -> pa.Array:
    if name == "lag_1":
        return _float_array([item.lags[1] for item in built])
    if name == "lag_7":
        return _float_array([item.lags[7] for item in built])
    if name == "lag_14":
        return _float_array([item.lags[14] for item in built])
    if name == "lag_28":
        return _float_array([item.lags[28] for item in built])
    if name == "rolling_mean_7":
        return _float_array([item.rolling_mean_7 for item in built])
    if name == "rolling_mean_28":
        return _float_array([item.rolling_mean_28 for item in built])
    if name == "rolling_std_28":
        return _float_array([item.rolling_std_28 for item in built])
    if name == "day_of_week":
        return pa.array([item.row.day.weekday() for item in built], type=pa.int64())
    if name == "week_of_year":
        return pa.array([item.row.day.isocalendar().week for item in built], type=pa.int64())
    if name == "month":
        return pa.array([item.row.day.month for item in built], type=pa.int64())
    if name == "price":
        return _float_array([item.row.price for item in built])
    if name == "discount_pct":
        return _float_array([item.row.discount_pct for item in built])
    if name == "promotion":
        return pa.array([item.row.promotion for item in built], type=pa.bool_())
    if name == "holiday":
        return pa.array([item.row.holiday for item in built], type=pa.bool_())
    if name == "store_id":
        return pa.array([item.row.store_id for item in built], type=pa.string())
    if name == "sku_id":
        return pa.array([item.row.sku_id for item in built], type=pa.string())
    if name == "category_id":
        return pa.array([item.row.category_id for item in built], type=pa.string())
    raise ValueError(f"Unknown feature {name}.")


def _float_array(values: Sequence[float | None]) -> pa.Array:
    return pa.array(list(values), type=pa.float64())


def _selected_features(config: Mapping[str, object] | None) -> tuple[str, ...]:
    if config is None or "features" not in config or config["features"] is None:
        return MODEL_FEATURES
    raw = config["features"]
    if isinstance(raw, str) or not isinstance(raw, Sequence) or not raw:
        raise ValueError("Feature config features must be a non-empty sequence of names.")
    names = tuple(raw)
    if any(not isinstance(name, str) or name == "" for name in names):
        raise ValueError("Feature names must be non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique.")
    unknown = [name for name in names if name not in _SPECS]
    if unknown:
        found = ", ".join(unknown)
        raise ValueError(f"Unknown feature {found}.")
    blocked = [name for name in names if not _SPECS[name].available_at_prediction_time]
    if blocked:
        found = ", ".join(blocked)
        raise ValueError(f"Feature {found} is not available at prediction time.")
    return names


def _grain(config: Mapping[str, object] | None) -> str:
    if config is None or "grain" not in config or config["grain"] is None:
        return "week"
    grain = config["grain"]
    if grain not in {"day", "week"}:
        raise ValueError("Feature grain must be day or week.")
    return str(grain)


def _read_rows(df: pa.Table) -> tuple[_Row, ...]:
    missing = [name for name in _REQUIRED if name not in df.column_names]
    if missing:
        found = ", ".join(missing)
        raise ValueError(f"Feature input is missing {found}.")
    if df.num_rows == 0:
        raise ValueError("Feature input has no rows.")
    days = _dates(df)
    stores = _strings(df, "store_id")
    skus = _strings(df, "sku_id")
    categories = _strings(df, "category_id")
    units = _optional_floats(df, "units_sold")
    prices = _optional_floats(df, "price")
    discounts = _optional_floats(df, "discount_pct")
    promotions = _optional_flags(df, "promotion")
    holidays = _optional_flags(df, "holiday")
    rows: list[_Row] = []
    seen: set[tuple[str, str, date]] = set()
    for index, day in enumerate(days):
        key = (stores[index], skus[index], day)
        if key in seen:
            stamp = day.isoformat()
            raise ValueError(
                f"Feature input repeats store {stores[index]} SKU {skus[index]} on {stamp}."
            )
        seen.add(key)
        rows.append(
            _Row(
                day=day,
                store_id=stores[index],
                sku_id=skus[index],
                category_id=categories[index],
                units_sold=units[index],
                price=prices[index],
                discount_pct=discounts[index],
                promotion=promotions[index],
                holiday=holidays[index],
            )
        )
    return tuple(rows)


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
    if pa.types.is_dictionary(column.type):
        column = column.cast(pa.string())
    if not (pa.types.is_string(column.type) or pa.types.is_large_string(column.type)):
        raise ValueError(f"Column {name} has type {column.type}, expected a string.")
    if column.null_count:
        raise ValueError(f"Column {name} contains nulls.")
    values = [str(value) for value in column.to_pylist()]
    if any(value == "" for value in values):
        raise ValueError(f"Column {name} contains a blank value.")
    return values


def _optional_floats(frame: pa.Table, name: str) -> list[float | None]:
    if name not in frame.column_names:
        return [None] * frame.num_rows
    column = frame.column(name)
    if not (pa.types.is_floating(column.type) or pa.types.is_integer(column.type)):
        raise ValueError(f"Column {name} has type {column.type}, expected a number.")
    values: list[float | None] = []
    for value in column.to_pylist():
        if value is None:
            values.append(None)
        else:
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f"Column {name} contains a non-finite value.")
            values.append(number)
    return values


def _optional_flags(frame: pa.Table, name: str) -> list[bool | None]:
    if name not in frame.column_names:
        return [None] * frame.num_rows
    column = frame.column(name)
    if not pa.types.is_boolean(column.type):
        raise ValueError(f"Column {name} has type {column.type}, expected a boolean.")
    return [None if value is None else bool(value) for value in column.to_pylist()]
