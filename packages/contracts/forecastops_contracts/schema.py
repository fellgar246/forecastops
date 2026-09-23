"""Column contract for retail demand observations and their dimensions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class LogicalType(StrEnum):
    """Storage type for a contract column."""

    DATE = "date"
    STRING = "string"
    INT64 = "int64"
    FLOAT64 = "float64"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class ColumnSpec:
    """One column in a retail demand table."""

    name: str
    logical_type: LogicalType
    required: bool = True
    description: str = ""


SCHEMA_VERSION = "retail-demand-v1"

OBSERVATION_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("date", LogicalType.DATE, description="Calendar day of the observation."),
    ColumnSpec("store_id", LogicalType.STRING, description="Stable store identifier."),
    ColumnSpec("sku_id", LogicalType.STRING, description="Stable SKU identifier."),
    ColumnSpec("category_id", LogicalType.STRING, description="Stable category identifier."),
    ColumnSpec("units_sold", LogicalType.INT64, description="Units sold, the forecast target."),
    ColumnSpec("price", LogicalType.FLOAT64, description="Selling price for the day."),
    ColumnSpec(
        "promotion",
        LogicalType.BOOLEAN,
        required=False,
        description="True when the SKU is on promotion at the store.",
    ),
    ColumnSpec(
        "discount_pct",
        LogicalType.FLOAT64,
        required=False,
        description="Discount as a percent from 0 to 100.",
    ),
    ColumnSpec(
        "stock_available",
        LogicalType.INT64,
        required=False,
        description="Units on hand at the store.",
    ),
    ColumnSpec(
        "stockout",
        LogicalType.BOOLEAN,
        required=False,
        description="True when the store has none of the SKU left to sell.",
    ),
    ColumnSpec("holiday", LogicalType.BOOLEAN, required=False, description="True on a holiday."),
    ColumnSpec(
        "campaign",
        LogicalType.STRING,
        required=False,
        description="Campaign name, or an empty string when there is none.",
    ),
    ColumnSpec(
        "supplier_lead_time",
        LogicalType.INT64,
        required=False,
        description="Supplier lead time in days.",
    ),
    ColumnSpec(
        "weather_index",
        LogicalType.FLOAT64,
        required=False,
        description="Weather index from 0 to 1.",
    ),
    ColumnSpec("store_region", LogicalType.STRING, required=False, description="Store region."),
    ColumnSpec("product_brand", LogicalType.STRING, required=False, description="Product brand."),
    ColumnSpec(
        "product_lifecycle",
        LogicalType.STRING,
        required=False,
        description="Product lifecycle: new, mature, or declining.",
    ),
)

STORE_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("store_id", LogicalType.STRING, description="Stable store identifier."),
    ColumnSpec("store_region", LogicalType.STRING, description="Store region."),
)

SKU_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("sku_id", LogicalType.STRING, description="Stable SKU identifier."),
    ColumnSpec("category_id", LogicalType.STRING, description="Category that owns the SKU."),
    ColumnSpec("product_brand", LogicalType.STRING, description="Product brand."),
    ColumnSpec(
        "product_lifecycle",
        LogicalType.STRING,
        description="Product lifecycle: new, mature, or declining.",
    ),
)

CATEGORY_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("category_id", LogicalType.STRING, description="Stable category identifier."),
    ColumnSpec("category_name", LogicalType.STRING, description="Human-readable category name."),
)

REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    column.name for column in OBSERVATION_COLUMNS if column.required
)
OPTIONAL_COLUMNS: tuple[str, ...] = tuple(
    column.name for column in OBSERVATION_COLUMNS if not column.required
)


def observation_schema_problems(
    names: Sequence[str],
    types: Mapping[str, str],
    *,
    exact: bool = False,
) -> tuple[str, ...]:
    """Return problems when an observation frame misses a required column or type."""

    return column_schema_problems(OBSERVATION_COLUMNS, names, types, exact=exact)


def store_schema_problems(
    names: Sequence[str],
    types: Mapping[str, str],
    *,
    exact: bool = False,
) -> tuple[str, ...]:
    """Return problems when a store dimension frame does not match its schema."""

    return column_schema_problems(STORE_COLUMNS, names, types, exact=exact)


def sku_schema_problems(
    names: Sequence[str],
    types: Mapping[str, str],
    *,
    exact: bool = False,
) -> tuple[str, ...]:
    """Return problems when a SKU dimension frame does not match its schema."""

    return column_schema_problems(SKU_COLUMNS, names, types, exact=exact)


def category_schema_problems(
    names: Sequence[str],
    types: Mapping[str, str],
    *,
    exact: bool = False,
) -> tuple[str, ...]:
    """Return problems when a category dimension frame does not match its schema."""

    return column_schema_problems(CATEGORY_COLUMNS, names, types, exact=exact)


def column_schema_problems(
    columns: Sequence[ColumnSpec],
    names: Sequence[str],
    types: Mapping[str, str],
    *,
    exact: bool = False,
) -> tuple[str, ...]:
    """Return human-readable schema problems. An empty tuple means the frame matches."""

    problems: list[str] = []
    expected = {column.name: column for column in columns}
    seen = list(names)
    if len(set(seen)) != len(seen):
        problems.append("Column names must be unique.")

    present = set(seen)
    for column in columns:
        if column.name in present:
            continue
        if column.required or exact:
            problems.append(f"Missing required column {column.name}.")

    if exact and seen != [column.name for column in columns]:
        ordered = ", ".join(column.name for column in columns)
        problems.append(f"Column order does not match the schema: {ordered}.")

    for name in seen:
        spec = expected.get(name)
        if spec is None:
            if exact:
                problems.append(f"Unexpected column {name}.")
            continue
        actual = types.get(name)
        if actual is None:
            problems.append(f"Column {name} has no type.")
        elif actual != spec.logical_type:
            problems.append(f"Column {name} has type {actual}, expected {spec.logical_type}.")
    return tuple(problems)
