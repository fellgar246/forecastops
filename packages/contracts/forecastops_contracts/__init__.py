"""Shared dataset contracts."""

from forecastops_contracts.schema import (
    CATEGORY_COLUMNS,
    OBSERVATION_COLUMNS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    SKU_COLUMNS,
    STORE_COLUMNS,
    ColumnSpec,
    LogicalType,
    category_schema_problems,
    column_schema_problems,
    observation_schema_problems,
    sku_schema_problems,
    store_schema_problems,
)

__all__ = [
    "CATEGORY_COLUMNS",
    "OBSERVATION_COLUMNS",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "SCHEMA_VERSION",
    "SKU_COLUMNS",
    "STORE_COLUMNS",
    "ColumnSpec",
    "LogicalType",
    "category_schema_problems",
    "column_schema_problems",
    "observation_schema_problems",
    "sku_schema_problems",
    "store_schema_problems",
]
