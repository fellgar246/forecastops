"""Schema version tests."""

from forecastops_contracts import (
    CATEGORY_COLUMNS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    SKU_COLUMNS,
    STORE_COLUMNS,
)


def test_schema_version() -> None:
    assert SCHEMA_VERSION == "retail-demand-v1"
    assert "units_sold" in REQUIRED_COLUMNS
    assert "date" in REQUIRED_COLUMNS
    assert "promotion" in OPTIONAL_COLUMNS
    assert set(REQUIRED_COLUMNS).isdisjoint(OPTIONAL_COLUMNS)


def test_dimension_schemas_include_region_brand_and_lifecycle() -> None:
    assert [column.name for column in STORE_COLUMNS] == ["store_id", "store_region"]
    assert [column.name for column in SKU_COLUMNS] == [
        "sku_id",
        "category_id",
        "product_brand",
        "product_lifecycle",
    ]
    assert [column.name for column in CATEGORY_COLUMNS] == ["category_id", "category_name"]
    assert all(column.logical_type == "string" for column in (*STORE_COLUMNS, *SKU_COLUMNS))
