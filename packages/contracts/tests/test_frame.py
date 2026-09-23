"""Frame checks against the retail demand column contract."""

from forecastops_contracts import (
    REQUIRED_COLUMNS,
    category_schema_problems,
    observation_schema_problems,
    sku_schema_problems,
    store_schema_problems,
)

REQUIRED_TYPES = {
    "date": "date",
    "store_id": "string",
    "sku_id": "string",
    "category_id": "string",
    "units_sold": "int64",
    "price": "float64",
}


def test_required_columns_pass_when_types_match() -> None:
    assert observation_schema_problems(REQUIRED_COLUMNS, REQUIRED_TYPES) == ()


def test_missing_or_renamed_required_column_fails() -> None:
    names = ["quantity" if name == "units_sold" else name for name in REQUIRED_COLUMNS]
    types = dict(REQUIRED_TYPES)
    types["quantity"] = types.pop("units_sold")

    problems = observation_schema_problems(names, types)

    assert any("units_sold" in problem for problem in problems)


def test_wrong_required_column_type_fails() -> None:
    types = dict(REQUIRED_TYPES)
    types["price"] = "string"

    problems = observation_schema_problems(REQUIRED_COLUMNS, types)

    assert any("price" in problem and "float64" in problem for problem in problems)


def test_store_sku_and_category_schemas_reject_a_missing_column() -> None:
    assert store_schema_problems(["store_id"], {"store_id": "string"}) != ()
    assert sku_schema_problems(["sku_id"], {"sku_id": "string"}) != ()
    assert category_schema_problems(["category_id"], {"category_id": "string"}) != ()
