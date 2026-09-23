"""Synthetic retail history: catalog shape, contract, and reproducible files."""

import csv
import os
import subprocess
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forecastops_contracts import (
    SCHEMA_VERSION,
    category_schema_problems,
    observation_schema_problems,
    sku_schema_problems,
    store_schema_problems,
)
from forecastops_ml.data.synthetic import (
    DEFAULT_MAX_ROWS,
    DEFAULT_SEED,
    build_plan,
    generate_dataset,
)

ROOT = Path(__file__).resolve().parents[2]


def test_default_catalog_covers_two_years_of_daily_history() -> None:
    plan = build_plan("default", DEFAULT_SEED)

    assert plan == build_plan("default", DEFAULT_SEED)
    assert len(plan.stores) == 20
    assert len(plan.skus) == 150
    assert len(plan.categories) == 8
    assert plan.date_min == date(2024, 10, 1)
    assert plan.date_max == date(2026, 9, 30)
    assert plan.observation_date_min == plan.date_min
    assert plan.observation_date_max == plan.date_max
    months = (
        (plan.date_max.year - plan.date_min.year) * 12 + plan.date_max.month - plan.date_min.month
    )
    assert months + 1 == 24
    assert (plan.date_max - plan.date_min).days + 1 == 730
    assert plan.row_count == 1_438_959
    assert 1_000_000 <= plan.row_count <= DEFAULT_MAX_ROWS
    assert {store.store_region for store in plan.stores} == {"north", "south", "central", "west"}
    assert {sku.product_lifecycle for sku in plan.skus} == {"new", "mature", "declining"}
    assert len({sku.product_brand for sku in plan.skus}) >= 4
    assert max(item.n_days for item in plan.series) == 730
    assert min(item.n_days for item in plan.series) < 730
    assert plan.stores[0].store_id == "store-01"
    assert plan.skus[0].sku_id == "sku-001"


def test_test_profile_is_small_enough_for_unit_tests() -> None:
    plan = build_plan("test", DEFAULT_SEED)

    assert len(plan.stores) == 2
    assert len(plan.skus) == 4
    assert (plan.date_max - plan.date_min).days + 1 == 56
    assert plan.row_count < 1_000


def test_row_cap_stops_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="MAX_DATASET_ROWS_DEMO"):
        generate_dataset(tmp_path, profile="default", seed=DEFAULT_SEED, max_rows=10)

    assert list(tmp_path.iterdir()) == []


def test_generated_frame_matches_the_contract(tmp_path: Path) -> None:
    summary = generate_dataset(
        tmp_path, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS
    )
    observations = pq.read_table(tmp_path / "observations.parquet")
    stores = pq.read_table(tmp_path / "stores.parquet")
    skus = pq.read_table(tmp_path / "skus.parquet")
    categories = pq.read_table(tmp_path / "categories.parquet")

    assert (
        observation_schema_problems(
            observations.column_names, _logical_types(observations), exact=True
        )
        == ()
    )
    assert store_schema_problems(stores.column_names, _logical_types(stores), exact=True) == ()
    assert sku_schema_problems(skus.column_names, _logical_types(skus), exact=True) == ()
    assert (
        category_schema_problems(categories.column_names, _logical_types(categories), exact=True)
        == ()
    )
    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["seed"] == DEFAULT_SEED
    assert summary["stores"] == 2
    assert summary["skus"] == 4
    assert summary["categories"] == 2
    assert summary["row_count"] == observations.num_rows
    assert date.fromisoformat(summary["date_max"]) - date.fromisoformat(
        summary["date_min"]
    ) == timedelta(days=55)


def test_observations_are_internally_consistent(tmp_path: Path) -> None:
    generate_dataset(tmp_path, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    rows = pq.read_table(tmp_path / "observations.parquet").to_pylist()
    store_rows = pq.read_table(tmp_path / "stores.parquet").to_pylist()
    stores = {row["store_id"]: row["store_region"] for row in store_rows}
    sku_rows = pq.read_table(tmp_path / "skus.parquet").to_pylist()
    skus = {row["sku_id"]: row for row in sku_rows}
    categories = {
        row["category_id"] for row in pq.read_table(tmp_path / "categories.parquet").to_pylist()
    }

    assert rows
    assert any(row["promotion"] for row in rows)
    assert any(row["stockout"] for row in rows)
    assert any(row["holiday"] for row in rows)
    assert {row["product_lifecycle"] for row in sku_rows} == {"new", "mature", "declining"}
    keys = [(row["date"], row["store_id"], row["sku_id"]) for row in rows]
    assert len(keys) == len(set(keys))

    by_series: dict[tuple[str, str], list[date]] = defaultdict(list)
    for row in rows:
        assert row["units_sold"] >= 0
        assert row["price"] > 0
        assert 0 <= row["discount_pct"] <= 100
        assert row["store_id"] in stores
        assert row["sku_id"] in skus
        assert row["category_id"] in categories
        assert row["category_id"] == skus[row["sku_id"]]["category_id"]
        assert row["store_region"] == stores[row["store_id"]]
        assert isinstance(row["store_id"], str)
        assert isinstance(row["sku_id"], str)
        assert row["date"] != row["store_id"]
        if row["promotion"]:
            assert row["discount_pct"] > 0
            assert row["campaign"]
        else:
            assert row["discount_pct"] == 0
            assert row["campaign"] == ""
        if row["stockout"]:
            assert row["units_sold"] == 0
            assert row["stock_available"] == 0
        else:
            assert row["stock_available"] > row["units_sold"]
        if row["date"] == date(2026, 9, 16):
            assert row["holiday"] is True
        by_series[(row["store_id"], row["sku_id"])].append(row["date"])

    for series_dates in by_series.values():
        ordered = sorted(series_dates)
        gaps = [(right - left).days for left, right in zip(ordered, ordered[1:], strict=False)]
        assert gaps == [1] * len(gaps)


def test_same_seed_writes_identical_parquet_bytes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(first, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    generate_dataset(second, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)

    assert (first / "observations.parquet").read_bytes() == (
        second / "observations.parquet"
    ).read_bytes()


def test_a_different_seed_changes_the_fact_table(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(first, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    generate_dataset(second, profile="test", seed=DEFAULT_SEED + 1, max_rows=DEFAULT_MAX_ROWS)

    assert (first / "observations.parquet").read_bytes() != (
        second / "observations.parquet"
    ).read_bytes()


def test_make_seed_data_writes_the_test_profile(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["RANDOM_SEED"] = str(DEFAULT_SEED)
    env["MAX_DATASET_ROWS_DEMO"] = str(DEFAULT_MAX_ROWS)
    first = tmp_path / "first"
    second = tmp_path / "second"
    for dest in (first, second):
        subprocess.run(
            ["make", "seed-data", f"OUT={dest}", "PROFILE=test"],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    assert (first / "observations.parquet").read_bytes() == (
        second / "observations.parquet"
    ).read_bytes()
    assert (first / "summary.json").is_file()
    with (first / "observations.csv").open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header[0] == "date"
    assert "units_sold" in header


def _logical_types(table: pa.Table) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in table.column_names:
        data_type = table.schema.field(name).type
        if pa.types.is_date32(data_type):
            found[name] = "date"
        elif pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
            found[name] = "string"
        elif pa.types.is_int64(data_type):
            found[name] = "int64"
        elif pa.types.is_float64(data_type):
            found[name] = "float64"
        elif pa.types.is_boolean(data_type):
            found[name] = "boolean"
        else:
            found[name] = str(data_type)
    return found
