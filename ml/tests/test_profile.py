"""Weekly aggregations preserve totals, and the demand profile is stable."""

import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from forecastops_ml.data.profile import (
    PROFILE_FILENAME,
    aggregate_category_week,
    aggregate_region_week,
    aggregate_sku_week,
    aggregate_store_week,
    build_profile,
    load_dataset,
    write_profile,
)
from forecastops_ml.data.quality import DatasetDimensions
from forecastops_ml.data.synthetic import DEFAULT_MAX_ROWS, DEFAULT_SEED, generate_dataset

ROOT = Path(__file__).resolve().parents[2]


def test_weekly_totals_match_and_weeks_start_on_monday() -> None:
    monday = date(2026, 9, 16) - timedelta(days=date(2026, 9, 16).weekday())
    sunday = monday + timedelta(days=6)
    next_monday = monday + timedelta(days=7)
    assert monday.weekday() == 0
    assert sunday.weekday() == 6
    frame = _frame(
        [
            (monday, "store-01", "sku-001", "beverages", "north", 1, 10.0, False, False, False),
            (
                monday + timedelta(days=2),
                "store-01",
                "sku-001",
                "beverages",
                "north",
                3,
                20.0,
                True,
                False,
                False,
            ),
            (sunday, "store-01", "sku-001", "beverages", "north", 0, 9.0, False, False, False),
            (sunday, "store-01", "sku-002", "snacks", "north", 2, 5.0, False, True, False),
            (sunday, "store-01", "sku-003", "dairy", "north", 5, 3.0, False, False, False),
            (monday, "store-02", "sku-004", "beverages", "south", 6, 8.0, False, False, False),
            (next_monday, "store-02", "sku-001", "beverages", "south", 7, 4.0, False, False, True),
            (next_monday, "store-01", "sku-001", "beverages", "north", 1, 4.0, False, False, False),
        ]
    )
    daily_total = 25
    aggregated = (
        aggregate_sku_week(frame),
        aggregate_category_week(frame),
        aggregate_store_week(frame),
        aggregate_region_week(frame),
    )
    for table in aggregated:
        assert _units_total(table) == daily_total
        assert {row["week_start"] for row in table.to_pylist()} == {monday, next_monday}
        assert all(row["week_start"].weekday() == 0 for row in table.to_pylist())

    sku_rows = aggregate_sku_week(frame).to_pylist()
    sku_001_week = _row(sku_rows, sku_id="sku-001", week_start=monday)
    assert sku_001_week["units_sold"] == 4
    assert sku_001_week["price"] == pytest.approx(17.5)
    assert sku_001_week["promotion"] is True
    assert sku_001_week["holiday"] is False
    assert sku_001_week["stockout"] is False
    sku_003 = _row(sku_rows, sku_id="sku-003", week_start=monday)
    assert sku_003["week_start"] == monday
    sku_001_next = _row(sku_rows, sku_id="sku-001", week_start=next_monday)
    assert sku_001_next["units_sold"] == 8
    assert sku_001_next["price"] == pytest.approx(4.0)
    assert sku_001_next["promotion"] is False
    assert sku_001_next["stockout"] is True
    snacks = _row(
        aggregate_category_week(frame).to_pylist(), category_id="snacks", week_start=monday
    )
    assert snacks["holiday"] is True
    assert aggregate_category_week(frame).num_rows < aggregate_sku_week(frame).num_rows


def test_zero_demand_week_has_null_price_and_empty_input_keeps_the_schema() -> None:
    monday = date(2026, 9, 14)
    assert monday.weekday() == 0
    frame = _frame(
        [
            (monday, "store-01", "sku-001", "beverages", "north", 0, 5.0, False, False, False),
            (
                monday + timedelta(days=1),
                "store-01",
                "sku-001",
                "beverages",
                "north",
                0,
                9.0,
                False,
                False,
                True,
            ),
        ]
    )
    row = aggregate_sku_week(frame).to_pylist()[0]
    assert row["units_sold"] == 0
    assert row["price"] is None
    assert row["stockout"] is True
    assert row["promotion"] is False

    empty = aggregate_store_week(frame.slice(0, 0))
    assert empty.num_rows == 0
    assert empty.column_names == [
        "store_id",
        "week_start",
        "units_sold",
        "price",
        "promotion",
        "holiday",
        "stockout",
    ]


def test_region_week_uses_the_store_dimension_when_the_column_is_absent() -> None:
    monday = date(2026, 9, 14)
    frame = _frame(
        [
            (monday, "store-01", "sku-001", "beverages", None, 2, 3.0, False, False, False),
            (monday, "store-02", "sku-001", "beverages", None, 5, 3.0, True, False, False),
        ]
    )
    stores = pa.table(
        {
            "store_id": pa.array(["store-01", "store-02"], type=pa.string()),
            "store_region": pa.array(["north", "south"], type=pa.string()),
        }
    )
    rows = aggregate_region_week(frame, stores).to_pylist()
    assert _units_total(aggregate_region_week(frame, stores)) == 7
    assert _row(rows, store_region="north", week_start=monday)["units_sold"] == 2
    assert _row(rows, store_region="south", week_start=monday)["promotion"] is True
    with pytest.raises(ValueError, match="store-99"):
        aggregate_region_week(
            _frame(
                [(monday, "store-99", "sku-001", "beverages", None, 1, 1.0, False, False, False)]
            ),
            stores,
        )


def test_profile_metrics_describe_the_frame() -> None:
    monday = date(2026, 8, 3)
    assert monday.weekday() == 0
    frame = _frame(
        [
            (monday, "store-01", "sku-001", "beverages", "north", 0, 10.0, False, False, True),
            (
                monday + timedelta(days=1),
                "store-01",
                "sku-001",
                "beverages",
                "north",
                10,
                10.0,
                True,
                False,
                False,
            ),
            (monday, "store-01", "sku-002", "snacks", "north", 4, 2.0, False, True, False),
            (
                monday + timedelta(days=1),
                "store-01",
                "sku-002",
                "snacks",
                "north",
                8,
                6.0,
                True,
                False,
                False,
            ),
            (
                date(2026, 9, 1),
                "store-02",
                "sku-001",
                "beverages",
                "south",
                5,
                8.0,
                False,
                False,
                False,
            ),
        ]
    )
    profile = build_profile(frame, _dimensions()).to_dict()
    distribution = profile["demand_distribution"]
    assert isinstance(distribution, dict)
    assert distribution["count"] == 5
    assert distribution["mean"] == pytest.approx(27 / 5)
    assert distribution["median"] == pytest.approx(5.0)
    assert distribution["p90"] == pytest.approx(9.2)
    assert distribution["share_of_zeros"] == pytest.approx(0.2)
    assert profile["volume_by_category"] == [
        {"category_id": "beverages", "category_name": "Beverages", "units_sold": 15},
        {"category_id": "snacks", "category_name": "Snacks", "units_sold": 12},
    ]
    assert profile["weekday_seasonality_index"] == [
        {"index": pytest.approx(2 / (27 / 5)), "weekday": "monday"},
        {"index": pytest.approx((23 / 3) / (27 / 5)), "weekday": "tuesday"},
    ]
    assert profile["month_seasonality_index"] == [
        {"index": pytest.approx((22 / 4) / (27 / 5)), "month": 8},
        {"index": pytest.approx(5 / (27 / 5)), "month": 9},
    ]
    assert profile["promotion_lift"] == [
        {
            "category_id": "beverages",
            "category_name": "Beverages",
            "lift": pytest.approx(4.0),
            "mean_with_promotion": pytest.approx(10.0),
            "mean_without_promotion": pytest.approx(2.5),
        },
        {
            "category_id": "snacks",
            "category_name": "Snacks",
            "lift": pytest.approx(2.0),
            "mean_with_promotion": pytest.approx(8.0),
            "mean_without_promotion": pytest.approx(4.0),
        },
    ]
    assert profile["stockout_prevalence"] == pytest.approx(0.2)
    assert profile["sparsity"] == pytest.approx(0.2)
    assert profile["series_length"] == {"max": 2, "median": 2.0, "min": 1}


def test_sparsity_counts_series_days_and_lift_without_a_baseline_is_null() -> None:
    day = date(2026, 8, 3)
    duplicated = _frame(
        [
            (day, "store-01", "sku-001", "beverages", "north", 0, 1.0, False, False, False),
            (day, "store-01", "sku-001", "beverages", "north", 0, 1.0, False, False, False),
            (
                day + timedelta(days=1),
                "store-01",
                "sku-001",
                "beverages",
                "north",
                4,
                1.0,
                False,
                False,
                False,
            ),
        ]
    )
    profile = build_profile(duplicated, _dimensions()).to_dict()
    distribution = profile["demand_distribution"]
    assert isinstance(distribution, dict)
    assert distribution["share_of_zeros"] == pytest.approx(2 / 3)
    assert profile["sparsity"] == pytest.approx(0.5)

    promoted = _frame(
        [
            (day, "store-01", "sku-001", "beverages", "north", 10, 1.0, True, False, False),
            (
                day + timedelta(days=1),
                "store-01",
                "sku-001",
                "beverages",
                "north",
                0,
                1.0,
                False,
                False,
                False,
            ),
        ]
    )
    lift = build_profile(promoted, _dimensions()).to_dict()["promotion_lift"]
    assert isinstance(lift, list)
    assert lift[0]["mean_with_promotion"] == pytest.approx(10.0)
    assert lift[0]["mean_without_promotion"] == pytest.approx(0.0)
    assert lift[0]["lift"] is None


def test_zero_demand_seasonality_index_is_null() -> None:
    day = date(2026, 8, 3)
    frame = _frame(
        [
            (day, "store-01", "sku-001", "mystery", "north", 0, 1.0, False, False, False),
            (
                day + timedelta(days=1),
                "store-01",
                "sku-001",
                "mystery",
                "north",
                0,
                1.0,
                False,
                False,
                False,
            ),
        ]
    )
    profile = build_profile(frame, _dimensions()).to_dict()
    weekday = profile["weekday_seasonality_index"]
    volume = profile["volume_by_category"]
    assert isinstance(weekday, list)
    assert all(item["index"] is None for item in weekday)
    assert isinstance(volume, list)
    assert volume[0]["category_name"] is None


def test_test_profile_is_deterministic(tmp_path: Path) -> None:
    generate_dataset(tmp_path, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    frame, dimensions = load_dataset(tmp_path)
    first = build_profile(frame, dimensions)
    second = build_profile(frame, dimensions)
    assert first == second
    assert _profile_text(first.to_dict()) == _profile_text(second.to_dict())

    document = first.to_dict()
    distribution = document["demand_distribution"]
    assert isinstance(distribution, dict)
    for key in ("count", "mean", "median", "p90", "share_of_zeros"):
        assert key in distribution
    assert distribution["count"] == frame.num_rows
    volume = document["volume_by_category"]
    assert isinstance(volume, list)
    assert sum(int(item["units_sold"]) for item in volume) == _units_total(frame)
    assert {item["weekday"] for item in document["weekday_seasonality_index"]} == {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }
    assert [item["month"] for item in document["month_seasonality_index"]] == [8, 9]
    assert {item["category_id"] for item in document["promotion_lift"]} == {
        item["category_id"] for item in volume
    }
    assert document["sparsity"] == distribution["share_of_zeros"]
    length = document["series_length"]
    assert isinstance(length, dict)
    assert length["min"] <= length["median"] <= length["max"]
    for table in (
        aggregate_sku_week(frame),
        aggregate_category_week(frame),
        aggregate_store_week(frame),
        aggregate_region_week(frame.drop(["store_region"]), dimensions.stores),
    ):
        assert _units_total(table) == _units_total(frame)

    destination = write_profile(first, tmp_path)
    assert destination.name == PROFILE_FILENAME
    assert destination.read_text(encoding="utf-8") == _profile_text(document)
    write_profile(second, tmp_path)
    assert destination.read_text(encoding="utf-8") == _profile_text(document)


def test_load_dataset_reads_csv_when_parquet_is_absent(tmp_path: Path) -> None:
    generate_dataset(tmp_path, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    for path in tmp_path.glob("*.parquet"):
        path.unlink()
    frame, dimensions = load_dataset(tmp_path)
    assert frame.num_rows > 0
    assert dimensions.stores.num_rows == 2
    assert dimensions.skus.num_rows == 4
    assert dimensions.categories.num_rows == 2


def test_make_profile_data_writes_profile_json(tmp_path: Path) -> None:
    generate_dataset(tmp_path, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    _run_profile(tmp_path)
    first = (tmp_path / PROFILE_FILENAME).read_text(encoding="utf-8")
    document = json.loads(first)
    assert {
        "demand_distribution",
        "month_seasonality_index",
        "promotion_lift",
        "series_length",
        "sparsity",
        "stockout_prevalence",
        "volume_by_category",
        "weekday_seasonality_index",
    } <= set(document)
    _run_profile(tmp_path)
    assert (tmp_path / PROFILE_FILENAME).read_text(encoding="utf-8") == first


def _run_profile(dataset: Path) -> None:
    subprocess.run(
        ["make", "profile-data", f"OUT={dataset}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _profile_text(document: dict[str, object]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _units_total(table: pa.Table) -> int:
    return int(sum(table.column("units_sold").to_pylist()))


def _row(rows: list[dict[str, object]], **expected: object) -> dict[str, object]:
    matches = [row for row in rows if all(row[key] == value for key, value in expected.items())]
    assert len(matches) == 1
    return matches[0]


def _dimensions() -> DatasetDimensions:
    return DatasetDimensions(
        stores=pa.table(
            {
                "store_id": pa.array(["store-01", "store-02"], type=pa.string()),
                "store_region": pa.array(["north", "south"], type=pa.string()),
            }
        ),
        skus=pa.table(
            {
                "sku_id": pa.array(["sku-001", "sku-002"], type=pa.string()),
                "category_id": pa.array(["beverages", "snacks"], type=pa.string()),
                "product_brand": pa.array(["Northwind", "Lumen"], type=pa.string()),
                "product_lifecycle": pa.array(["mature", "mature"], type=pa.string()),
            }
        ),
        categories=pa.table(
            {
                "category_id": pa.array(["beverages", "snacks"], type=pa.string()),
                "category_name": pa.array(["Beverages", "Snacks"], type=pa.string()),
            }
        ),
    )


def _frame(
    rows: list[tuple[date, str, str, str, str | None, int, float, bool, bool, bool]],
) -> pa.Table:
    regions = [row[4] for row in rows]
    columns: dict[str, pa.Array] = {
        "date": pa.array([row[0] for row in rows], type=pa.date32()),
        "store_id": pa.array([row[1] for row in rows], type=pa.string()),
        "sku_id": pa.array([row[2] for row in rows], type=pa.string()),
        "category_id": pa.array([row[3] for row in rows], type=pa.string()),
        "units_sold": pa.array([row[5] for row in rows], type=pa.int64()),
        "price": pa.array([row[6] for row in rows], type=pa.float64()),
        "promotion": pa.array([row[7] for row in rows], type=pa.bool_()),
        "holiday": pa.array([row[8] for row in rows], type=pa.bool_()),
        "stockout": pa.array([row[9] for row in rows], type=pa.bool_()),
    }
    if any(region is not None for region in regions):
        columns["store_region"] = pa.array(regions, type=pa.string())
    return pa.table(columns)
