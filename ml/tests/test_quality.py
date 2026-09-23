"""Dataset validation: blocking rules, advisory rates, fixtures, and the JSON report."""

import json
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import pytest

from forecastops_ml.data import (
    AdvisoryCode,
    BlockingCode,
    DatasetDimensions,
    ValidationConfig,
    validate_dataset,
)
from forecastops_ml.data.quality import (
    duplicate_rate,
    find_calendar_gaps,
    find_duplicate_grain,
    find_extreme_outliers,
    find_invalid_promotion_discount,
    find_missing_required,
    find_negative_price,
    find_negative_units_sold,
    find_stale_observations,
    find_unknown_sku,
    find_unknown_store,
    missing_value_rate,
    stockout_rate,
    unexpected_category_count,
    zero_demand_fraction,
)
from forecastops_ml.data.synthetic import DEFAULT_MAX_ROWS, DEFAULT_SEED, generate_dataset

AS_OF = date(2026, 9, 30)
FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "fixtures"
    / "retail-demand-v1"
    / "validation"
)
COLUMN_TYPES = {
    "date": pa.date32(),
    "store_id": pa.string(),
    "store_region": pa.string(),
    "sku_id": pa.string(),
    "category_id": pa.string(),
    "category_name": pa.string(),
    "units_sold": pa.int64(),
    "price": pa.float64(),
    "promotion": pa.bool_(),
    "discount_pct": pa.float64(),
    "stockout": pa.bool_(),
    "product_brand": pa.string(),
    "product_lifecycle": pa.string(),
}
EXPECTED_COUNTS = {
    BlockingCode.DUPLICATE_GRAIN: 1,
    BlockingCode.MISSING_REQUIRED: 1,
    BlockingCode.NEGATIVE_UNITS_SOLD: 1,
    BlockingCode.NEGATIVE_PRICE: 1,
    BlockingCode.INVALID_PROMOTION_DISCOUNT: 1,
    BlockingCode.UNKNOWN_STORE: 5,
    BlockingCode.UNKNOWN_SKU: 5,
    BlockingCode.CALENDAR_GAP: 1,
    BlockingCode.STALE_DATASET: 1,
    BlockingCode.EXTREME_OUTLIER: 1,
}


def test_duplicate_grain_counts_repeated_keys_only() -> None:
    repeated = _frame([AS_OF, AS_OF], [10, 12])
    unique = _frame([AS_OF - timedelta(days=1), AS_OF], [10, 12])
    incomplete = _frame([None, None], [10, 12])

    found = find_duplicate_grain(repeated)

    assert found is not None
    assert found.code == BlockingCode.DUPLICATE_GRAIN
    assert found.count == 1
    assert duplicate_rate(repeated) == pytest.approx(0.5)
    assert find_duplicate_grain(unique) is None
    assert find_duplicate_grain(incomplete) is None
    assert duplicate_rate(incomplete) == 0


def test_missing_required_counts_nulls_blanks_and_absent_columns() -> None:
    blank_store = _frame([AS_OF], [10], store="")
    null_units = _frame([AS_OF, AS_OF - timedelta(days=1)], [None, 10])
    absent_price = _frame([AS_OF, AS_OF - timedelta(days=1)], [10, 11], omit=("price",))

    blank = find_missing_required(blank_store)
    missing_units = find_missing_required(null_units)
    missing_price = find_missing_required(absent_price)

    assert blank is not None and blank.count == 1
    assert missing_units is not None and missing_units.count == 1
    assert missing_value_rate(null_units) == pytest.approx(1 / 12)
    assert missing_price is not None and missing_price.count == 2
    assert find_missing_required(_frame([AS_OF], [10])) is None


def test_negative_units_sold_ignores_zero_and_null() -> None:
    frame = _frame([AS_OF, AS_OF - timedelta(days=1), AS_OF - timedelta(days=2)], [-1, 0, None])

    found = find_negative_units_sold(frame)

    assert found is not None
    assert found.code == BlockingCode.NEGATIVE_UNITS_SOLD
    assert found.count == 1
    assert find_negative_units_sold(_frame([AS_OF], [0])) is None


def test_negative_price_allows_zero() -> None:
    frame = _frame([AS_OF], [10], price=-0.01)

    found = find_negative_price(frame)

    assert found is not None
    assert found.code == BlockingCode.NEGATIVE_PRICE
    assert found.count == 1
    assert find_negative_price(_frame([AS_OF], [10], price=0)) is None


def test_promotion_discount_accepts_the_closed_range() -> None:
    assert (
        find_invalid_promotion_discount(_frame([AS_OF], [10], promotion=True, discount=0)) is None
    )
    assert (
        find_invalid_promotion_discount(_frame([AS_OF], [10], promotion=True, discount=100)) is None
    )
    assert (
        find_invalid_promotion_discount(_frame([AS_OF], [10], promotion=False, discount=None))
        is None
    )

    missing = find_invalid_promotion_discount(_frame([AS_OF], [10], promotion=True, discount=None))
    high = find_invalid_promotion_discount(_frame([AS_OF], [10], discount=100.01))
    low = find_invalid_promotion_discount(_frame([AS_OF], [10], discount=-0.01))
    absent = find_invalid_promotion_discount(
        _frame([AS_OF], [10], promotion=True, omit=("discount_pct",))
    )

    assert missing is not None and missing.count == 1
    assert high is not None and high.code == BlockingCode.INVALID_PROMOTION_DISCOUNT
    assert low is not None and low.count == 1
    assert absent is not None and absent.count == 1


def test_unknown_store_skips_blank_identifiers() -> None:
    dimensions = _dimensions()
    unknown = find_unknown_store(_frame([AS_OF], [10], store="store-99"), dimensions.stores)
    blank = find_unknown_store(_frame([AS_OF], [10], store=""), dimensions.stores)
    known = find_unknown_store(_frame([AS_OF], [10]), dimensions.stores)

    assert unknown is not None
    assert unknown.code == BlockingCode.UNKNOWN_STORE
    assert unknown.count == 1
    assert blank is None
    assert known is None


def test_unknown_sku_flags_identifiers_missing_from_the_dimension() -> None:
    dimensions = _dimensions()
    found = find_unknown_sku(_frame([AS_OF], [10], sku="sku-999"), dimensions.skus)

    assert found is not None
    assert found.code == BlockingCode.UNKNOWN_SKU
    assert found.count == 1
    assert find_unknown_sku(_frame([AS_OF], [10], sku=None), dimensions.skus) is None


def test_calendar_gap_flags_a_break_longer_than_one_day() -> None:
    contiguous = _frame([AS_OF, AS_OF - timedelta(days=1)], [10, 10])
    gap = _frame([AS_OF, AS_OF - timedelta(days=2), AS_OF - timedelta(days=3)], [10, 10, 10])

    found = find_calendar_gaps(gap)

    assert found is not None
    assert found.code == BlockingCode.CALENDAR_GAP
    assert found.count == 1
    assert find_calendar_gaps(contiguous) is None
    assert find_calendar_gaps(_frame([AS_OF], [10])) is None


def test_staleness_window_is_strict() -> None:
    fresh = _frame([AS_OF - timedelta(days=14)], [10])
    stale = _frame([AS_OF - timedelta(days=15)], [10])

    found = find_stale_observations(stale, as_of=AS_OF, staleness_days=14)

    assert found is not None
    assert found.code == BlockingCode.STALE_DATASET
    assert found.count == 1
    assert "2026-09-15" in found.message
    assert find_stale_observations(fresh, as_of=AS_OF, staleness_days=14) is None


def test_extreme_outlier_uses_a_strict_positive_median() -> None:
    days = [AS_OF - timedelta(days=offset) for offset in range(4)]
    at_threshold = _frame(days, [10, 10, 10, 200])
    above = _frame(days, [10, 10, 10, 201])
    zeros = _frame(days, [0, 0, 0, 0])

    found = find_extreme_outliers(above, multiple=20)

    assert found is not None
    assert found.code == BlockingCode.EXTREME_OUTLIER
    assert found.count == 1
    assert find_extreme_outliers(at_threshold, multiple=20) is None
    assert find_extreme_outliers(zeros, multiple=20) is None


def test_advisory_rates_do_not_invalidate_the_dataset() -> None:
    frame = _frame(
        [AS_OF - timedelta(days=offset) for offset in range(4)],
        [10, 0, 5, 8],
        category=["beverages", "beverages", "toys", "beverages"],
        stockout=[False, True, False, False],
    )
    dimensions = _dimensions()

    report = validate_dataset(frame, dimensions, ValidationConfig(as_of=AS_OF))

    assert report.status == "valid"
    assert report.blocking == ()
    assert report.stockout_rate == pytest.approx(0.25)
    assert stockout_rate(frame) == pytest.approx(0.25)
    assert zero_demand_fraction(frame) == pytest.approx(0.25)
    assert unexpected_category_count(frame, dimensions.categories) == 1
    assert [item.code for item in report.advisory] == [
        AdvisoryCode.STOCKOUT_RATE,
        AdvisoryCode.UNEXPECTED_CATEGORY,
        AdvisoryCode.ZERO_DEMAND_FRACTION,
    ]
    assert report.advisory[0].rate == pytest.approx(0.25)
    assert report.advisory[2].rate == pytest.approx(0.25)


def test_several_blocking_findings_keep_check_order() -> None:
    frame = _frame([AS_OF, AS_OF], [-1, 10], price=-2)

    report = validate_dataset(frame, _dimensions(), ValidationConfig(as_of=AS_OF))

    assert report.status == "invalid"
    assert [item.code for item in report.blocking] == [
        BlockingCode.DUPLICATE_GRAIN,
        BlockingCode.NEGATIVE_UNITS_SOLD,
        BlockingCode.NEGATIVE_PRICE,
    ]


def test_config_rejects_negative_thresholds() -> None:
    with pytest.raises(ValueError, match="staleness_days"):
        ValidationConfig(as_of=AS_OF, staleness_days=-1)
    with pytest.raises(ValueError, match="outlier_multiple"):
        ValidationConfig(as_of=AS_OF, outlier_multiple=0)


def test_valid_fixture_has_no_blocking_findings() -> None:
    report = validate_dataset(_load_observations("valid"), _load_dimensions(), _load_config())

    assert report.status == "valid"
    assert report.blocking == ()
    assert report.advisory == ()
    assert report.missing_value_rate == 0
    assert report.duplicate_rate == 0
    assert report.stockout_rate == 0


@pytest.mark.parametrize("code", list(BlockingCode))
def test_blocking_fixture_fails_only_that_rule(code: BlockingCode) -> None:
    report = validate_dataset(_load_observations(code.value), _load_dimensions(), _load_config())

    assert report.status == "invalid"
    assert [(item.code, item.count) for item in report.blocking] == [(code, EXPECTED_COUNTS[code])]
    assert report.advisory == ()


def test_report_json_is_written_beside_the_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    frame = _frame([AS_OF, AS_OF], [10, 11])
    report = validate_dataset(
        frame, _dimensions(), ValidationConfig(as_of=AS_OF), report_dir=dataset
    )

    document = json.loads((dataset / "quality_report.json").read_text(encoding="utf-8"))

    assert document["status"] == "invalid"
    assert document["blocking"][0]["code"] == BlockingCode.DUPLICATE_GRAIN
    assert document["blocking"][0]["message"]
    assert set(document["rates"]) == {"duplicates", "missing_values", "stockouts"}
    assert document == report.to_dict()
    with pytest.raises(ValueError, match="directory"):
        validate_dataset(
            frame,
            _dimensions(),
            ValidationConfig(as_of=AS_OF),
            report_dir=dataset / "quality_report.json",
        )


def test_generated_history_passes_with_advisory_findings(tmp_path: Path) -> None:
    summary = generate_dataset(
        tmp_path, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS
    )
    report = validate_dataset(
        pq.read_table(tmp_path / "observations.parquet"),
        DatasetDimensions(
            stores=pq.read_table(tmp_path / "stores.parquet"),
            skus=pq.read_table(tmp_path / "skus.parquet"),
            categories=pq.read_table(tmp_path / "categories.parquet"),
        ),
        ValidationConfig(as_of=date.fromisoformat(summary["date_max"])),
    )

    assert report.status == "valid"
    assert report.blocking == ()
    assert report.stockout_rate > 0
    assert {item.code for item in report.advisory} == {
        AdvisoryCode.STOCKOUT_RATE,
        AdvisoryCode.ZERO_DEMAND_FRACTION,
    }


def _frame(
    dates: Sequence[date | None],
    units: Sequence[int | None],
    *,
    store: object = "store-01",
    sku: object = "sku-001",
    category: object = "beverages",
    price: object = 4.5,
    promotion: object = False,
    discount: object = 0.0,
    stockout: object = False,
    omit: Sequence[str] = (),
) -> pa.Table:
    n = len(dates)
    columns = {
        "date": pa.array(list(dates), type=pa.date32()),
        "store_id": pa.array(_expand(store, n), type=pa.string()),
        "sku_id": pa.array(_expand(sku, n), type=pa.string()),
        "category_id": pa.array(_expand(category, n), type=pa.string()),
        "units_sold": pa.array(list(units), type=pa.int64()),
        "price": pa.array(_expand(price, n), type=pa.float64()),
        "promotion": pa.array(_expand(promotion, n), type=pa.bool_()),
        "discount_pct": pa.array(_expand(discount, n), type=pa.float64()),
        "stockout": pa.array(_expand(stockout, n), type=pa.bool_()),
    }
    for name in omit:
        del columns[name]
    return pa.table(columns)


def _expand(value: object, n: int) -> list[object]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return [value] * n
    items = list(value)
    if len(items) != n:
        raise ValueError("Column length does not match the date column.")
    return items


def _dimensions() -> DatasetDimensions:
    return DatasetDimensions(
        stores=pa.table(
            {
                "store_id": pa.array(["store-01"], type=pa.string()),
                "store_region": pa.array(["north"], type=pa.string()),
            }
        ),
        skus=pa.table(
            {
                "sku_id": pa.array(["sku-001"], type=pa.string()),
                "category_id": pa.array(["beverages"], type=pa.string()),
                "product_brand": pa.array(["Northwind"], type=pa.string()),
                "product_lifecycle": pa.array(["mature"], type=pa.string()),
            }
        ),
        categories=pa.table(
            {
                "category_id": pa.array(["beverages"], type=pa.string()),
                "category_name": pa.array(["Beverages"], type=pa.string()),
            }
        ),
    )


def _load_config() -> ValidationConfig:
    document = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    config = ValidationConfig(
        as_of=date.fromisoformat(document["as_of"]),
        staleness_days=int(document["staleness_days"]),
        outlier_multiple=float(document["outlier_multiple"]),
    )
    assert config == ValidationConfig(as_of=AS_OF)
    return config


def _load_dimensions() -> DatasetDimensions:
    return DatasetDimensions(
        stores=_read_csv("dimensions/stores.csv"),
        skus=_read_csv("dimensions/skus.csv"),
        categories=_read_csv("dimensions/categories.csv"),
    )


def _load_observations(name: str) -> pa.Table:
    return _read_csv(f"observations/{name}.csv")


def _read_csv(relative: str) -> pa.Table:
    path = FIXTURES / relative
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    column_types = {name: COLUMN_TYPES[name] for name in header}
    return pa_csv.read_csv(
        path,
        convert_options=pa_csv.ConvertOptions(column_types=column_types, strings_can_be_null=True),
    )
