"""Lags, rolling windows, cutoff alignment, and leakage."""

from datetime import date, timedelta

import pyarrow as pa
import pytest

from forecastops_ml.features import (
    MODEL_FEATURES,
    aggregate_store_sku_week,
    build_features,
    require_prediction_time_features,
)

START = date(2026, 8, 3)


def test_lags_and_rolling_windows_use_only_earlier_days() -> None:
    units = [100, 1, 1, 1, 1, 1, 1, 1]
    frame = _daily(units)
    features = build_features(frame, START + timedelta(days=7), {"grain": "day"})
    last = _row(features.frame, START + timedelta(days=7))

    assert last["lag_1"] == 1.0
    assert last["lag_7"] == 100.0
    assert last["rolling_mean_7"] == pytest.approx((100 + 6) / 7)
    assert last["rolling_mean_7"] != pytest.approx(1.0)
    assert last["day_of_week"] == (START + timedelta(days=7)).weekday()
    assert last["week_of_year"] == (START + timedelta(days=7)).isocalendar().week
    assert last["month"] == 8
    assert last["price"] == 2.5
    assert last["promotion"] is True
    assert last["holiday"] is False


def test_weekly_lag_is_one_week_and_metadata_covers_model_inputs() -> None:
    days = [START + timedelta(weeks=offset) for offset in range(4)]
    frame = _frame(days, [10, 20, 30, 40])
    features = build_features(frame, days[-1], {"grain": "week"})
    last = _row(features.frame, days[-1])

    assert last["lag_1"] == 30.0
    assert last["lag_7"] is None
    names = features.metadata.column("name").to_pylist()
    flags = features.metadata.column("available_at_prediction_time").to_pylist()
    assert names == list(MODEL_FEATURES)
    assert set(features.metadata.column_names) == {
        "available_at_prediction_time",
        "description",
        "dtype",
        "name",
        "source",
    }
    assert all(flags)
    assert "stockout" not in features.frame.column_names
    assert "stock_available" not in features.frame.column_names


def test_cutoff_blocks_targets_after_the_cutoff() -> None:
    days = [START + timedelta(days=offset) for offset in range(6)]
    frame = _daily_on(days, [1, 2, 3, 4, 5, 6])
    cutoff = days[3]
    features = build_features(frame, cutoff, {"grain": "day"})
    on_cutoff = _row(features.frame, cutoff)
    after = _row(features.frame, days[5])

    assert on_cutoff["lag_1"] == 3.0
    assert after["lag_1"] is None
    assert after["lag_7"] is None


def test_mutating_targets_after_the_cutoff_leaves_history_unchanged() -> None:
    days = [START + timedelta(days=offset) for offset in range(8)]
    frame = _daily_on(days, [float(index + 1) for index in range(8)])
    cutoff = days[4]
    original = build_features(frame, cutoff, {"grain": "day"})
    mutated = frame.set_column(
        frame.schema.get_field_index("units_sold"),
        "units_sold",
        pa.array([1, 2, 3, 4, 5, 9_999, 9_999, 9_999], type=pa.int64()),
    )
    rebuilt = build_features(mutated, cutoff, {"grain": "day"})

    assert _rows_through(original.frame, cutoff) == _rows_through(rebuilt.frame, cutoff)
    assert _row(rebuilt.frame, days[6])["lag_1"] is None
    assert _row(rebuilt.frame, days[5])["lag_1"] == 5.0


def test_unavailable_features_are_rejected() -> None:
    frame = _daily([1, 2, 3])
    with pytest.raises(ValueError, match="not available at prediction time"):
        build_features(frame, START, {"grain": "day", "features": ["lag_1", "stockout"]})
    bad = pa.table(
        {
            "name": ["stockout"],
            "description": ["Stock-out flag."],
            "dtype": ["bool"],
            "source": ["stockout"],
            "available_at_prediction_time": [False],
        }
    )
    with pytest.raises(ValueError, match="not available at prediction time"):
        require_prediction_time_features(bad)


def test_store_sku_week_preserves_units_and_starts_on_monday() -> None:
    thursday = date(2026, 8, 6)
    friday = thursday + timedelta(days=1)
    next_monday = date(2026, 8, 10)
    frame = pa.table(
        {
            "date": pa.array([thursday, friday, next_monday], type=pa.date32()),
            "store_id": pa.array(["store-01", "store-01", "store-01"], type=pa.string()),
            "sku_id": pa.array(["sku-1", "sku-1", "sku-1"], type=pa.string()),
            "category_id": pa.array(["beverages", "beverages", "beverages"], type=pa.string()),
            "units_sold": pa.array([2, 4, 10], type=pa.int64()),
            "price": pa.array([3.0, 6.0, 5.0], type=pa.float64()),
            "discount_pct": pa.array([10.0, 20.0, 0.0], type=pa.float64()),
            "promotion": pa.array([False, True, False], type=pa.bool_()),
            "holiday": pa.array([False, False, True], type=pa.bool_()),
            "stockout": pa.array([True, False, False], type=pa.bool_()),
        }
    )
    weekly = aggregate_store_sku_week(frame)

    assert weekly.column("date").to_pylist() == [date(2026, 8, 3), next_monday]
    assert weekly.column("units_sold").to_pylist() == [6, 10]
    assert weekly.column("price").to_pylist() == [pytest.approx(5.0), pytest.approx(5.0)]
    assert weekly.column("promotion").to_pylist() == [True, False]
    assert weekly.column("holiday").to_pylist() == [False, True]
    assert "stockout" not in weekly.column_names
    assert sum(weekly.column("units_sold").to_pylist()) == 16


def _daily(units: list[int]) -> pa.Table:
    days = [START + timedelta(days=offset) for offset in range(len(units))]
    return _daily_on(days, [float(value) for value in units])


def _daily_on(days: list[date], units: list[float]) -> pa.Table:
    return _frame(days, units)


def _frame(days: list[date], units: list[float]) -> pa.Table:
    count = len(days)
    return pa.table(
        {
            "date": pa.array(days, type=pa.date32()),
            "store_id": pa.array(["store-01"] * count, type=pa.string()),
            "sku_id": pa.array(["sku-1"] * count, type=pa.string()),
            "category_id": pa.array(["beverages"] * count, type=pa.string()),
            "units_sold": pa.array(units, type=pa.float64()),
            "price": pa.array([2.5] * count, type=pa.float64()),
            "discount_pct": pa.array([10.0] * count, type=pa.float64()),
            "promotion": pa.array([True] * count, type=pa.bool_()),
            "holiday": pa.array([False] * count, type=pa.bool_()),
        }
    )


def _row(frame: pa.Table, day: date) -> dict[str, object]:
    rows = frame.to_pylist()
    matched = [row for row in rows if row["date"] == day]
    assert len(matched) == 1
    return matched[0]


def _rows_through(frame: pa.Table, cutoff: date) -> list[dict[str, object]]:
    rows = [row for row in frame.to_pylist() if row["date"] <= cutoff]
    return sorted(rows, key=lambda row: row["date"])
