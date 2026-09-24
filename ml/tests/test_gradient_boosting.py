"""Gradient boosting scores, determinism, and SHAP drivers."""

from datetime import date, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forecastops_ml.baselines import local_catalog
from forecastops_ml.data.synthetic import DEFAULT_MAX_ROWS, DEFAULT_SEED, generate_dataset
from forecastops_ml.evaluation import MIN_PUBLISHED_FOLDS, Backtester
from forecastops_ml.explainability import shap_drivers
from forecastops_ml.features import aggregate_store_sku_week
from forecastops_ml.training import GradientBoostingForecaster

RELATIVE_TOLERANCE = 1e-6
START = date(2026, 8, 3)


def test_shap_drivers_are_ordered_by_magnitude() -> None:
    attribution = shap_drivers(
        [0.2, -1.5, 0.0, 3.0, -0.4, 1.0, -4.0],
        ["a", "b", "c", "d", "e", "f", "g"],
        top_n=2,
    )

    assert [item.feature for item in attribution.positive_drivers] == ["d", "f"]
    assert [item.shap_value for item in attribution.positive_drivers] == [3.0, 1.0]
    assert [item.feature for item in attribution.negative_drivers] == ["g", "b"]
    assert [item.shap_value for item in attribution.negative_drivers] == [-4.0, -1.5]


def test_daily_grain_requires_the_full_profile() -> None:
    with pytest.raises(ValueError, match="profile is full"):
        GradientBoostingForecaster(grain="day", profile="test")


def test_gradient_boosting_scores_on_the_test_fixture(tmp_path) -> None:
    weekly = _weekly_fixture(tmp_path)
    model = GradientBoostingForecaster(seed=DEFAULT_SEED)
    report = Backtester().evaluate(
        model,
        weekly,
        MIN_PUBLISHED_FOLDS,
        horizon=1,
    )

    assert model.model_family == "gradient_boosting"
    assert model.training_config["library"] == "lightgbm"
    assert report.global_metrics.row_count > 0
    assert report.global_metrics.wape >= 0.0

    catalog = local_catalog()
    table = catalog.compare(weekly, folds=MIN_PUBLISHED_FOLDS, horizon=1)
    families = [row.family for row in table.models]
    assert "gradient_boosting" in families
    scored = next(row for row in table.models if row.family == "gradient_boosting")
    assert scored.wape >= 0.0


def test_same_seed_reproduces_forecasts(tmp_path) -> None:
    weekly = _weekly_fixture(tmp_path)
    first = _forecasts(weekly)
    second = _forecasts(weekly)

    assert first == pytest.approx(second, rel=RELATIVE_TOLERANCE)


def test_attribute_lists_positive_and_negative_drivers() -> None:
    model = GradientBoostingForecaster(seed=DEFAULT_SEED)
    history = _contrast_history()
    model.fit(history, None)
    promoted = _future_row(START + timedelta(weeks=32), promotion=True, holiday=True)
    attribution = model.attribute(promoted)

    assert attribution.positive_drivers
    assert attribution.negative_drivers
    assert len(attribution.positive_drivers) <= 5
    assert len(attribution.negative_drivers) <= 5
    positive = [item.shap_value for item in attribution.positive_drivers]
    negative = [item.shap_value for item in attribution.negative_drivers]
    assert positive == sorted(positive, reverse=True)
    assert negative == sorted(negative)
    assert all(value > 0.0 for value in positive)
    assert all(value < 0.0 for value in negative)
    names = {item.feature for item in attribution.positive_drivers + attribution.negative_drivers}
    assert "cause" not in names


def _forecasts(frame: pa.Table) -> list[float]:
    model = GradientBoostingForecaster(seed=DEFAULT_SEED)
    report_dates = sorted(set(frame.column("date").to_pylist()))
    origin = report_dates[-1]
    history = frame.filter(pa.array([day < origin for day in frame.column("date").to_pylist()]))
    future = frame.filter(pa.array([day == origin for day in frame.column("date").to_pylist()]))
    model.fit(history, {"fold": 1, "origin": origin.isoformat()})
    return list(model.predict(1, future.drop(["units_sold"])).p50)


def _weekly_fixture(directory) -> pa.Table:
    generate_dataset(directory, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    daily = pq.read_table(directory / "observations.parquet")
    return aggregate_store_sku_week(daily)


def _contrast_history() -> pa.Table:
    rows: list[dict[str, object]] = []
    for offset in range(32):
        day = START + timedelta(weeks=offset)
        promoted = offset % 2 == 0
        holiday = offset % 4 >= 2
        units = 40 + (30 if promoted else 0) - (25 if holiday else 0)
        rows.append(
            {
                "date": day,
                "store_id": "store-01",
                "sku_id": "sku-1",
                "category_id": "beverages",
                "units_sold": units,
                "price": 4.0,
                "discount_pct": 0.0,
                "promotion": promoted,
                "holiday": holiday,
            }
        )
    return pa.Table.from_pylist(rows)


def _future_row(day: date, *, promotion: bool, holiday: bool = False) -> pa.Table:
    return pa.table(
        {
            "date": pa.array([day], type=pa.date32()),
            "store_id": pa.array(["store-01"], type=pa.string()),
            "sku_id": pa.array(["sku-1"], type=pa.string()),
            "category_id": pa.array(["beverages"], type=pa.string()),
            "price": pa.array([4.0], type=pa.float64()),
            "discount_pct": pa.array([0.0], type=pa.float64()),
            "promotion": pa.array([promotion], type=pa.bool_()),
            "holiday": pa.array([holiday], type=pa.bool_()),
        }
    )
