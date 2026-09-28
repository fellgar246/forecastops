"""Series filters and totals stay on the API side of the dashboard."""

from datetime import date

from forecastops_api.series import Observation, SeriesPoint, assemble_series, point_matches


def test_store_and_category_filters_keep_matching_points() -> None:
    points = [
        SeriesPoint("store-1|sku-1", date(2026, 10, 1), 1.0, 2.0, 3.0, None),
        SeriesPoint("store-2|sku-2", date(2026, 10, 1), 4.0, 5.0, 6.0, None),
    ]
    categories = {"sku-1": "grocery", "sku-2": "electronics"}
    selected = [
        point
        for point in points
        if point_matches(
            point.series_id,
            store="store-1",
            sku=None,
            category="grocery",
            sku_categories=categories,
        )
    ]

    series = assemble_series(selected, [], horizon_days=7, category="grocery")

    assert [item.series_id for item in series.items] == ["store-1|sku-1"]
    assert series.p50_total == 2.0
    assert series.p10_total == 1.0
    assert series.series[0].point_count == 1


def test_missing_quantile_has_no_total() -> None:
    points = [SeriesPoint("store-1|sku-1", date(2026, 10, 1), None, 2.0, None, None)]

    series = assemble_series(points, [], horizon_days=7, category=None)

    assert series.p50_total == 2.0
    assert series.p10_total is None
    assert series.p90_total is None
    assert series.daily[0].p50 == 2.0


def test_history_sums_observations_before_the_cutoff() -> None:
    points = [SeriesPoint("store-1|sku-1", date(2026, 10, 3), None, 2.0, None, None)]
    observations = [
        Observation(date(2026, 10, 1), "store-1|sku-1", "grocery", 4.0),
        Observation(date(2026, 10, 1), "store-2|sku-9", "grocery", 9.0),
        Observation(date(2026, 10, 3), "store-1|sku-1", "grocery", 7.0),
    ]

    series = assemble_series(points, observations, horizon_days=7, category=None)

    assert [(item.date, item.actual) for item in series.history] == [(date(2026, 10, 1), 4.0)]
    assert series.cutoff == date(2026, 10, 3)
