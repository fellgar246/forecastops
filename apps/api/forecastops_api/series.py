"""Filter stored forecast points and total them for the dashboard.

Totals are computed here so the web app can display them without summing.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date, timedelta
from typing import NamedTuple

from forecastops_api.schemas import (
    DailyDemand,
    ForecastPointResponse,
    ForecastSeries,
    HistoryPoint,
    SeriesSummary,
)


class SeriesPoint(NamedTuple):
    """One stored forecast point."""

    series_id: str
    day: date
    p10: float | None
    p50: float
    p90: float | None
    actual: float | None


class Observation(NamedTuple):
    """One historical demand row used for the history line."""

    day: date
    series_id: str
    category_id: str
    units: float


def split_series_id(series_id: str) -> tuple[str, str]:
    """Return ``(store_id, sku_id)`` from ``store|sku``."""

    store, separator, sku = series_id.partition("|")
    if not separator:
        return series_id, ""
    return store, sku


def point_matches(
    series_id: str,
    *,
    store: str | None,
    sku: str | None,
    category: str | None,
    sku_categories: dict[str, str],
) -> bool:
    """Return whether a series matches the requested filters."""

    series_store, series_sku = split_series_id(series_id)
    store_ok = store is None or series_store == store
    sku_ok = sku is None or series_sku == sku
    category_ok = category is None or sku_categories.get(series_sku) == category
    return store_ok and sku_ok and category_ok


def empty_series() -> ForecastSeries:
    """Return a series payload with no points and no totals."""

    return ForecastSeries(
        items=[],
        series=[],
        history=[],
        daily=[],
        cutoff=None,
        p10_total=None,
        p50_total=None,
        p90_total=None,
    )


def assemble_series(
    points: Sequence[SeriesPoint],
    observations: Sequence[Observation],
    *,
    horizon_days: int,
    category: str | None,
) -> ForecastSeries:
    """Build the series payload, including per-series and daily totals."""

    if not points:
        return empty_series()
    items = [
        ForecastPointResponse(
            series_id=point.series_id,
            date=point.day,
            p10=point.p10,
            p50=point.p50,
            p90=point.p90,
            actual=point.actual,
        )
        for point in points
    ]
    cutoff = min(point.day for point in points)
    allowed = {point.series_id for point in points}
    return ForecastSeries(
        items=items,
        series=_series_summaries(points),
        history=_history(observations, allowed, cutoff, horizon_days, category),
        daily=_daily(points),
        cutoff=cutoff,
        p10_total=_sum_optional(point.p10 for point in points),
        p50_total=_sum_optional(point.p50 for point in points),
        p90_total=_sum_optional(point.p90 for point in points),
    )


def _series_summaries(points: Sequence[SeriesPoint]) -> list[SeriesSummary]:
    grouped: dict[str, list[SeriesPoint]] = defaultdict(list)
    for point in points:
        grouped[point.series_id].append(point)
    summaries = [
        SeriesSummary(
            series_id=series_id,
            point_count=len(rows),
            p10_total=_sum_optional(row.p10 for row in rows),
            p50_total=_sum_optional(row.p50 for row in rows),
            p90_total=_sum_optional(row.p90 for row in rows),
        )
        for series_id, rows in grouped.items()
    ]
    return sorted(summaries, key=lambda item: item.series_id)


def _daily(points: Sequence[SeriesPoint]) -> list[DailyDemand]:
    grouped: dict[date, list[SeriesPoint]] = defaultdict(list)
    for point in points:
        grouped[point.day].append(point)
    return [
        DailyDemand(
            date=day,
            p10=_sum_optional(row.p10 for row in rows),
            p50=_sum_optional(row.p50 for row in rows),
            p90=_sum_optional(row.p90 for row in rows),
            actual=_sum_optional(row.actual for row in rows),
        )
        for day, rows in sorted(grouped.items())
    ]


def _history(
    observations: Sequence[Observation],
    allowed: set[str],
    cutoff: date,
    horizon_days: int,
    category: str | None,
) -> list[HistoryPoint]:
    window = max(horizon_days, 28)
    start = cutoff - timedelta(days=window)
    totals: dict[date, float] = defaultdict(float)
    for row in observations:
        if row.series_id not in allowed or row.day >= cutoff or row.day < start:
            continue
        if category and row.category_id != category:
            continue
        totals[row.day] += row.units
    return [HistoryPoint(date=day, actual=totals[day]) for day in sorted(totals)]


def _sum_optional(values: Iterable[float | None]) -> float | None:
    """Sum numbers. Return none when any value is missing or the sequence is empty."""

    total = 0.0
    seen = False
    for value in values:
        if value is None:
            return None
        total += float(value)
        seen = True
    if not seen:
        return None
    return total
