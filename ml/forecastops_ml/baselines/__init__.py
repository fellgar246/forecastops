"""Naive, seasonal-naive, and Holt-Winters baselines, plus the local catalog.

The default catalog scores ``naive``, weekly ``seasonal_naive``,
``holt_winters``, and ``gradient_boosting``. Yearly seasonal naive is
available for weekly history once that history covers 52 weeks. Holt-Winters
fits category-week aggregates. Gradient boosting fits store-SKU weeks.
"""

from forecastops_ml.baselines.catalog import (
    COMPARISON_FILENAME,
    ComparisonRow,
    ComparisonTable,
    ModelCatalog,
    local_catalog,
    write_comparison,
    yearly_season_available,
)
from forecastops_ml.baselines.holt_winters import (
    MIN_HISTORY_WEEKS,
    SEASON_LENGTH,
    HoltWintersForecaster,
)
from forecastops_ml.baselines.naive import (
    WEEKLY_SEASON_LENGTH,
    YEARLY_SEASON_LENGTH,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    weekly_seasonal_naive,
    yearly_seasonal_naive,
)

__all__ = [
    "COMPARISON_FILENAME",
    "MIN_HISTORY_WEEKS",
    "SEASON_LENGTH",
    "WEEKLY_SEASON_LENGTH",
    "YEARLY_SEASON_LENGTH",
    "ComparisonRow",
    "ComparisonTable",
    "HoltWintersForecaster",
    "ModelCatalog",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "local_catalog",
    "weekly_seasonal_naive",
    "write_comparison",
    "yearly_season_available",
    "yearly_seasonal_naive",
]
