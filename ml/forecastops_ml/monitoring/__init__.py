"""Drift monitoring for a recent window against the training baseline.

The report status is ``HEALTHY``, ``WARNING``, or ``RETRAIN_RECOMMENDED``.
A recommendation does not train or approve a model.
"""

from forecastops_ml.monitoring.report import (
    HEALTHY,
    RETRAIN_RECOMMENDED,
    WARNING,
    CoverageDrift,
    FrequencyDrift,
    MonitoringReport,
    NumericDrift,
    StatusName,
    build_monitoring_report,
    dataset_age_days,
    monitoring_status,
    split_recent_window,
)
from forecastops_ml.monitoring.statistics import (
    PSI_EPSILON,
    absolute_frequency_deltas,
    frequency_delta,
    kolmogorov_smirnov,
    population_stability_index,
)
from forecastops_ml.monitoring.thresholds import (
    DEFAULT_FREQUENCY_WARNING_DELTA,
    DEFAULT_MAX_DATASET_AGE_DAYS,
    DEFAULT_PSI_RETRAIN,
    DEFAULT_PSI_WARNING,
    DEFAULT_RECENT_WINDOW_DAYS,
    DEFAULT_WAPE_DEGRADATION_LIMIT,
    DriftThresholds,
)

__all__ = [
    "DEFAULT_FREQUENCY_WARNING_DELTA",
    "DEFAULT_MAX_DATASET_AGE_DAYS",
    "DEFAULT_PSI_RETRAIN",
    "DEFAULT_PSI_WARNING",
    "DEFAULT_RECENT_WINDOW_DAYS",
    "DEFAULT_WAPE_DEGRADATION_LIMIT",
    "HEALTHY",
    "PSI_EPSILON",
    "RETRAIN_RECOMMENDED",
    "WARNING",
    "CoverageDrift",
    "DriftThresholds",
    "FrequencyDrift",
    "MonitoringReport",
    "NumericDrift",
    "StatusName",
    "absolute_frequency_deltas",
    "build_monitoring_report",
    "dataset_age_days",
    "frequency_delta",
    "kolmogorov_smirnov",
    "monitoring_status",
    "population_stability_index",
    "split_recent_window",
]
