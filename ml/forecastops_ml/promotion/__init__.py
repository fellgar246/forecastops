"""Promotion gate and human approval for model versions.

A finished training job stays ``TRAINED``. The gate may reject it or leave it
pending. A person approves or rejects a pending model. Production is a later
human transition from ``APPROVED``.
"""

from forecastops_ml.promotion.gate import (
    HUMAN_REJECTION,
    QUALITY_GATE,
    GateChecks,
    PromotionDecision,
    SegmentRegression,
    empirical_p90_coverage,
    gate,
)
from forecastops_ml.promotion.models import ModelStatus, ModelVersion, RegistryStatus
from forecastops_ml.promotion.reference import reference_report, seasonal_naive_for
from forecastops_ml.promotion.service import PromotionError, PromotionService
from forecastops_ml.promotion.thresholds import (
    DEFAULT_MAX_BIAS,
    DEFAULT_MAX_SEGMENT_WAPE_REGRESSION,
    DEFAULT_MIN_P90_COVERAGE,
    PromotionThresholds,
)

__all__ = [
    "DEFAULT_MAX_BIAS",
    "DEFAULT_MAX_SEGMENT_WAPE_REGRESSION",
    "DEFAULT_MIN_P90_COVERAGE",
    "HUMAN_REJECTION",
    "QUALITY_GATE",
    "GateChecks",
    "ModelStatus",
    "ModelVersion",
    "PromotionDecision",
    "PromotionError",
    "PromotionService",
    "PromotionThresholds",
    "RegistryStatus",
    "SegmentRegression",
    "empirical_p90_coverage",
    "gate",
    "reference_report",
    "seasonal_naive_for",
]
