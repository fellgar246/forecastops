"""Feature construction shared by training, evaluation, and inference.

:func:`build_features` is the only lag and rolling-window implementation.
Training, backtesting, and inference call it. A feature that is not knowable
at prediction time is rejected.
"""

from forecastops_ml.features.build import (
    FEATURE_SPECS,
    MODEL_FEATURES,
    FeatureFrame,
    FeatureSpec,
    build_features,
    metadata_table,
    require_prediction_time_features,
)
from forecastops_ml.features.week import aggregate_store_sku_week

__all__ = [
    "FEATURE_SPECS",
    "MODEL_FEATURES",
    "FeatureFrame",
    "FeatureSpec",
    "aggregate_store_sku_week",
    "build_features",
    "metadata_table",
    "require_prediction_time_features",
]
