"""Model training entry points.

:class:`GradientBoostingForecaster` fits one global LightGBM model. ``fit``
and ``predict`` both call :func:`forecastops_ml.features.build_features`.

:func:`export_deepar_channels` converts retail observations into DeepAR train
and test channels. :class:`FakeDeepARRunner` returns a fixture forecast for
local tests. :func:`submit_deepar_training` submits one CPU job when the
training flags are open.
"""

from forecastops_ml.training.deepar_export import (
    DeepARChannels,
    DeepARRecord,
    export_deepar_channels,
)
from forecastops_ml.training.deepar_forecast import (
    PREDICTION_QUANTILES,
    FakeDeepARRunner,
    ForecastPoint,
    load_fixture_forecast,
    map_deepar_predictions,
)
from forecastops_ml.training.deepar_job import (
    CPU_INSTANCE_TYPE,
    TrainingGates,
    block_reason,
    build_training_job_request,
    gates_from_environ,
    submit_deepar_training,
)
from forecastops_ml.training.gradient_boosting import GradientBoostingForecaster

__all__ = [
    "CPU_INSTANCE_TYPE",
    "PREDICTION_QUANTILES",
    "DeepARChannels",
    "DeepARRecord",
    "FakeDeepARRunner",
    "ForecastPoint",
    "GradientBoostingForecaster",
    "TrainingGates",
    "block_reason",
    "build_training_job_request",
    "export_deepar_channels",
    "gates_from_environ",
    "load_fixture_forecast",
    "map_deepar_predictions",
    "submit_deepar_training",
]
