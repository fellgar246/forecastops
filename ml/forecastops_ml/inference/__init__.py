"""Batch inference over stored models.

A forecast run moves ``QUEUED`` → ``RUNNING`` → ``SUCCEEDED`` or ``FAILED``.
Local execution uses the local catalog, or a fake quantile model for
``deepar`` when no cloud job is configured. That fake model emits P10, P50,
and P90. Cloud execution submits a SageMaker batch transform for ``deepar``
and reads the artifact under ``forecasts/``. It does not create a real-time
or serverless endpoint.

Gradient boosting forecasts are ``GradientBoostingForecaster.predict``. That
path calls :func:`forecastops_ml.features.build_features` with the training
cutoff, then scores the fitted tree model.
"""

from forecastops_ml.inference.batch import (
    BatchInference,
    BatchTransformClient,
    BotoBatchTransformClient,
    BotoForecastObjects,
    ForecastObjects,
    MemoryForecastObjects,
    open_batch_inference,
)
from forecastops_ml.inference.errors import (
    BatchCapacityError,
    InferenceError,
    public_inference_error,
)
from forecastops_ml.inference.local import local_family_forecast, quantile_forecast
from forecastops_ml.inference.machine import (
    admit_forecast,
    advance_status,
    execute_prediction,
)
from forecastops_ml.inference.points import (
    StoredForecastPoint,
    forecast_document,
    load_forecast_document,
)

__all__ = [
    "BatchCapacityError",
    "BatchInference",
    "BatchTransformClient",
    "BotoBatchTransformClient",
    "BotoForecastObjects",
    "ForecastObjects",
    "InferenceError",
    "MemoryForecastObjects",
    "StoredForecastPoint",
    "admit_forecast",
    "advance_status",
    "execute_prediction",
    "forecast_document",
    "load_forecast_document",
    "local_family_forecast",
    "open_batch_inference",
    "public_inference_error",
    "quantile_forecast",
]
