"""Choose a model registry for the cloud profile.

Local mode returns none, so approval updates only the internal model row and
forecasts do not consult a registry.
"""

from forecastops_api.settings import ExecutionMode, Settings
from forecastops_ml.registry import ModelRegistry, open_model_registry


def select_model_registry(settings: Settings) -> ModelRegistry | None:
    """Return a live registry when cloud forecasts are enabled.

    The client is constructed only when execution is in AWS and SageMaker is
    enabled. Local mode never loads the cloud SDK.
    """

    if settings.execution_mode is not ExecutionMode.AWS or not settings.sagemaker_enabled:
        return None
    return open_model_registry(settings.aws_region)
