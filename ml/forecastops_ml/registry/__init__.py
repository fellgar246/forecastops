"""Cloud model registry for versions a person must approve.

Registration happens after evaluation and the quality gate. A cloud forecast
loads a model only when the registry status is ``Approved`` and the internal
status is ``APPROVED`` or ``PRODUCTION``. Local forecasts ignore the registry.
"""

from forecastops_ml.registry.adapter import (
    BotoModelRegistry,
    MemoryModelRegistry,
    ModelRegistry,
    RegistryEntry,
    RegistryError,
    open_model_registry,
)
from forecastops_ml.registry.selection import ForecastSelectionError, select_forecast_model
from forecastops_ml.registry.status import registry_status_for

__all__ = [
    "BotoModelRegistry",
    "ForecastSelectionError",
    "MemoryModelRegistry",
    "ModelRegistry",
    "RegistryEntry",
    "RegistryError",
    "open_model_registry",
    "registry_status_for",
    "select_forecast_model",
]
