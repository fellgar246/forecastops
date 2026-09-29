"""Choose whether a model version may serve a forecast."""

from forecastops_ml.promotion.models import ModelStatus, RegistryStatus

_FORECASTABLE = frozenset({ModelStatus.APPROVED, ModelStatus.PRODUCTION})


class ForecastSelectionError(ValueError):
    """The model cannot serve this forecast."""


def select_forecast_model(
    *,
    internal_status: ModelStatus,
    registry_status: RegistryStatus | None = None,
    cloud: bool,
) -> None:
    """Allow a forecast, or refuse a model that is not approved.

    Local selection uses ``internal_status`` alone. A cloud forecast also
    requires registry status ``Approved``.
    """

    if internal_status not in _FORECASTABLE:
        raise ForecastSelectionError("Forecasts require a model in APPROVED or PRODUCTION status.")
    if cloud and registry_status is not RegistryStatus.APPROVED:
        raise ForecastSelectionError("Cloud forecasts require a model registry status of Approved.")
