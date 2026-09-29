"""Map an internal model status to a cloud registry status."""

from forecastops_ml.promotion.models import ModelStatus, RegistryStatus

_REGISTRY_STATUS: dict[ModelStatus, RegistryStatus] = {
    ModelStatus.PENDING_APPROVAL: RegistryStatus.PENDING_MANUAL_APPROVAL,
    ModelStatus.APPROVED: RegistryStatus.APPROVED,
    ModelStatus.PRODUCTION: RegistryStatus.APPROVED,
    ModelStatus.REJECTED: RegistryStatus.REJECTED,
}


def registry_status_for(status: ModelStatus) -> RegistryStatus:
    """Return the registry status that records ``status``.

    Training and evaluation are not registry states. ``APPROVED`` and
    ``PRODUCTION`` both record as ``Approved``.
    """

    try:
        return _REGISTRY_STATUS[status]
    except KeyError as exc:
        raise ValueError(f"{status.value} is not recorded in the model registry.") from exc
