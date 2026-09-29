"""Model versions and the statuses a promotion decision can set."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType


class ModelStatus(StrEnum):
    """Lifecycle of one registered model version.

    Training finishes at ``TRAINED``. Scoring moves it to ``EVALUATED``. The
    quality gate then sets ``PENDING_APPROVAL`` or ``REJECTED``. A person
    moves a pending model to ``APPROVED`` or ``REJECTED``. Only an approved
    model can enter ``PRODUCTION``.
    """

    TRAINED = "TRAINED"
    EVALUATED = "EVALUATED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"
    REJECTED = "REJECTED"


class RegistryStatus(StrEnum):
    """Approval state stored in the cloud model registry.

    ``APPROVED`` and ``PRODUCTION`` both appear here as ``Approved``. A model
    that is still training or only evaluated is not recorded yet.
    """

    PENDING_MANUAL_APPROVAL = "PendingManualApproval"
    APPROVED = "Approved"
    REJECTED = "Rejected"


@dataclass(frozen=True)
class ModelVersion:
    """One registered model and the metrics used to decide promotion.

    ``registry_arn`` is empty until a cloud registry records the model.
    ``registry_status`` is set at the same time. ``approved_at`` is set only
    after a person approves it. ``approved_by`` and ``rejected_by`` store the
    actor id of that decision. ``metrics`` holds the evaluation numbers copied
    onto the version. A finished training job starts here as ``TRAINED`` with
    no approval timestamp.
    """

    id: str
    model_family: str
    version: str
    training_run_id: str
    status: ModelStatus
    metrics: Mapping[str, object]
    registry_arn: str
    approved_at: datetime | None
    created_at: datetime
    dataset_version: str | None = None
    approved_by: str | None = None
    rejected_by: str | None = None
    rejection_reason: str | None = None
    registry_status: RegistryStatus | None = None

    def __post_init__(self) -> None:
        _require_text(self.id, "Model id")
        _require_text(self.model_family, "Model family")
        _require_text(self.version, "Model version")
        _require_text(self.training_run_id, "Training run id")
        if not isinstance(self.status, ModelStatus):
            raise ValueError("Model status must be a ModelStatus value.")
        if not isinstance(self.registry_arn, str):
            raise ValueError("registry_arn must be a string.")
        if self.registry_status is not None and not isinstance(
            self.registry_status, RegistryStatus
        ):
            raise ValueError("registry_status must be a RegistryStatus value.")
        if self.dataset_version is not None:
            _require_text(self.dataset_version, "Dataset version")
        if self.approved_at is not None and self.approved_at.tzinfo is None:
            raise ValueError("approved_at must include a timezone.")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must include a timezone.")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this model version."""

        return {
            "approved_at": None if self.approved_at is None else self.approved_at.isoformat(),
            "approved_by": self.approved_by,
            "created_at": self.created_at.isoformat(),
            "dataset_version": self.dataset_version,
            "id": self.id,
            "metrics": dict(self.metrics),
            "model_family": self.model_family,
            "registry_arn": self.registry_arn,
            "registry_status": None if self.registry_status is None else self.registry_status.value,
            "rejected_by": self.rejected_by,
            "rejection_reason": self.rejection_reason,
            "status": self.status.value,
            "training_run_id": self.training_run_id,
            "version": self.version,
        }


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{label} must be a non-empty string.")
