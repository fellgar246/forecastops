"""Register a model version and record a person's approval decision.

The adapter is optional. Local promotion leaves it unset and updates only the
internal model row. When it is set, register, approve, and reject update the
cloud registry in the same operation as the internal status.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from forecastops_ml.promotion.models import RegistryStatus

_GROUP_PREFIX = "forecastops"


class RegistryError(ValueError):
    """The model registry refused the operation."""


@dataclass(frozen=True)
class RegistryEntry:
    """One model package recorded by the registry."""

    arn: str
    version: str
    status: RegistryStatus
    actor_id: str | None = None


class ModelRegistry(Protocol):
    """Register a candidate and record a later human decision."""

    def register(
        self,
        *,
        model_id: str,
        model_family: str,
        version: str,
        status: RegistryStatus,
    ) -> RegistryEntry:
        """Record ``model_id`` with ``status`` and return its registry identity."""

    def approve(self, arn: str, *, actor_id: str) -> RegistryEntry:
        """Set ``arn`` to ``Approved`` for ``actor_id``."""

    def reject(self, arn: str, *, actor_id: str) -> RegistryEntry:
        """Set ``arn`` to ``Rejected`` for ``actor_id``."""

    def get(self, arn: str) -> RegistryEntry:
        """Return the package stored at ``arn``."""


class SageMakerRegistryApi(Protocol):
    """Subset of the model-registry API used by :class:`BotoModelRegistry`."""

    def create_model_package_group(self, **kwargs: Any) -> Mapping[str, Any]:
        """Create one model package group."""

    def describe_model_package_group(self, **kwargs: Any) -> Mapping[str, Any]:
        """Describe one model package group."""

    def create_model_package(self, **kwargs: Any) -> Mapping[str, Any]:
        """Create one model package."""

    def update_model_package(self, **kwargs: Any) -> Mapping[str, Any]:
        """Update the approval status of one model package."""

    def describe_model_package(self, **kwargs: Any) -> Mapping[str, Any]:
        """Describe one model package."""


class MemoryModelRegistry:
    """In-memory registry used when a test stands in for the cloud service."""

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}
        self._next = 0

    def register(
        self,
        *,
        model_id: str,
        model_family: str,
        version: str,
        status: RegistryStatus,
    ) -> RegistryEntry:
        """Record a package and assign the next registry version."""

        _require_status(status)
        _require_text(model_id, "Model id")
        _require_text(model_family, "Model family")
        _require_text(version, "Model version")
        self._next += 1
        arn = (
            "arn:aws:sagemaker:us-east-1:000000000000:model-package/"
            f"{_group_name(model_family)}/{self._next}"
        )
        entry = RegistryEntry(arn=arn, version=str(self._next), status=status, actor_id=None)
        self._entries[arn] = entry
        return entry

    def approve(self, arn: str, *, actor_id: str) -> RegistryEntry:
        """Move a pending package to ``Approved`` and store ``actor_id``."""

        return self._decide(arn, actor_id=actor_id, status=RegistryStatus.APPROVED)

    def reject(self, arn: str, *, actor_id: str) -> RegistryEntry:
        """Move a pending package to ``Rejected`` and store ``actor_id``."""

        return self._decide(arn, actor_id=actor_id, status=RegistryStatus.REJECTED)

    def get(self, arn: str) -> RegistryEntry:
        """Return the package stored at ``arn``."""

        try:
            return self._entries[arn]
        except KeyError as exc:
            raise RegistryError(f"Model package {arn} is not registered.") from exc

    def _decide(self, arn: str, *, actor_id: str, status: RegistryStatus) -> RegistryEntry:
        actor = _require_text(actor_id, "Actor id")
        current = self.get(arn)
        if current.status is not RegistryStatus.PENDING_MANUAL_APPROVAL:
            raise RegistryError(
                f"Model package {arn} cannot move to {status.value} from {current.status.value}."
            )
        entry = RegistryEntry(
            arn=current.arn,
            version=current.version,
            status=status,
            actor_id=actor,
        )
        self._entries[arn] = entry
        return entry


class BotoModelRegistry:
    """Adapt the SageMaker model registry to :class:`ModelRegistry`."""

    def __init__(
        self,
        api: SageMakerRegistryApi,
        *,
        image: str | None = None,
        model_data_url: str | None = None,
    ) -> None:
        self._api = api
        self._image = image
        self._model_data_url = model_data_url

    def register(
        self,
        *,
        model_id: str,
        model_family: str,
        version: str,
        status: RegistryStatus,
    ) -> RegistryEntry:
        """Create a model package in the family group."""

        _require_status(status)
        _require_text(model_id, "Model id")
        _require_text(model_family, "Model family")
        group = _group_name(model_family)
        self._ensure_group(group)
        request: dict[str, Any] = {
            "ModelPackageGroupName": group,
            "ModelApprovalStatus": status.value,
            "ModelPackageDescription": model_id,
            "CustomerMetadataProperties": {
                "ModelId": model_id,
                "ModelFamily": model_family,
                "Version": version,
            },
        }
        inference = _inference_specification(self._image, self._model_data_url)
        if inference is not None:
            request["InferenceSpecification"] = inference
        response = self._api.create_model_package(**request)
        arn = _require_arn(response.get("ModelPackageArn"), "model package")
        return self.get(arn)

    def approve(self, arn: str, *, actor_id: str) -> RegistryEntry:
        """Set ``arn`` to ``Approved`` and store ``actor_id`` on the package."""

        return self._decide(arn, actor_id=actor_id, status=RegistryStatus.APPROVED)

    def reject(self, arn: str, *, actor_id: str) -> RegistryEntry:
        """Set ``arn`` to ``Rejected`` and store ``actor_id`` on the package."""

        return self._decide(arn, actor_id=actor_id, status=RegistryStatus.REJECTED)

    def get(self, arn: str) -> RegistryEntry:
        """Return the package described for ``arn``."""

        _require_arn(arn, "model package")
        described = self._api.describe_model_package(ModelPackageName=arn)
        return _entry(described)

    def _decide(self, arn: str, *, actor_id: str, status: RegistryStatus) -> RegistryEntry:
        actor = _require_text(actor_id, "Actor id")
        _require_arn(arn, "model package")
        self._api.update_model_package(
            ModelPackageArn=arn,
            ModelApprovalStatus=status.value,
            ApprovalDescription=actor,
        )
        return self.get(arn)

    def _ensure_group(self, name: str) -> None:
        try:
            self._api.describe_model_package_group(ModelPackageGroupName=name)
        except Exception as exc:
            if not _resource_missing(exc):
                raise
            self._api.create_model_package_group(
                ModelPackageGroupName=name,
                ModelPackageGroupDescription="ForecastOps model versions",
            )


def open_model_registry(region: str) -> ModelRegistry:
    """Build a live registry client for ``region``.

    The SDK import stays inside this function so local mode never loads it.
    """

    import boto3

    api = boto3.client("sagemaker", region_name=region)
    return BotoModelRegistry(api)


def _group_name(model_family: str) -> str:
    cleaned = model_family.strip().replace("_", "-")
    if cleaned == "":
        raise RegistryError("Model family must be a non-empty string.")
    return f"{_GROUP_PREFIX}-{cleaned}"


def _inference_specification(
    image: str | None,
    model_data_url: str | None,
) -> dict[str, Any] | None:
    if not image or not model_data_url:
        return None
    return {
        "Containers": [{"Image": image, "ModelDataUrl": model_data_url}],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
    }


def _entry(payload: Mapping[str, Any]) -> RegistryEntry:
    arn = _require_arn(payload.get("ModelPackageArn"), "model package")
    status_value = payload.get("ModelApprovalStatus")
    if not isinstance(status_value, str):
        raise RegistryError("The model registry did not return an approval status.")
    try:
        status = RegistryStatus(status_value)
    except ValueError as exc:
        raise RegistryError(f"Unexpected registry status {status_value}.") from exc
    description = payload.get("ApprovalDescription")
    actor_id = description if isinstance(description, str) and description.strip() else None
    return RegistryEntry(
        arn=arn,
        version=_version(payload.get("ModelPackageVersion")),
        status=status,
        actor_id=actor_id,
    )


def _version(value: object) -> str:
    if isinstance(value, int) and value > 0:
        return str(value)
    if isinstance(value, str) and value.strip() != "":
        return value
    raise RegistryError("The model registry did not return a version.")


def _require_arn(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("arn:"):
        raise RegistryError(f"The model registry did not return a {label} ARN.")
    return value


def _require_status(status: RegistryStatus) -> RegistryStatus:
    if not isinstance(status, RegistryStatus):
        raise RegistryError("Registry status must be a RegistryStatus value.")
    return status


def _require_text(value: str, label: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise RegistryError(f"{label} must be a non-empty string.")
    return value


def _resource_missing(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    if not isinstance(error, dict):
        return False
    return error.get("Code") == "ResourceNotFound"
