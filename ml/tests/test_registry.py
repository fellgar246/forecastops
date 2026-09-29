"""Registry status mapping, forecast selection, and the registry adapter."""

from collections.abc import Mapping
from typing import Any

import pytest

from forecastops_ml.promotion import ModelStatus, RegistryStatus
from forecastops_ml.registry import (
    BotoModelRegistry,
    ForecastSelectionError,
    MemoryModelRegistry,
    RegistryError,
    registry_status_for,
    select_forecast_model,
)


@pytest.mark.parametrize(
    ("internal", "registry"),
    [
        (ModelStatus.PENDING_APPROVAL, RegistryStatus.PENDING_MANUAL_APPROVAL),
        (ModelStatus.APPROVED, RegistryStatus.APPROVED),
        (ModelStatus.PRODUCTION, RegistryStatus.APPROVED),
        (ModelStatus.REJECTED, RegistryStatus.REJECTED),
    ],
)
def test_internal_status_maps_to_the_registry_status(
    internal: ModelStatus,
    registry: RegistryStatus,
) -> None:
    assert registry_status_for(internal) is registry


def test_registry_statuses_are_pending_approved_and_rejected() -> None:
    assert {status.value for status in RegistryStatus} == {
        "PendingManualApproval",
        "Approved",
        "Rejected",
    }


@pytest.mark.parametrize("status", [ModelStatus.TRAINED, ModelStatus.EVALUATED])
def test_training_and_evaluation_are_not_registry_statuses(status: ModelStatus) -> None:
    with pytest.raises(ValueError, match=status.value):
        registry_status_for(status)


@pytest.mark.parametrize(
    "registry_status",
    [RegistryStatus.PENDING_MANUAL_APPROVAL, RegistryStatus.REJECTED],
)
def test_cloud_forecast_rejects_pending_and_rejected_versions(
    registry_status: RegistryStatus,
) -> None:
    with pytest.raises(ForecastSelectionError, match="Approved"):
        select_forecast_model(
            internal_status=ModelStatus.APPROVED,
            registry_status=registry_status,
            cloud=True,
        )


@pytest.mark.parametrize("internal", [ModelStatus.APPROVED, ModelStatus.PRODUCTION])
def test_cloud_forecast_loads_an_approved_model(internal: ModelStatus) -> None:
    select_forecast_model(
        internal_status=internal,
        registry_status=RegistryStatus.APPROVED,
        cloud=True,
    )


def test_local_forecast_ignores_the_registry() -> None:
    select_forecast_model(
        internal_status=ModelStatus.APPROVED,
        registry_status=None,
        cloud=False,
    )
    select_forecast_model(
        internal_status=ModelStatus.PRODUCTION,
        registry_status=RegistryStatus.PENDING_MANUAL_APPROVAL,
        cloud=False,
    )


@pytest.mark.parametrize(
    "internal",
    [ModelStatus.PENDING_APPROVAL, ModelStatus.REJECTED, ModelStatus.TRAINED],
)
def test_forecast_rejects_an_unapproved_internal_status(internal: ModelStatus) -> None:
    with pytest.raises(ForecastSelectionError, match="APPROVED or PRODUCTION"):
        select_forecast_model(internal_status=internal, cloud=False)


def test_memory_registry_approve_and_reject_store_the_actor() -> None:
    registry = MemoryModelRegistry()
    pending = registry.register(
        model_id="model-1",
        model_family="gradient_boosting",
        version="1",
        status=RegistryStatus.PENDING_MANUAL_APPROVAL,
    )
    assert pending.status is RegistryStatus.PENDING_MANUAL_APPROVAL
    assert pending.actor_id is None
    assert pending.version == "1"

    approved = registry.approve(pending.arn, actor_id="analyst-7")
    assert approved.status is RegistryStatus.APPROVED
    assert approved.actor_id == "analyst-7"
    assert approved.arn == pending.arn
    assert registry.get(pending.arn).version == pending.version

    rejected = registry.register(
        model_id="model-2",
        model_family="deepar",
        version="1",
        status=RegistryStatus.PENDING_MANUAL_APPROVAL,
    )
    decision = registry.reject(rejected.arn, actor_id="analyst-8")
    assert decision.status is RegistryStatus.REJECTED
    assert decision.actor_id == "analyst-8"


def test_memory_registry_can_record_a_rejected_candidate() -> None:
    registry = MemoryModelRegistry()
    recorded = registry.register(
        model_id="model-1",
        model_family="gradient_boosting",
        version="1",
        status=RegistryStatus.REJECTED,
    )
    assert registry.get(recorded.arn).status is RegistryStatus.REJECTED
    with pytest.raises(RegistryError, match="cannot move"):
        registry.approve(recorded.arn, actor_id="analyst-7")


def test_boto_registry_registers_and_records_the_human_decision() -> None:
    api = _FakeRegistryApi()
    registry = BotoModelRegistry(api)
    pending = registry.register(
        model_id="model-1",
        model_family="gradient_boosting",
        version="3",
        status=RegistryStatus.PENDING_MANUAL_APPROVAL,
    )

    assert pending.status is RegistryStatus.PENDING_MANUAL_APPROVAL
    assert pending.version == "1"
    assert api.groups == {"forecastops-gradient-boosting"}
    created = api.created[-1]
    assert created["ModelApprovalStatus"] == "PendingManualApproval"
    assert created["CustomerMetadataProperties"] == {
        "ModelId": "model-1",
        "ModelFamily": "gradient_boosting",
        "Version": "3",
    }

    approved = registry.approve(pending.arn, actor_id="analyst-7")
    assert approved.status is RegistryStatus.APPROVED
    assert approved.actor_id == "analyst-7"
    assert api.updates[-1]["ModelApprovalStatus"] == "Approved"
    assert api.updates[-1]["ApprovalDescription"] == "analyst-7"

    other = registry.register(
        model_id="model-2",
        model_family="deepar",
        version="1",
        status=RegistryStatus.REJECTED,
    )
    assert other.status is RegistryStatus.REJECTED
    rejected = registry.reject(
        registry.register(
            model_id="model-3",
            model_family="deepar",
            version="1",
            status=RegistryStatus.PENDING_MANUAL_APPROVAL,
        ).arn,
        actor_id="analyst-9",
    )
    assert rejected.status is RegistryStatus.REJECTED
    assert rejected.actor_id == "analyst-9"


class _Missing(Exception):
    def __init__(self) -> None:
        super().__init__("missing")
        self.response = {"Error": {"Code": "ResourceNotFound"}}


class _FakeRegistryApi:
    def __init__(self) -> None:
        self.groups: set[str] = set()
        self.packages: dict[str, dict[str, object]] = {}
        self.created: list[Mapping[str, Any]] = []
        self.updates: list[Mapping[str, Any]] = []
        self._next = 0

    def describe_model_package_group(self, **kwargs: Any) -> Mapping[str, Any]:
        name = str(kwargs["ModelPackageGroupName"])
        if name not in self.groups:
            raise _Missing()
        return {"ModelPackageGroupName": name}

    def create_model_package_group(self, **kwargs: Any) -> Mapping[str, Any]:
        name = str(kwargs["ModelPackageGroupName"])
        self.groups.add(name)
        return {"ModelPackageGroupArn": f"arn:aws:sagemaker:us-east-1:0:model-package-group/{name}"}

    def create_model_package(self, **kwargs: Any) -> Mapping[str, Any]:
        self.created.append(dict(kwargs))
        self._next += 1
        group = str(kwargs["ModelPackageGroupName"])
        arn = f"arn:aws:sagemaker:us-east-1:0:model-package/{group}/{self._next}"
        self.packages[arn] = {
            "ModelPackageArn": arn,
            "ModelPackageVersion": self._next,
            "ModelApprovalStatus": kwargs["ModelApprovalStatus"],
        }
        return {"ModelPackageArn": arn}

    def update_model_package(self, **kwargs: Any) -> Mapping[str, Any]:
        self.updates.append(dict(kwargs))
        package = self.packages[str(kwargs["ModelPackageArn"])]
        package["ModelApprovalStatus"] = kwargs["ModelApprovalStatus"]
        package["ApprovalDescription"] = kwargs["ApprovalDescription"]
        return {"ModelPackageArn": kwargs["ModelPackageArn"]}

    def describe_model_package(self, **kwargs: Any) -> Mapping[str, Any]:
        return self.packages[str(kwargs["ModelPackageName"])]
