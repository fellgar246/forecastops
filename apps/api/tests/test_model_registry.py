"""Model registry selection for cloud forecasts."""

import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from envfile import ROOT, parse_env_file
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from forecastops_api.artifacts import LocalArtifactStore
from forecastops_api.db import Base
from forecastops_api.main import create_app
from forecastops_api.persistence import DatasetRow, ModelVersionRow, TrainingRunRow
from forecastops_api.registry import select_model_registry
from forecastops_api.settings import ExecutionMode, Settings, get_settings
from forecastops_ml.data.synthetic import DEFAULT_MAX_ROWS, DEFAULT_SEED, generate_dataset
from forecastops_ml.promotion import RegistryStatus
from forecastops_ml.registry import BotoModelRegistry, MemoryModelRegistry


def test_selector_stays_closed_unless_cloud_execution_and_sagemaker_are_on(
    monkeypatch,
) -> None:
    constructed: list[tuple[str, str | None]] = []
    fake = types.ModuleType("boto3")

    def client(service_name: str, region_name: str | None = None) -> object:
        constructed.append((service_name, region_name))
        return object()

    fake.client = client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake)

    assert select_model_registry(_settings(monkeypatch)) is None
    assert constructed == []

    selected = select_model_registry(
        _settings(monkeypatch, execution_mode="aws", sagemaker_enabled=True)
    )
    assert isinstance(selected, BotoModelRegistry)
    assert constructed == [("sagemaker", "us-east-1")]


def test_cloud_forecast_refuses_pending_and_rejected_registry_status(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.app.state.settings.execution_mode = ExecutionMode.AWS

    for registry_status in ("PendingManualApproval", "Rejected"):
        model_id = _insert_model(
            client,
            status="APPROVED",
            registry_status=registry_status,
        )
        response = client.post(
            "/forecasts",
            json={"model_id": model_id, "horizon": 7, "granularity": "day"},
        )
        assert response.status_code == 409
        assert response.json()["code"] == "model_not_approved"
        assert "Approved" in response.json()["message"]


def test_registry_approval_updates_both_representations(tmp_path: Path) -> None:
    client = _client(tmp_path)
    registry = MemoryModelRegistry()
    client.app.state.model_registry = registry
    dataset_id = _valid_dataset(client, tmp_path / "data")
    training = client.post(
        "/training-runs",
        json={"dataset_id": dataset_id, "model_family": "seasonal_naive"},
    )
    assert training.status_code == 202

    model = client.get("/models").json()["items"][0]
    assert model["status"] == "PENDING_APPROVAL"
    assert model["registry_status"] == "PendingManualApproval"
    assert registry.get(model["registry_arn"]).status is RegistryStatus.PENDING_MANUAL_APPROVAL

    blocked = client.post(
        "/forecasts",
        json={"model_id": model["id"], "horizon": 7, "granularity": "day"},
    )
    assert blocked.status_code == 409
    assert blocked.json()["code"] == "model_not_approved"

    approved = client.post(
        f"/models/{model['id']}/approve",
        json={"actor_id": "analyst-1"},
    )
    assert approved.status_code == 200
    body = approved.json()
    assert body["status"] == "APPROVED"
    assert body["registry_status"] == "Approved"
    assert registry.get(body["registry_arn"]).actor_id == "analyst-1"

    forecast = client.post(
        "/forecasts",
        json={"model_id": model["id"], "horizon": 7, "granularity": "day"},
    )
    assert forecast.status_code == 202


def test_registry_rejection_stores_the_actor(tmp_path: Path) -> None:
    client = _client(tmp_path)
    registry = MemoryModelRegistry()
    client.app.state.model_registry = registry
    pending = registry.register(
        model_id="model-1",
        model_family="seasonal_naive",
        version="1",
        status=RegistryStatus.PENDING_MANUAL_APPROVAL,
    )
    model_id = _insert_model(
        client,
        status="PENDING_APPROVAL",
        registry_status="PendingManualApproval",
        registry_arn=pending.arn,
    )

    rejected = client.post(
        f"/models/{model_id}/reject",
        json={"actor_id": "analyst-1", "reason": "human"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "REJECTED"
    assert rejected.json()["registry_status"] == "Rejected"
    assert registry.get(pending.arn).actor_id == "analyst-1"


def test_local_approval_leaves_the_registry_empty(tmp_path: Path) -> None:
    client = _client(tmp_path)
    dataset_id = _valid_dataset(client, tmp_path / "data")
    training = client.post(
        "/training-runs",
        json={"dataset_id": dataset_id, "model_family": "seasonal_naive"},
    )
    assert training.status_code == 202
    model = client.get("/models").json()["items"][0]
    assert model["registry_arn"] == ""
    assert model["registry_status"] == ""

    approved = client.post(
        f"/models/{model['id']}/approve",
        json={"actor_id": "analyst-1"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"
    assert approved.json()["registry_arn"] == ""
    assert approved.json()["registry_status"] == ""


def _settings(
    monkeypatch,
    *,
    execution_mode: str = "local",
    sagemaker_enabled: bool = False,
) -> Settings:
    for key, value in parse_env_file(ROOT / ".env.example").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("EXECUTION_MODE", execution_mode)
    monkeypatch.setenv("SAGEMAKER_ENABLED", "true" if sagemaker_enabled else "false")
    return Settings(_env_file=None)


def _client(tmp_path: Path) -> TestClient:
    database = tmp_path / "api.db"
    get_settings.cache_clear()
    application = create_app()
    engine = create_engine(f"sqlite+pysqlite:///{database}")
    application.state.engine = engine
    application.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    application.state.artifact_dir = tmp_path / "artifacts"
    application.state.artifact_store = LocalArtifactStore(application.state.artifact_dir)
    Base.metadata.create_all(engine)
    return TestClient(application)


def _insert_model(
    client: TestClient,
    *,
    status: str,
    registry_status: str,
    registry_arn: str = "",
) -> str:
    factory = client.app.state.session_factory
    session = factory()
    now = datetime.now(UTC)
    dataset_id = str(uuid4())
    run_id = str(uuid4())
    model_id = str(uuid4())
    session.add(
        DatasetRow(
            id=dataset_id,
            name="retail-test",
            version="1",
            source="synthetic",
            status="valid",
            uri=str(client.app.state.artifact_dir),
            schema_version="retail-demand-v1",
            row_count=1,
            date_min=None,
            date_max=None,
            quality_report=None,
            created_at=now,
        )
    )
    session.add(
        TrainingRunRow(
            id=run_id,
            dataset_id=dataset_id,
            dataset_version="1",
            model_family="seasonal_naive",
            configuration={},
            status="COMPLETED",
            started_at=now,
            finished_at=now,
            artifact_uri="",
            metrics=None,
            git_sha="",
            pipeline_execution_arn="",
            error_message=None,
            created_at=now,
        )
    )
    session.add(
        ModelVersionRow(
            id=model_id,
            model_family="seasonal_naive",
            version="1",
            training_run_id=run_id,
            dataset_id=dataset_id,
            dataset_version="1",
            registry_arn=registry_arn,
            registry_status=registry_status,
            status=status,
            metrics=None,
            approved_at=None,
            approved_by=None,
            rejected_by=None,
            rejection_reason=None,
            created_at=now,
        )
    )
    session.commit()
    session.close()
    return model_id


def _valid_dataset(client: TestClient, directory: Path) -> str:
    generate_dataset(directory, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    created = client.post(
        "/datasets",
        json={"name": "retail-test", "source": "synthetic", "uri": str(directory)},
    )
    assert created.status_code == 202
    dataset_id = created.json()["job_id"]
    validated = client.post(f"/datasets/{dataset_id}/validate", json={"as_of": "2026-09-30"})
    assert validated.status_code == 200
    return str(dataset_id)
