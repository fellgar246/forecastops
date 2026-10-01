"""Queued forecasts, the daily cap, idempotency, and batch failure."""

import json
import sys
import types
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from forecastops_api.artifacts import LocalArtifactStore
from forecastops_api.db import Base
from forecastops_api.forecast_jobs import select_batch_inference
from forecastops_api.main import create_app
from forecastops_api.persistence import ModelVersionRow, TrainingRunRow
from forecastops_api.repositories import Repository
from forecastops_api.services import ForecastService
from forecastops_api.settings import ExecutionMode, Settings, get_settings
from forecastops_ml.data.synthetic import DEFAULT_MAX_ROWS, DEFAULT_SEED, generate_dataset
from forecastops_ml.inference import (
    BatchInference,
    MemoryForecastObjects,
    StoredForecastPoint,
    forecast_document,
)


def test_selector_stays_closed_unless_cloud_execution_and_sagemaker_are_on(
    monkeypatch,
) -> None:
    constructed: list[str] = []
    fake = types.ModuleType("boto3")

    def client(service_name: str, region_name: str | None = None) -> object:
        constructed.append(service_name)
        _ = region_name
        return object()

    fake.client = client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake)

    assert select_batch_inference(_settings(monkeypatch)) is None
    assert constructed == []

    closed = select_batch_inference(
        _settings(
            monkeypatch,
            execution_mode="aws",
            sagemaker_enabled=True,
            aws_ml_enabled=False,
        )
    )
    assert closed is None
    assert constructed == []

    refusing = select_batch_inference(
        _settings(
            monkeypatch,
            execution_mode="aws",
            sagemaker_enabled=True,
            online_inference=True,
            aws_ml_enabled=True,
        )
    )
    assert isinstance(refusing, BatchInference)
    assert constructed == []


def test_queued_quantile_forecast_becomes_a_readable_series(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.app.state.execute_forecasts_inline = False
    dataset_id = _valid_dataset(client, tmp_path / "data")
    model_id = _insert_model(client, dataset_id, family="deepar")

    created = client.post(
        "/forecasts",
        json={"model_id": model_id, "horizon": 4, "granularity": "day"},
    )
    assert created.status_code == 202
    forecast_id = created.json()["job_id"]
    queued = client.get(f"/forecasts/{forecast_id}")
    assert queued.status_code == 200
    assert queued.json()["status"] == "QUEUED"

    _execute(client, forecast_id)

    stored = client.get(f"/forecasts/{forecast_id}").json()
    assert stored["status"] == "SUCCEEDED"
    assert stored["output_uri"].endswith(f"forecasts/{forecast_id}/forecast.json")
    artifact = json.loads(Path(stored["output_uri"]).read_text(encoding="utf-8"))
    series = client.get(f"/forecasts/{forecast_id}/series")
    assert series.status_code == 200
    points = series.json()["items"]
    assert points
    assert len(points) == len(artifact["points"])
    assert all(point["p10"] is not None for point in points)
    assert all(point["p50"] is not None for point in points)
    assert all(point["p90"] is not None for point in points)
    assert all(point["p10"] <= point["p50"] <= point["p90"] for point in points)


def test_failed_forecast_hides_dataset_rows(tmp_path: Path) -> None:
    client = _client(tmp_path)
    dataset_id = _valid_dataset(client, tmp_path / "data")
    model_id = _insert_model(client, dataset_id, family="deepar")
    leaked = LeakingPredictor()
    client.app.state.forecast_predictor = leaked

    created = client.post(
        "/forecasts",
        json={"model_id": model_id, "horizon": 4, "granularity": "day"},
    )
    assert created.status_code == 202
    forecast_id = created.json()["job_id"]
    stored = client.get(f"/forecasts/{forecast_id}").json()

    assert stored["status"] == "FAILED"
    assert stored["error_message"] == "Batch inference failed."
    assert leaked.secret not in stored["error_message"]
    assert "424242" not in stored["error_message"]
    assert "sku-secret-999" not in stored["error_message"]


def test_daily_cap_returns_429_and_idempotency_replays_the_original_run(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.app.state.execute_forecasts_inline = False
    client.app.state.settings = client.app.state.settings.model_copy(
        update={"max_batch_inference_jobs_per_day": 1}
    )
    dataset_id = _valid_dataset(client, tmp_path / "data")
    model_id = _insert_model(client, dataset_id, family="deepar")
    body = {
        "model_id": model_id,
        "horizon": 4,
        "granularity": "day",
        "idempotency_key": "forecast-once",
    }

    first = client.post("/forecasts", json=body)
    assert first.status_code == 202
    forecast_id = first.json()["job_id"]

    blocked = client.post(
        "/forecasts",
        json={
            "model_id": model_id,
            "horizon": 4,
            "granularity": "day",
            "idempotency_key": "forecast-again",
        },
    )
    assert blocked.status_code == 429
    assert blocked.json()["code"] == "batch_inference_limit"

    replay = client.post(
        "/forecasts",
        json={**body, "horizon": 2},
    )
    assert replay.status_code == 202
    assert replay.json()["job_id"] == forecast_id
    stored = client.get(f"/forecasts/{forecast_id}").json()
    assert stored["status"] == "QUEUED"
    assert stored["horizon"] == 4
    assert len(client.get("/forecasts").json()["items"]) == 1


def test_cloud_deepar_forecast_reads_the_batch_artifact(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.app.state.settings = client.app.state.settings.model_copy(
        update={
            "execution_mode": ExecutionMode.AWS,
            "sagemaker_enabled": True,
            "online_inference": False,
            "aws_ml_enabled": True,
        }
    )
    dataset_id = _valid_dataset(client, tmp_path / "data")
    model_id = _insert_model(
        client,
        dataset_id,
        family="deepar",
        registry_status="Approved",
    )
    output = forecast_document(
        (
            StoredForecastPoint(
                series_id="store-1|sku-1",
                date=datetime(2026, 2, 1, tzinfo=UTC).date(),
                p10=8.0,
                p50=10.0,
                p90=12.0,
            ),
        )
    )
    objects = MemoryForecastObjects()
    batch_client = CompletingClient(objects, output)
    client.app.state.batch_inference = BatchInference(
        batch_client,
        objects,
        bucket="forecastops-artifacts",
        sagemaker_enabled=True,
        online_inference=False,
    )

    created = client.post(
        "/forecasts",
        json={"model_id": model_id, "horizon": 7, "granularity": "day"},
    )
    assert created.status_code == 202
    forecast_id = created.json()["job_id"]
    stored = client.get(f"/forecasts/{forecast_id}").json()
    assert stored["status"] == "SUCCEEDED"
    series = client.get(f"/forecasts/{forecast_id}/series").json()["items"]
    assert series == [
        {
            "series_id": "store-1|sku-1",
            "date": "2026-02-01",
            "p10": 8.0,
            "p50": 10.0,
            "p90": 12.0,
            "actual": None,
        }
    ]
    request = batch_client.requests[0]
    assert "EndpointName" not in request
    assert "/forecasts/" in request["TransformOutput"]["S3OutputPath"]  # type: ignore[index]
    assert batch_client.endpoint_calls == 0


class LeakingPredictor:
    """Predictor that raises with an observation row in the exception."""

    def __init__(self) -> None:
        self.secret = ""

    def predict(
        self,
        frame: pa.Table,
        *,
        horizon: int,
        granularity: str,
    ) -> tuple[StoredForecastPoint, ...]:
        _ = horizon, granularity
        row = frame.to_pylist()[0]
        self.secret = json.dumps(row, default=str)
        raise RuntimeError(f"{self.secret} sku-secret-999 units_sold 424242")


class CompletingClient:
    """Fake transform client that writes a finished forecast artifact."""

    def __init__(self, objects: MemoryForecastObjects, output: bytes) -> None:
        self.objects = objects
        self.output = output
        self.requests: list[Mapping[str, object]] = []
        self.endpoint_calls = 0

    def create_transform_job(self, request: Mapping[str, object]) -> str:
        self.requests.append(request)
        output = request["TransformOutput"]
        assert isinstance(output, Mapping)
        prefix = str(output["S3OutputPath"])
        self.objects.put_bytes(prefix.rstrip("/") + "/forecast.json", self.output)
        return "arn:aws:sagemaker:us-east-1:000000000000:transform-job/example"

    def describe_transform_job(self, job_name: str) -> Mapping[str, object]:
        return {"TransformJobStatus": "Completed", "TransformJobName": job_name}

    def create_endpoint(self, **kwargs: object) -> str:
        self.endpoint_calls += 1
        raise AssertionError(kwargs)


def _execute(client: TestClient, forecast_id: str) -> None:
    factory = client.app.state.session_factory
    session: Session = factory()
    try:
        service = ForecastService(
            Repository(session),
            client.app.state.settings,
            client.app.state.artifact_store,
            getattr(client.app.state, "model_registry", None),
            batch=client.app.state.batch_inference,
            predictor=client.app.state.forecast_predictor,
        )
        service.execute_forecast(forecast_id)
        session.commit()
    finally:
        session.close()


def _client(tmp_path: Path) -> TestClient:
    database = tmp_path / "api.db"
    get_settings.cache_clear()
    application = create_app()
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite+pysqlite:///{database}")
    application.state.engine = engine
    application.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    application.state.artifact_dir = tmp_path / "artifacts"
    application.state.artifact_store = LocalArtifactStore(application.state.artifact_dir)
    Base.metadata.create_all(engine)
    return TestClient(application)


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
    assert validated.json()["status"] == "valid"
    return str(dataset_id)


def _insert_model(
    client: TestClient,
    dataset_id: str,
    *,
    family: str,
    registry_status: str = "",
) -> str:
    dataset = client.get(f"/datasets/{dataset_id}").json()
    factory = client.app.state.session_factory
    session = factory()
    now = datetime.now(UTC)
    run_id = str(uuid4())
    model_id = str(uuid4())
    session.add(
        TrainingRunRow(
            id=run_id,
            dataset_id=dataset_id,
            dataset_version=dataset["version"],
            model_family=family,
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
            model_family=family,
            version="1",
            training_run_id=run_id,
            dataset_id=dataset_id,
            dataset_version=dataset["version"],
            registry_arn="",
            registry_status=registry_status,
            status="APPROVED",
            metrics=None,
            approved_at=now,
            approved_by="analyst-1",
            rejected_by=None,
            rejection_reason=None,
            created_at=now,
        )
    )
    session.commit()
    session.close()
    return model_id


def _settings(
    monkeypatch,
    *,
    execution_mode: str = "local",
    sagemaker_enabled: bool = False,
    online_inference: bool = False,
    aws_ml_enabled: bool = False,
) -> Settings:
    from envfile import ROOT, parse_env_file

    for key, value in parse_env_file(ROOT / ".env.example").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("EXECUTION_MODE", execution_mode)
    if execution_mode == "aws":
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("COGNITO_USER_POOL_ID", "us-east-1_testpool")
        monkeypatch.setenv("COGNITO_APP_CLIENT_ID", "test-client")
    monkeypatch.setenv("SAGEMAKER_ENABLED", "true" if sagemaker_enabled else "false")
    monkeypatch.setenv("ONLINE_INFERENCE", "true" if online_inference else "false")
    monkeypatch.setenv("AWS_ML_ENABLED", "true" if aws_ml_enabled else "false")
    monkeypatch.setenv("ARTIFACTS_BUCKET", "forecastops-artifacts")
    return Settings(_env_file=None)
