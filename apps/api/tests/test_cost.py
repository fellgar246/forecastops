"""Cost ceilings, kill switches, and the cost snapshot."""

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from forecastops_api.artifacts import LocalArtifactStore
from forecastops_api.cost import read_last_cleanup
from forecastops_api.db import Base
from forecastops_api.main import create_app
from forecastops_api.persistence import (
    AIExplanationRow,
    DatasetRow,
    ForecastPointRow,
    ForecastRunRow,
    ModelVersionRow,
    TrainingRunRow,
)
from forecastops_api.settings import get_settings
from forecastops_ml.data.synthetic import DEFAULT_MAX_ROWS, DEFAULT_SEED, generate_dataset
from forecastops_ml.explanations import ExplanationDraft, ExplanationUsage
from forecastops_ml.explanations.clients import BedrockExplanationClient
from forecastops_ml.inference import BatchInference, MemoryForecastObjects


def test_each_limit_rejects_the_request_that_crosses_it(tmp_path: Path) -> None:
    client = _client(tmp_path)

    client.app.state.settings = client.app.state.settings.model_copy(
        update={"max_training_jobs_per_day": 0}
    )
    training = client.post(
        "/training-runs",
        json={"dataset_id": "dataset-1", "model_family": "seasonal_naive", "configuration": {}},
    )
    assert training.status_code == 429
    assert training.json()["code"] == "training_limit"
    assert client.get("/training-runs").json()["items"] == []

    client.app.state.settings = client.app.state.settings.model_copy(
        update={"max_training_jobs_per_day": 2, "allow_gpu_training": True}
    )
    gpu = client.post(
        "/training-runs",
        json={"dataset_id": "dataset-1", "model_family": "seasonal_naive", "configuration": {}},
    )
    assert gpu.status_code == 409
    assert gpu.json()["code"] == "gpu_training_disabled"
    client.app.state.settings = client.app.state.settings.model_copy(
        update={"allow_gpu_training": False}
    )
    requested = client.post(
        "/training-runs",
        json={
            "dataset_id": "dataset-1",
            "model_family": "seasonal_naive",
            "configuration": {"instance_type": "ml.g4dn.xlarge"},
        },
    )
    assert requested.status_code == 409
    assert requested.json()["code"] == "gpu_training_disabled"
    assert client.get("/training-runs").json()["items"] == []

    client.app.state.settings = client.app.state.settings.model_copy(
        update={"max_dataset_rows_demo": 1}
    )
    directory = tmp_path / "rows"
    generate_dataset(directory, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    dataset = client.post(
        "/datasets",
        json={"name": "too-large", "source": "synthetic", "uri": str(directory)},
    )
    assert dataset.status_code == 422
    assert dataset.json()["code"] == "dataset_row_limit"
    assert client.get("/datasets").json()["items"] == []
    client.app.state.settings = client.app.state.settings.model_copy(
        update={"max_dataset_rows_demo": DEFAULT_MAX_ROWS}
    )

    client.app.state.settings = client.app.state.settings.model_copy(
        update={"max_forecast_horizon_days": 1}
    )
    horizon = client.post(
        "/forecasts",
        json={"model_id": "missing", "horizon": 2, "granularity": "day"},
    )
    assert horizon.status_code == 422
    assert horizon.json()["code"] == "horizon_too_long"
    assert client.get("/forecasts").json()["items"] == []

    dataset_id = _valid_dataset(client, tmp_path / "batch")
    model_id = _insert_model(client, dataset_id)
    client.app.state.settings = client.app.state.settings.model_copy(
        update={"max_batch_inference_jobs_per_day": 0, "max_forecast_horizon_days": 90}
    )
    batch = client.post(
        "/forecasts",
        json={"model_id": model_id, "horizon": 4, "granularity": "day"},
    )
    assert batch.status_code == 429
    assert batch.json()["code"] == "batch_inference_limit"
    assert client.get("/forecasts").json()["items"] == []

    forecast_id = _seed_forecast(client)
    client.app.state.settings = client.app.state.settings.model_copy(
        update={"max_bedrock_calls_per_day": 1, "max_batch_inference_jobs_per_day": 5}
    )
    _seed_explanation_call(client, forecast_id)
    calls = client.post(f"/forecasts/{forecast_id}/explanation")
    assert calls.status_code == 429
    assert calls.json()["code"] == "explanation_quota"
    assert "1 call" in calls.json()["message"]

    client.app.state.settings = client.app.state.settings.model_copy(
        update={"max_bedrock_calls_per_day": 30, "max_bedrock_input_tokens": 1}
    )
    entered: list[str] = []

    class Boom:
        model_id = "unused"

        def explain(self, package: object) -> ExplanationDraft:
            entered.append("called")
            raise AssertionError(package)

        @property
        def usage(self) -> ExplanationUsage:
            return ExplanationUsage(input_tokens=0, output_tokens=0, latency_ms=0)

    client.app.state.explanation_client = Boom()
    blocked_input = client.post(f"/forecasts/{forecast_id}/explanation")
    assert blocked_input.status_code == 429
    assert "input token ceiling" in blocked_input.json()["message"]
    assert entered == []

    client.app.state.settings = client.app.state.settings.model_copy(
        update={"max_bedrock_input_tokens": 5000, "max_bedrock_output_tokens": 1}
    )
    client.app.state.explanation_client = _WideOutput()
    blocked_output = client.post(f"/forecasts/{forecast_id}/explanation")
    assert blocked_output.status_code == 429
    assert blocked_output.json()["code"] == "explanation_quota"
    assert "output token ceiling" in blocked_output.json()["message"]
    assert client.get(f"/forecasts/{forecast_id}/explanation").status_code == 404


def test_kill_switches_block_jobs_and_local_forecasts_still_run(tmp_path: Path) -> None:
    client = _client(tmp_path)
    dataset_id = _valid_dataset(client, tmp_path / "local")
    model_id = _insert_model(client, dataset_id)
    before = len(client.get("/training-runs").json()["items"])
    client.app.state.settings = client.app.state.settings.model_copy(
        update={
            "training_enabled": False,
            "bedrock_enabled": False,
            "aws_ml_enabled": False,
        }
    )

    training = client.post(
        "/training-runs",
        json={"dataset_id": dataset_id, "model_family": "seasonal_naive", "configuration": {}},
    )
    assert training.status_code == 409
    assert training.json()["code"] == "training_disabled"
    assert len(client.get("/training-runs").json()["items"]) == before

    forecast = client.post(
        "/forecasts",
        json={"model_id": model_id, "horizon": 4, "granularity": "day"},
    )
    assert forecast.status_code == 202
    forecast_id = forecast.json()["job_id"]
    stored = client.get(f"/forecasts/{forecast_id}")
    assert stored.status_code == 200
    assert stored.json()["status"] == "SUCCEEDED"
    series = client.get(f"/forecasts/{forecast_id}/series")
    assert series.status_code == 200
    assert series.json()["items"]
    assert series.json()["p50_total"] is not None

    client.app.state.settings = client.app.state.settings.model_copy(
        update={"training_enabled": True, "aws_ml_enabled": False}
    )
    cloud = client.post(
        "/training-runs",
        json={"dataset_id": dataset_id, "model_family": "deepar", "configuration": {}},
    )
    assert cloud.status_code == 409
    assert cloud.json()["code"] == "aws_ml_disabled"
    assert len(client.get("/training-runs").json()["items"]) == before

    transform = _SilentTransform()
    client.app.state.batch_inference = BatchInference(
        transform,
        MemoryForecastObjects(),
        bucket="forecastops-artifacts",
        sagemaker_enabled=True,
        online_inference=False,
    )
    deepar_id = _insert_model(client, dataset_id, family="deepar")
    blocked = client.post(
        "/forecasts",
        json={"model_id": deepar_id, "horizon": 4, "granularity": "day"},
    )
    assert blocked.status_code == 409
    assert blocked.json()["code"] == "aws_ml_disabled"
    assert transform.requests == []


def test_bedrock_switch_refuses_the_provider_and_keeps_the_local_mock(tmp_path: Path) -> None:
    client = _client(tmp_path)
    forecast_id = _seed_forecast(client)
    runtime = _SpyRuntime()
    client.app.state.explanation_client = BedrockExplanationClient(
        "anthropic.claude-3-haiku-20240307-v1:0",
        max_output_tokens=700,
        region="us-east-1",
        runtime=runtime,
    )
    refused = client.post(f"/forecasts/{forecast_id}/explanation")
    assert refused.status_code == 409
    assert refused.json()["code"] == "bedrock_disabled"
    assert runtime.calls == 0

    client.app.state.explanation_client = None
    explained = client.post(f"/forecasts/{forecast_id}/explanation")
    assert explained.status_code == 202
    assert explained.json()["model_id"] == "mock-explainer"
    series = client.get(f"/forecasts/{forecast_id}/series")
    assert series.status_code == 200
    assert series.json()["items"][0]["p50"] == 1840


def test_cost_snapshot_uses_records_flags_and_the_cleanup_stamp(tmp_path: Path) -> None:
    client = _client(tmp_path)
    now = datetime.now(UTC)
    _seed_usage(client, now)
    stamp = tmp_path / "last_cleanup_at"
    stamp.write_text("2026-09-30T16:05:00Z\n", encoding="utf-8")
    client.app.state.settings = client.app.state.settings.model_copy(
        update={
            "cost_state_file": stamp,
            "aws_enabled": True,
            "aws_ml_enabled": True,
            "bedrock_enabled": True,
            "sagemaker_enabled": True,
            "training_enabled": False,
            "online_inference": False,
        }
    )

    response = client.get("/cost")
    assert response.status_code == 200
    body = response.json()
    assert body["monthly_budget_usd"] == 5
    assert body["training_runs_this_month"] == 1
    assert body["explanation_calls_today"] == 1
    assert body["approximate_explanation_tokens"] == 20
    assert body["estimated_spend_usd"] is None
    assert body["spend_note"] == "Billing is not configured."
    assert body["aws_enabled"] is True
    assert body["aws_ml_enabled"] is True
    assert body["bedrock_enabled"] is True
    assert body["sagemaker_enabled"] is True
    assert body["training_enabled"] is False
    assert body["online_inference"] is False
    assert body["last_cleanup_at"].startswith("2026-09-30T16:05:00")


def test_missing_or_invalid_cleanup_stamp_stays_empty(tmp_path: Path) -> None:
    missing = tmp_path / "absent"
    assert read_last_cleanup(missing) is None
    blank = tmp_path / "blank"
    blank.write_text("not-a-time\n", encoding="utf-8")
    assert read_last_cleanup(blank) is None
    client = _client(tmp_path)
    response = client.get("/cost")
    assert response.status_code == 200
    assert response.json()["last_cleanup_at"] is None
    assert response.json()["estimated_spend_usd"] is None


class _WideOutput:
    model_id = "wide-explainer"

    def __init__(self) -> None:
        self._usage = ExplanationUsage(input_tokens=4, output_tokens=40, latency_ms=1)

    def explain(self, package: object) -> ExplanationDraft:
        _ = package
        return ExplanationDraft(
            summary="The median forecast is 1840.",
            drivers=["historical_context"],
            risks=["The range may be wide."],
            uncertainty_note="The forecast remains uncertain.",
            recommended_checks=["Compare the median forecast with history."],
        )

    @property
    def usage(self) -> ExplanationUsage:
        return self._usage


class _SpyRuntime:
    def __init__(self) -> None:
        self.calls = 0

    def invoke_model(self, **kwargs: object) -> Mapping[str, object]:
        self.calls += 1
        raise AssertionError(kwargs)


class _SilentTransform:
    def __init__(self) -> None:
        self.requests: list[Mapping[str, object]] = []

    def create_transform_job(self, request: Mapping[str, object]) -> str:
        self.requests.append(request)
        raise AssertionError("batch job started")

    def describe_transform_job(self, job_name: str) -> Mapping[str, object]:
        return {"TransformJobStatus": "Completed", "TransformJobName": job_name}


def _client(tmp_path: Path) -> TestClient:
    database = tmp_path / "api.db"
    get_settings.cache_clear()
    application = create_app()
    engine = create_engine(f"sqlite+pysqlite:///{database}")
    application.state.engine = engine
    application.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    application.state.artifact_dir = tmp_path / "artifacts"
    application.state.artifact_store = LocalArtifactStore(application.state.artifact_dir)
    application.state.settings = application.state.settings.model_copy(
        update={"cost_state_file": tmp_path / "last_cleanup_at"}
    )
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
    return str(dataset_id)


def _insert_model(client: TestClient, dataset_id: str, *, family: str = "seasonal_naive") -> str:
    dataset = client.get(f"/datasets/{dataset_id}").json()
    session = client.app.state.session_factory()
    now = datetime.now(UTC)
    run_id = f"run-{family}"
    model_id = f"model-{family}"
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
            registry_status="",
            status="APPROVED",
            metrics={"wape": 0.1, "bias": 0.0},
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


def _seed_forecast(client: TestClient) -> str:
    now = datetime.now(UTC)
    session = client.app.state.session_factory()
    session.add(
        DatasetRow(
            id="dataset-1",
            name="retail-test",
            version="1",
            source="synthetic",
            status="valid",
            uri="missing",
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
            id="run-1",
            dataset_id="dataset-1",
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
            id="model-1",
            model_family="seasonal_naive",
            version="1",
            training_run_id="run-1",
            dataset_id="dataset-1",
            dataset_version="1",
            registry_arn="",
            registry_status="",
            status="APPROVED",
            metrics={"wape": 0.132, "bias": 0.004},
            approved_at=now,
            approved_by="analyst-1",
            rejected_by=None,
            rejection_reason=None,
            created_at=now,
        )
    )
    session.add(
        ForecastRunRow(
            id="forecast-1",
            model_version_id="model-1",
            dataset_id="dataset-1",
            dataset_version="1",
            horizon=56,
            granularity="day",
            status="SUCCEEDED",
            output_uri="",
            error_message=None,
            created_at=now,
        )
    )
    session.add(
        ForecastPointRow(
            forecast_run_id="forecast-1",
            series_id="CDMX-01|sku-1",
            date=datetime(2026, 9, 21, tzinfo=UTC).date(),
            p10=1530,
            p50=1840,
            p90=2130,
            actual=None,
        )
    )
    session.commit()
    session.close()
    return "forecast-1"


def _seed_explanation_call(client: TestClient, forecast_id: str) -> None:
    session = client.app.state.session_factory()
    session.add(
        AIExplanationRow(
            id="prior-call",
            forecast_run_id=forecast_id,
            scope_key='{"series":"other"}',
            scope={"series": "other"},
            model_id="mock-explainer",
            prompt_version="v1",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            explanation=None,
            package=None,
            status="invalid",
            created_at=datetime.now(UTC),
        )
    )
    session.commit()
    session.close()


def _seed_usage(client: TestClient, now: datetime) -> None:
    session = client.app.state.session_factory()
    session.add(
        DatasetRow(
            id="dataset-usage",
            name="usage",
            version="1",
            source="synthetic",
            status="valid",
            uri="missing",
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
            id="run-old",
            dataset_id="dataset-usage",
            dataset_version="1",
            model_family="naive",
            configuration={},
            status="COMPLETED",
            started_at=None,
            finished_at=None,
            artifact_uri="",
            metrics=None,
            git_sha="",
            pipeline_execution_arn="",
            error_message=None,
            created_at=datetime(2020, 1, 15, tzinfo=UTC),
        )
    )
    session.add(
        TrainingRunRow(
            id="run-now",
            dataset_id="dataset-usage",
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
            id="model-usage",
            model_family="seasonal_naive",
            version="1",
            training_run_id="run-now",
            dataset_id="dataset-usage",
            dataset_version="1",
            registry_arn="",
            registry_status="",
            status="APPROVED",
            metrics=None,
            approved_at=now,
            approved_by="analyst-1",
            rejected_by=None,
            rejection_reason=None,
            created_at=now,
        )
    )
    session.add(
        ForecastRunRow(
            id="forecast-usage",
            model_version_id="model-usage",
            dataset_id="dataset-usage",
            dataset_version="1",
            horizon=7,
            granularity="day",
            status="SUCCEEDED",
            output_uri="",
            error_message=None,
            created_at=now,
        )
    )
    session.add(
        AIExplanationRow(
            id="explain-old",
            forecast_run_id="forecast-usage",
            scope_key='{"series":"old"}',
            scope={"series": "old"},
            model_id="mock-explainer",
            prompt_version="v1",
            input_tokens=100,
            output_tokens=100,
            latency_ms=1,
            explanation=None,
            package=None,
            status="invalid",
            created_at=datetime(2020, 1, 15, tzinfo=UTC),
        )
    )
    session.add(
        AIExplanationRow(
            id="explain-now",
            forecast_run_id="forecast-usage",
            scope_key='{"series":"now"}',
            scope={"series": "now"},
            model_id="mock-explainer",
            prompt_version="v1",
            input_tokens=12,
            output_tokens=8,
            latency_ms=3,
            explanation=None,
            package=None,
            status="valid",
            created_at=now,
        )
    )
    session.commit()
    session.close()
