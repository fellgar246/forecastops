"""Batch inference state machine, artifact reload, and the transform adapter."""

import builtins
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pyarrow as pa
import pytest

from forecastops_ml.inference import (
    BatchCapacityError,
    BatchInference,
    InferenceError,
    MemoryForecastObjects,
    StoredForecastPoint,
    admit_forecast,
    advance_status,
    execute_prediction,
    forecast_document,
    load_forecast_document,
    open_batch_inference,
    public_inference_error,
    quantile_forecast,
)
from forecastops_ml.inference.batch import CPU_INSTANCE_TYPE, transform_request

ROOT = Path(__file__).resolve().parents[2]
INFRA = ROOT / "infra"


def test_status_moves_from_queued_to_succeeded_or_failed() -> None:
    assert advance_status("QUEUED", "start") == "RUNNING"
    assert advance_status("RUNNING", "succeed") == "SUCCEEDED"
    assert advance_status("RUNNING", "fail") == "FAILED"
    assert advance_status("QUEUED", "fail") == "FAILED"
    with pytest.raises(InferenceError, match="SUCCEEDED"):
        advance_status("SUCCEEDED", "start")


def test_daily_cap_rejects_a_new_job_and_allows_a_replay() -> None:
    with pytest.raises(BatchCapacityError, match="limit of 5"):
        admit_forecast(jobs_started_today=5, daily_limit=5, replay=False)
    admit_forecast(jobs_started_today=5, daily_limit=5, replay=True)


def test_prediction_failure_drops_dataset_rows() -> None:
    leaked = '{"sku_id": "sku-secret-999", "units_sold": 424242}'

    def predict() -> tuple[StoredForecastPoint, ...]:
        raise RuntimeError(f"failed on row {leaked}")

    result = execute_prediction(predict)

    assert result.status == "FAILED"
    assert result.points == ()
    assert result.error_message == "Batch inference failed."
    assert result.error_message is not None
    assert "sku-secret-999" not in result.error_message
    assert "424242" not in result.error_message
    assert public_inference_error(RuntimeError(leaked)) == "Batch inference failed."


def test_fake_quantile_model_round_trips_through_the_artifact() -> None:
    frame = pa.table(
        {
            "date": pa.array([date(2026, 1, 1), date(2026, 1, 2)], type=pa.date32()),
            "store_id": ["store-1", "store-1"],
            "sku_id": ["sku-1", "sku-1"],
            "units_sold": [10.0, 20.0],
        }
    )

    points = quantile_forecast(frame, horizon=2, granularity="day")
    loaded = load_forecast_document(forecast_document(points))

    assert [(point.date.isoformat(), point.p10, point.p50, point.p90) for point in loaded] == [
        ("2026-01-03", 16.0, 20.0, 24.0),
        ("2026-01-04", 16.0, 20.0, 24.0),
    ]
    assert all(point.p10 is not None and point.p10 <= point.p50 <= point.p90 for point in loaded)  # type: ignore[operator]


def test_disabled_adapter_does_not_call_the_client() -> None:
    client = BombClient()
    adapter = BatchInference(
        client,
        MemoryForecastObjects(),
        bucket="forecastops-artifacts",
        sagemaker_enabled=False,
        online_inference=False,
    )

    with pytest.raises(InferenceError, match="disabled"):
        adapter.run(run_id="run-1", model_name="model-1", horizon=7, granularity="day")

    assert client.requests == []


def test_online_inference_is_refused_before_a_client_call() -> None:
    client = BombClient()
    adapter = BatchInference(
        client,
        MemoryForecastObjects(),
        bucket="forecastops-artifacts",
        sagemaker_enabled=True,
        online_inference=True,
    )

    with pytest.raises(InferenceError, match="real-time endpoint"):
        adapter.run(run_id="run-1", model_name="model-1", horizon=7, granularity="day")

    assert client.requests == []
    assert client.endpoint_calls == 0


def test_completed_transform_uses_a_cpu_batch_job_under_forecasts() -> None:
    objects = MemoryForecastObjects()
    output = forecast_document(
        (
            StoredForecastPoint(
                series_id="store-1|sku-1",
                date=date(2026, 1, 8),
                p10=8.0,
                p50=10.0,
                p90=12.0,
            ),
        )
    )
    client = CompletingClient(objects, output)
    adapter = BatchInference(
        client,
        objects,
        bucket="forecastops-artifacts",
        sagemaker_enabled=True,
        online_inference=False,
    )

    points = adapter.run(run_id="run-1", model_name="model-1", horizon=7, granularity="day")

    assert points[0].p10 == 8.0
    assert points[0].p50 == 10.0
    assert points[0].p90 == 12.0
    request = client.requests[0]
    assert "EndpointName" not in request
    assert "EndpointConfigName" not in request
    output_path = request["TransformOutput"]["S3OutputPath"]  # type: ignore[index]
    assert output_path == "s3://forecastops-artifacts/forecasts/run-1/"
    resources = request["TransformResources"]
    assert resources["InstanceType"] == CPU_INSTANCE_TYPE  # type: ignore[index]
    assert resources["InstanceCount"] == 1  # type: ignore[index]
    assert client.endpoint_calls == 0
    stored = json.loads(objects.get_bytes("s3://forecastops-artifacts/forecasts/run-1/input.json"))
    assert "units_sold" not in stored


def test_transform_failure_reason_does_not_keep_dataset_rows() -> None:
    client = FailingClient('{"sku_id": "sku-secret-999", "units_sold": 424242}')
    adapter = BatchInference(
        client,
        MemoryForecastObjects(),
        bucket="forecastops-artifacts",
        sagemaker_enabled=True,
        online_inference=False,
    )

    with pytest.raises(InferenceError, match="Batch inference failed") as caught:
        adapter.run(run_id="run-1", model_name="model-1", horizon=7, granularity="day")

    assert "sku-secret-999" not in str(caught.value)
    assert "424242" not in str(caught.value)


def test_transform_request_has_no_endpoint_fields() -> None:
    request = transform_request(
        job_name="forecastops-batch-run-1",
        model_name="model-1",
        input_uri="s3://forecastops-artifacts/forecasts/run-1/input.json",
        output_prefix="s3://forecastops-artifacts/forecasts/run-1/",
        horizon=7,
        granularity="day",
    )

    assert "create_endpoint" not in json.dumps(request).lower()
    assert "EndpointName" not in request


def test_opening_the_adapter_checks_region_and_bucket_before_boto(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name == "boto3" or name.startswith("boto3."):
            raise AssertionError("boto3 was imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)

    with pytest.raises(InferenceError, match="us-east-1"):
        open_batch_inference(region="eu-west-1", bucket="forecastops-artifacts")
    with pytest.raises(InferenceError, match="artifacts bucket"):
        open_batch_inference(region="us-east-1", bucket=" ")


def test_inference_code_does_not_create_an_endpoint() -> None:
    root = ROOT / "ml" / "forecastops_ml" / "inference"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))

    assert "create_endpoint" not in text
    assert "CreateEndpoint" not in text
    assert "aws_sagemaker_endpoint" not in text


def test_default_infrastructure_declares_no_realtime_endpoint() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in INFRA.rglob("*.tf"))

    assert "aws_sagemaker_endpoint" not in combined
    assert "aws_sagemaker_endpoint_configuration" not in combined
    assert "aws_sagemaker_serverless" not in combined


class BombClient:
    """Client that fails if a transform or endpoint call is made."""

    def __init__(self) -> None:
        self.requests: list[Mapping[str, object]] = []
        self.endpoint_calls = 0

    def create_transform_job(self, request: Mapping[str, object]) -> str:
        self.requests.append(request)
        raise AssertionError("create_transform_job was called")

    def describe_transform_job(self, job_name: str) -> Mapping[str, object]:
        raise AssertionError(f"describe_transform_job was called for {job_name}")

    def create_endpoint(self, **kwargs: object) -> str:
        self.endpoint_calls += 1
        raise AssertionError(f"create_endpoint was called with {kwargs}")


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
        raise AssertionError(f"create_endpoint was called with {kwargs}")


class FailingClient:
    """Fake transform client that fails with a caller-supplied reason."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.requests: list[Mapping[str, object]] = []

    def create_transform_job(self, request: Mapping[str, object]) -> str:
        self.requests.append(request)
        return "arn:aws:sagemaker:us-east-1:000000000000:transform-job/failed"

    def describe_transform_job(self, job_name: str) -> Mapping[str, object]:
        return {
            "TransformJobStatus": "Failed",
            "TransformJobName": job_name,
            "FailureReason": self.reason,
        }
