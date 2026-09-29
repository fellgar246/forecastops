"""SageMaker batch transform for DeepAR forecasts.

The adapter submits one transform job and reads the object it writes under
``forecasts/``. It does not create a real-time or serverless endpoint.
``ONLINE_INFERENCE`` stays off: a process with that flag set is refused
before any client call.
"""

import json
from collections.abc import Mapping
from typing import Any, Protocol, cast

from forecastops_ml.inference.errors import InferenceError, sanitize_failure_reason
from forecastops_ml.inference.points import StoredForecastPoint, load_forecast_document

CPU_INSTANCE_TYPE = "ml.c5.xlarge"
_REGION = "us-east-1"
_JOB_PREFIX = "forecastops-batch"


class BatchTransformClient(Protocol):
    """The transform calls this adapter is allowed to make."""

    def create_transform_job(self, request: Mapping[str, object]) -> str:
        """Create one batch transform job and return its ARN."""

    def describe_transform_job(self, job_name: str) -> Mapping[str, object]:
        """Return the status of one transform job."""


class ForecastObjects(Protocol):
    """Read and write the transform input and output objects."""

    def put_bytes(self, uri: str, body: bytes) -> None:
        """Store ``body`` at ``uri``."""

    def get_bytes(self, uri: str) -> bytes:
        """Return the bytes stored at ``uri``."""


class SageMakerTransformApi(Protocol):
    """Subset of the transform API used by :class:`BotoBatchTransformClient`."""

    def create_transform_job(self, **kwargs: Any) -> Mapping[str, Any]:
        """Create one transform job."""

    def describe_transform_job(self, **kwargs: Any) -> Mapping[str, Any]:
        """Describe one transform job."""


class ObjectStorageApi(Protocol):
    """Subset of the object API used to move transform files."""

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> Mapping[str, Any]:
        """Write one object."""

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        """Read one object."""


class MemoryForecastObjects:
    """In-memory stand-in for the forecast object store."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_bytes(self, uri: str, body: bytes) -> None:
        """Store ``body`` at ``uri``."""

        self.objects[uri] = body

    def get_bytes(self, uri: str) -> bytes:
        """Return the bytes stored at ``uri``."""

        try:
            return self.objects[uri]
        except KeyError as exc:
            raise InferenceError("Batch inference output was not found.") from exc


class BotoBatchTransformClient:
    """Adapt the transform API to :class:`BatchTransformClient`."""

    def __init__(self, api: SageMakerTransformApi) -> None:
        self._api = api

    def create_transform_job(self, request: Mapping[str, object]) -> str:
        """Create one batch transform job and return its ARN."""

        response = self._api.create_transform_job(**cast(dict[str, Any], dict(request)))
        arn = response.get("TransformJobArn", "")
        if not isinstance(arn, str) or arn == "":
            raise InferenceError("Batch inference did not return a job id.")
        return arn

    def describe_transform_job(self, job_name: str) -> Mapping[str, object]:
        """Return the status of one transform job."""

        return self._api.describe_transform_job(TransformJobName=job_name)


class BotoForecastObjects:
    """Read and write ``s3://`` objects for a transform job."""

    def __init__(self, api: ObjectStorageApi) -> None:
        self._api = api

    def put_bytes(self, uri: str, body: bytes) -> None:
        """Store ``body`` at ``uri``."""

        bucket, key = split_s3_uri(uri)
        self._api.put_object(Bucket=bucket, Key=key, Body=body)

    def get_bytes(self, uri: str) -> bytes:
        """Return the bytes stored at ``uri``."""

        bucket, key = split_s3_uri(uri)
        response = self._api.get_object(Bucket=bucket, Key=key)
        body = response.get("Body")
        read = getattr(body, "read", None)
        if not callable(read):
            raise InferenceError("Batch inference output was not found.")
        payload = read()
        if not isinstance(payload, bytes):
            raise InferenceError("Batch inference output was not found.")
        return payload


class BatchInference:
    """Submit one DeepAR batch transform and load its forecast artifact."""

    def __init__(
        self,
        client: BatchTransformClient | None,
        objects: ForecastObjects | None,
        *,
        bucket: str,
        sagemaker_enabled: bool,
        online_inference: bool,
    ) -> None:
        self._client = client
        self._objects = objects
        self._bucket = bucket
        self._sagemaker_enabled = sagemaker_enabled
        self._online_inference = online_inference

    def run(
        self,
        *,
        run_id: str,
        model_name: str,
        horizon: int,
        granularity: str,
    ) -> tuple[StoredForecastPoint, ...]:
        """Run one batch job and return the points in its artifact.

        The transform request names a model and an output prefix under
        ``forecasts/``. It does not name an endpoint.
        """

        if self._online_inference:
            raise InferenceError("Batch inference does not create a real-time endpoint.")
        if not self._sagemaker_enabled:
            raise InferenceError("SageMaker batch inference is disabled.")
        client = self._client
        objects = self._objects
        if client is None or objects is None:
            raise InferenceError("SageMaker batch inference is disabled.")
        if self._bucket.strip() == "":
            raise InferenceError("Batch inference requires an artifacts bucket.")
        if type(horizon) is not int or horizon < 1:
            raise InferenceError("Forecast horizon must be a positive number of periods.")
        if granularity not in {"day", "week"}:
            raise InferenceError("Forecast granularity must be day or week.")

        job_name = transform_job_name(run_id)
        input_uri = f"s3://{self._bucket}/forecasts/{run_id}/input.json"
        output_prefix = f"s3://{self._bucket}/forecasts/{run_id}/"
        output_uri = f"{output_prefix}forecast.json"
        request = transform_request(
            job_name=job_name,
            model_name=model_name,
            input_uri=input_uri,
            output_prefix=output_prefix,
            horizon=horizon,
            granularity=granularity,
        )
        objects.put_bytes(input_uri, _input_body(model_name, horizon, granularity))
        client.create_transform_job(request)
        description = client.describe_transform_job(job_name)
        status = description.get("TransformJobStatus")
        if status == "Failed":
            raise InferenceError(sanitize_failure_reason(description.get("FailureReason")))
        if status != "Completed":
            raise InferenceError("Batch inference did not finish.")
        return load_forecast_document(objects.get_bytes(output_uri))


def transform_job_name(run_id: str) -> str:
    """Return a transform job name derived from ``run_id``."""

    cleaned = "".join(char if char.isalnum() or char == "-" else "-" for char in run_id)
    name = f"{_JOB_PREFIX}-{cleaned}".strip("-")
    if len(name) > 63:
        name = name[:63].rstrip("-")
    if name == "" or not name[0].isalnum():
        raise InferenceError("Batch inference could not name the transform job.")
    return name


def transform_request(
    *,
    job_name: str,
    model_name: str,
    input_uri: str,
    output_prefix: str,
    horizon: int,
    granularity: str,
) -> dict[str, object]:
    """Build a batch transform request that does not target an endpoint."""

    if "/forecasts/" not in output_prefix:
        raise InferenceError("Batch inference output must be written under forecasts/.")
    return {
        "TransformJobName": job_name,
        "ModelName": model_name,
        "BatchStrategy": "SingleRecord",
        "MaxPayloadInMB": 6,
        "TransformInput": {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": input_uri,
                }
            },
            "ContentType": "application/json",
            "SplitType": "None",
        },
        "TransformOutput": {
            "S3OutputPath": output_prefix,
            "AssembleWith": "None",
        },
        "TransformResources": {
            "InstanceType": CPU_INSTANCE_TYPE,
            "InstanceCount": 1,
        },
        "Environment": {
            "FORECAST_HORIZON": str(horizon),
            "FORECAST_GRANULARITY": granularity,
        },
    }


def open_batch_inference(*, region: str, bucket: str) -> BatchInference:
    """Open a live batch adapter. The caller has already checked the flags."""

    if region != _REGION:
        raise InferenceError("Batch inference is available in us-east-1.")
    if bucket.strip() == "":
        raise InferenceError("Batch inference requires an artifacts bucket.")
    import boto3

    sagemaker = boto3.client("sagemaker", region_name=region)
    storage = boto3.client("s3", region_name=region)
    return BatchInference(
        BotoBatchTransformClient(sagemaker),
        BotoForecastObjects(storage),
        bucket=bucket,
        sagemaker_enabled=True,
        online_inference=False,
    )


def split_s3_uri(uri: str) -> tuple[str, str]:
    """Return the bucket and key of an ``s3://`` uri."""

    prefix = "s3://"
    if not uri.startswith(prefix):
        raise InferenceError("Batch inference output was not found.")
    remainder = uri[len(prefix) :]
    bucket, separator, key = remainder.partition("/")
    if separator != "/" or bucket == "" or key == "":
        raise InferenceError("Batch inference output was not found.")
    return bucket, key


def _input_body(model_name: str, horizon: int, granularity: str) -> bytes:
    payload = {
        "granularity": granularity,
        "horizon": horizon,
        "model_name": model_name,
    }
    return (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
