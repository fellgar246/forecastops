"""Submit one manual DeepAR training job.

The job runs on a single CPU instance, stops at the runtime ceiling, and uses
fixed hyperparameters. The training API is not called unless SageMaker,
training, and AWS ML are all enabled and GPU training stays off.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol, cast

from forecastops_ml.training.deepar_export import DeepARChannels
from forecastops_ml.training.deepar_forecast import PREDICTION_QUANTILES

CPU_INSTANCE_TYPE = "ml.c5.xlarge"
RUNTIME_CEILING_MINUTES = 45
DAILY_JOB_CEILING = 2
# Public SageMaker DeepAR algorithm image for us-east-1, version 1.
DEEPAR_TRAINING_IMAGE = "522234722520.dkr.ecr.us-east-1.amazonaws.com/forecasting-deepar:1"
_JOB_PREFIX = "forecastops-deepar"
_GPU_PREFIXES = ("ml.p", "ml.g", "ml.trn")


@dataclass(frozen=True)
class TrainingGates:
    """Flags and ceilings that decide whether a job may be submitted."""

    sagemaker_enabled: bool
    training_enabled: bool
    aws_ml_enabled: bool
    allow_gpu_training: bool
    max_training_runtime_minutes: int
    max_training_jobs_per_day: int


@dataclass(frozen=True)
class TrainingSubmission:
    """Result of attempting to submit one training job."""

    submitted: bool
    reason: str
    exit_code: int
    job_name: str = ""
    request: Mapping[str, object] | None = None


class TrainingJobClient(Protocol):
    """The two training-service calls this command is allowed to make."""

    def training_jobs_started_today(self, *, day: date) -> int:
        """Return how many DeepAR jobs were created on ``day``."""

    def create_training_job(self, request: Mapping[str, object]) -> str:
        """Create one training job and return its ARN."""


class ObjectUploader(Protocol):
    """Write channel bytes to the artifact bucket."""

    def put_bytes(self, bucket: str, key: str, body: bytes) -> None:
        """Store ``body`` at ``key``."""


class SageMakerCreateApi(Protocol):
    """Subset of the training API used by :class:`BotoTrainingJobClient`."""

    def create_training_job(self, **kwargs: Any) -> Mapping[str, Any]:
        """Create one training job."""

    def list_training_jobs(self, **kwargs: Any) -> Mapping[str, Any]:
        """List training jobs."""


class S3PutApi(Protocol):
    """Subset of the object API used to upload channels."""

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> Mapping[str, Any]:
        """Write one object."""


class BotoTrainingJobClient:
    """Adapt the training API to :class:`TrainingJobClient`."""

    def __init__(self, api: SageMakerCreateApi) -> None:
        self._api = api

    def training_jobs_started_today(self, *, day: date) -> int:
        """Count DeepAR jobs created on ``day`` in UTC."""

        start = datetime(day.year, day.month, day.day, tzinfo=UTC)
        token: str | None = None
        seen: set[str] = set()
        count = 0
        while True:
            kwargs: dict[str, Any] = {
                "CreationTimeAfter": start,
                "NameContains": _JOB_PREFIX,
                "MaxResults": 100,
                "SortBy": "CreationTime",
                "SortOrder": "Descending",
            }
            if token is not None:
                kwargs["NextToken"] = token
            page = self._api.list_training_jobs(**kwargs)
            summaries = page.get("TrainingJobSummaries", [])
            if isinstance(summaries, list):
                count += len(summaries)
            next_token = page.get("NextToken")
            if not isinstance(next_token, str) or next_token in seen:
                return count
            seen.add(next_token)
            token = next_token

    def create_training_job(self, request: Mapping[str, object]) -> str:
        """Create one training job and return its ARN."""

        response = self._api.create_training_job(**cast(dict[str, Any], dict(request)))
        arn = response.get("TrainingJobArn", "")
        if not isinstance(arn, str) or arn == "":
            raise ValueError("The training service did not return a job ARN.")
        return arn


class BotoObjectUploader:
    """Adapt the object API to :class:`ObjectUploader`."""

    def __init__(self, api: S3PutApi) -> None:
        self._api = api

    def put_bytes(self, bucket: str, key: str, body: bytes) -> None:
        """Store ``body`` at ``key``."""

        self._api.put_object(Bucket=bucket, Key=key, Body=body)


def open_aws_clients(region: str) -> tuple[TrainingJobClient, ObjectUploader]:
    """Build live training and object clients for ``region``."""

    import boto3

    sagemaker = boto3.client("sagemaker", region_name=region)
    storage = boto3.client("s3", region_name=region)
    return BotoTrainingJobClient(sagemaker), BotoObjectUploader(storage)


def gates_from_environ(environ: Mapping[str, str]) -> TrainingGates:
    """Read training flags. Missing enable flags stay off."""

    return TrainingGates(
        sagemaker_enabled=_bool_flag(environ, "SAGEMAKER_ENABLED", default=False),
        training_enabled=_bool_flag(environ, "TRAINING_ENABLED", default=False),
        aws_ml_enabled=_bool_flag(environ, "AWS_ML_ENABLED", default=False),
        allow_gpu_training=_bool_flag(environ, "ALLOW_GPU_TRAINING", default=False),
        max_training_runtime_minutes=_positive_int(
            environ,
            "MAX_TRAINING_RUNTIME_MINUTES",
            default=RUNTIME_CEILING_MINUTES,
        ),
        max_training_jobs_per_day=_non_negative_int(
            environ,
            "MAX_TRAINING_JOBS_PER_DAY",
            default=DAILY_JOB_CEILING,
        ),
    )


def block_reason(gates: TrainingGates) -> str | None:
    """Return why a job must not be submitted, or none when the gates are open."""

    reasons: list[str] = []
    if not gates.sagemaker_enabled:
        reasons.append("SAGEMAKER_ENABLED is false")
    if not gates.training_enabled:
        reasons.append("TRAINING_ENABLED is false")
    if not gates.aws_ml_enabled:
        reasons.append("AWS_ML_ENABLED is false")
    if gates.allow_gpu_training:
        reasons.append("ALLOW_GPU_TRAINING is true")
    if gates.max_training_runtime_minutes > RUNTIME_CEILING_MINUTES:
        configured = gates.max_training_runtime_minutes
        reasons.append(
            f"MAX_TRAINING_RUNTIME_MINUTES is {configured}, above {RUNTIME_CEILING_MINUTES}"
        )
    if gates.max_training_jobs_per_day > DAILY_JOB_CEILING:
        configured = gates.max_training_jobs_per_day
        reasons.append(f"MAX_TRAINING_JOBS_PER_DAY is {configured}, above {DAILY_JOB_CEILING}")
    if not reasons:
        return None
    detail = "; ".join(reasons)
    return f"DeepAR training was not submitted because {detail}."


def require_cpu_instance(instance_type: str) -> str:
    """Return ``instance_type`` when it is a CPU training instance."""

    lowered = instance_type.lower()
    if lowered.startswith(_GPU_PREFIXES) or "gpu" in lowered:
        raise ValueError("DeepAR training uses a CPU instance. GPU training is disabled.")
    if not (lowered.startswith("ml.c") or lowered.startswith("ml.m")):
        raise ValueError(f"DeepAR training instance {instance_type} is not a CPU instance.")
    return instance_type


def deepar_training_image(region: str) -> str:
    """Return the DeepAR training image for ``region``."""

    if region != "us-east-1":
        raise ValueError("DeepAR training is configured for the us-east-1 image.")
    return DEEPAR_TRAINING_IMAGE


def build_training_job_request(
    channels: DeepARChannels,
    *,
    gates: TrainingGates,
    bucket: str,
    role_arn: str,
    region: str,
    job_name: str,
) -> dict[str, object]:
    """Build one CreateTrainingJob request with fixed hyperparameters."""

    instance_type = require_cpu_instance(CPU_INSTANCE_TYPE)
    image = deepar_training_image(region)
    model_prefix = f"s3://{bucket}/models/{job_name}/"
    return {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": image,
            "TrainingInputMode": "File",
        },
        "RoleArn": role_arn,
        "InputDataConfig": [
            _channel(bucket, job_name, "train"),
            _channel(bucket, job_name, "test"),
        ],
        "OutputDataConfig": {"S3OutputPath": model_prefix},
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 10,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": gates.max_training_runtime_minutes * 60,
        },
        "HyperParameters": _hyperparameters(channels),
    }


def submit_deepar_training(
    channels: DeepARChannels,
    *,
    gates: TrainingGates,
    bucket: str,
    role_arn: str,
    region: str,
    client: TrainingJobClient | None,
    uploader: ObjectUploader | None,
    now: datetime | None = None,
) -> TrainingSubmission:
    """Upload channels and submit one job when the gates are open.

    A closed gate returns before any client call. A daily cap that is already
    reached lists jobs and does not create one.
    """

    reason = block_reason(gates)
    if reason is not None:
        return TrainingSubmission(submitted=False, reason=reason, exit_code=0)
    if client is None or uploader is None:
        raise ValueError("An open training gate needs a training client and an uploader.")
    moment = _utc_now(now)
    try:
        deepar_training_image(region)
    except ValueError as exc:
        return TrainingSubmission(submitted=False, reason=str(exc), exit_code=1)
    started = client.training_jobs_started_today(day=moment.date())
    if started >= gates.max_training_jobs_per_day:
        limit = gates.max_training_jobs_per_day
        return TrainingSubmission(
            submitted=False,
            reason=(
                "DeepAR training was not submitted because "
                f"{started} training jobs already started today and the daily limit is {limit}."
            ),
            exit_code=1,
        )
    job_name = f"{_JOB_PREFIX}-{moment.strftime('%Y%m%d%H%M%S')}"
    request = build_training_job_request(
        channels,
        gates=gates,
        bucket=bucket,
        role_arn=role_arn,
        region=region,
        job_name=job_name,
    )
    uploader.put_bytes(
        bucket,
        f"training/{job_name}/train/train.json",
        _utf8(channels.jsonl("train")),
    )
    uploader.put_bytes(
        bucket,
        f"training/{job_name}/test/test.json",
        _utf8(channels.jsonl("test")),
    )
    uploader.put_bytes(
        bucket,
        f"training/{job_name}/limits.json",
        _utf8(_limits_json(channels, gates)),
    )
    client.create_training_job(request)
    model_uri = f"s3://{bucket}/models/{job_name}/"
    training_uri = f"s3://{bucket}/training/{job_name}/"
    return TrainingSubmission(
        submitted=True,
        reason=(
            f"Submitted DeepAR training job {job_name}. "
            f"Training channels are under {training_uri} and model artifacts are under {model_uri}."
        ),
        exit_code=0,
        job_name=job_name,
        request=request,
    )


def _channel(bucket: str, job_name: str, name: str) -> dict[str, object]:
    return {
        "ChannelName": name,
        "ContentType": "application/jsonlines",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{bucket}/training/{job_name}/{name}/",
                "S3DataDistributionType": "FullyReplicated",
            }
        },
    }


def _hyperparameters(channels: DeepARChannels) -> dict[str, str]:
    batch = min(32, len(channels.train))
    quantiles = ", ".join(str(quantile) for quantile in PREDICTION_QUANTILES)
    cardinality = f"[{channels.store_cardinality}, {channels.category_cardinality}]"
    return {
        "cardinality": cardinality,
        "context_length": str(channels.prediction_length),
        "dropout_rate": "0.1",
        "epochs": "20",
        "learning_rate": "0.001",
        "likelihood": "negative-binomial",
        "mini_batch_size": str(batch),
        "num_cells": "40",
        "num_dynamic_feat": "2",
        "num_eval_samples": "100",
        "num_layers": "2",
        "prediction_length": str(channels.prediction_length),
        "test_quantiles": f"[{quantiles}]",
        "time_freq": channels.frequency,
    }


def _limits_json(channels: DeepARChannels, gates: TrainingGates) -> str:
    body = {
        "allow_gpu_training": False,
        "frequency": channels.frequency,
        "grain": channels.grain,
        "hyperparameter_search": False,
        "instance_count": 1,
        "instance_type": CPU_INSTANCE_TYPE,
        "max_training_jobs_per_day": gates.max_training_jobs_per_day,
        "max_training_runtime_minutes": gates.max_training_runtime_minutes,
        "model_family": "deepar",
        "prediction_length": channels.prediction_length,
        "quantiles": list(PREDICTION_QUANTILES),
        "series_count": len(channels.train),
    }
    return json.dumps(body, indent=2, sort_keys=True) + "\n"


def _bool_flag(environ: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{name} must be true or false.")


def _positive_int(environ: Mapping[str, str], name: str, *, default: int) -> int:
    number = _int_flag(environ, name, default=default)
    if number < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return number


def _non_negative_int(environ: Mapping[str, str], name: str, *, default: int) -> int:
    number = _int_flag(environ, name, default=default)
    if number < 0:
        raise ValueError(f"{name} must be zero or a positive integer.")
    return number


def _int_flag(environ: Mapping[str, str], name: str, *, default: int) -> int:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _utc_now(now: datetime | None) -> datetime:
    moment = now if now is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("now must be timezone-aware.")
    return moment.astimezone(UTC)


def _utf8(text: str) -> bytes:
    return text.encode("utf-8")
