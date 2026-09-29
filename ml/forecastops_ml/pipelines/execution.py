"""Start one training pipeline execution.

The pipeline API is not called unless SageMaker, training, and AWS ML are all
enabled and the other training limits stay inside their ceilings. This module
does not register a schedule.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from forecastops_ml.pipelines.definition import (
    TrainingPipeline,
    planned_locations,
    render_pipeline,
)
from forecastops_ml.pipelines.metadata import (
    TrainingRunRecord,
    TrainingRunStore,
    finished_training_run,
    started_training_run,
)
from forecastops_ml.training.deepar_job import (
    ObjectUploader,
    TrainingGates,
    block_reason,
)

PIPELINE_NAME = "forecastops-training"
_DEEPAR_PREFIX = "DeepAR training was not submitted because "


@dataclass(frozen=True)
class PipelineExecutionResult:
    """One pipeline execution returned by the pipeline service."""

    arn: str
    status: str
    finished: bool = False
    metrics: Mapping[str, object] | None = None
    artifact_uris: Mapping[str, str] | None = None


@dataclass(frozen=True)
class PipelineSubmission:
    """Result of attempting to start one pipeline execution."""

    started: bool
    reason: str
    exit_code: int
    execution_arn: str = ""
    training_run: TrainingRunRecord | None = None


class MissingPipeline(Exception):
    """The named pipeline does not exist yet."""


class PipelineClient(Protocol):
    """The pipeline calls this command is allowed to make."""

    def upsert_pipeline(
        self,
        *,
        name: str,
        definition: Mapping[str, Any],
        role_arn: str,
    ) -> str:
        """Create or update the pipeline and return its ARN."""

    def start_execution(
        self,
        *,
        pipeline_name: str,
        parameters: Mapping[str, str],
    ) -> PipelineExecutionResult:
        """Start one execution and return its handle."""


class SageMakerPipelineApi(Protocol):
    """Subset of the pipeline API used by :class:`BotoPipelineClient`."""

    def create_pipeline(self, **kwargs: Any) -> Mapping[str, Any]:
        """Create one pipeline."""

    def update_pipeline(self, **kwargs: Any) -> Mapping[str, Any]:
        """Update one pipeline."""

    def describe_pipeline(self, **kwargs: Any) -> Mapping[str, Any]:
        """Describe one pipeline."""

    def start_pipeline_execution(self, **kwargs: Any) -> Mapping[str, Any]:
        """Start one execution."""


class BotoPipelineClient:
    """Adapt the pipeline API to :class:`PipelineClient`."""

    def __init__(self, api: SageMakerPipelineApi) -> None:
        self._api = api

    def upsert_pipeline(
        self,
        *,
        name: str,
        definition: Mapping[str, Any],
        role_arn: str,
    ) -> str:
        """Create the pipeline, or update it when it already exists."""

        body = json.dumps(definition)
        try:
            self._api.describe_pipeline(PipelineName=name)
        except Exception as exc:
            if not _missing_pipeline(exc):
                raise
            response = self._api.create_pipeline(
                PipelineName=name,
                PipelineDefinition=body,
                RoleArn=role_arn,
            )
        else:
            response = self._api.update_pipeline(
                PipelineName=name,
                PipelineDefinition=body,
                RoleArn=role_arn,
            )
        return _pipeline_arn(response)

    def start_execution(
        self,
        *,
        pipeline_name: str,
        parameters: Mapping[str, str],
    ) -> PipelineExecutionResult:
        """Start one execution. The call does not wait for it to finish."""

        response = self._api.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineParameters=[
                {"Name": name, "Value": value} for name, value in parameters.items()
            ],
        )
        arn = response.get("PipelineExecutionArn", "")
        if not isinstance(arn, str) or not arn.startswith("arn:"):
            raise ValueError("The pipeline service did not return an execution ARN.")
        return PipelineExecutionResult(arn=arn, status="QUEUED", finished=False)


class ObjectTrainingRunStore:
    """Write the training-run document into the artifact bucket."""

    def __init__(self, uploader: ObjectUploader, bucket: str) -> None:
        self._uploader = uploader
        self._bucket = bucket

    def save(self, record: TrainingRunRecord) -> None:
        """Store the training run as JSON."""

        body = json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"
        key = f"training/runs/{record.id}/training_run.json"
        self._uploader.put_bytes(self._bucket, key, body.encode("utf-8"))


def open_pipeline_clients(region: str) -> tuple[PipelineClient, ObjectUploader]:
    """Build live pipeline and object clients for ``region``."""

    import boto3

    from forecastops_ml.training.deepar_job import BotoObjectUploader

    pipeline = boto3.client("sagemaker", region_name=region)
    storage = boto3.client("s3", region_name=region)
    return BotoPipelineClient(pipeline), BotoObjectUploader(storage)


def pipeline_block_reason(gates: TrainingGates) -> str | None:
    """Return why an execution must not start, or none when the gates are open."""

    reason = block_reason(gates)
    if reason is None:
        return None
    if reason.startswith(_DEEPAR_PREFIX):
        detail = reason[len(_DEEPAR_PREFIX) :]
        return f"The training pipeline was not started because {detail}"
    return f"The training pipeline was not started. {reason}"


def start_pipeline(
    pipeline: TrainingPipeline,
    *,
    gates: TrainingGates,
    jobs_started_today: int,
    client: PipelineClient | None,
    store: TrainingRunStore | None,
) -> PipelineSubmission:
    """Upsert the pipeline and start one execution when the gates are open.

    A closed gate returns before any client call. A daily cap that is already
    reached does not create or start an execution.
    """

    reason = pipeline_block_reason(gates)
    if reason is not None:
        return PipelineSubmission(started=False, reason=reason, exit_code=0)
    if jobs_started_today >= gates.max_training_jobs_per_day:
        limit = gates.max_training_jobs_per_day
        return PipelineSubmission(
            started=False,
            reason=(
                "The training pipeline was not started because "
                f"{jobs_started_today} training jobs already started today "
                f"and the daily limit is {limit}."
            ),
            exit_code=1,
        )
    rendered = render_pipeline(pipeline)
    if client is None or store is None:
        raise ValueError("An open training gate needs a pipeline client and a training run store.")
    client.upsert_pipeline(
        name=PIPELINE_NAME,
        definition=rendered.document,
        role_arn=pipeline.pipeline_role_arn,
    )
    result = client.start_execution(
        pipeline_name=PIPELINE_NAME,
        parameters=rendered.execution_parameters,
    )
    locations = (
        result.artifact_uris
        if result.artifact_uris is not None
        else planned_locations(pipeline.bucket, rendered.dataset_version)
    )
    if result.finished:
        if result.metrics is None:
            raise ValueError("A finished training run needs a metric document.")
        record = finished_training_run(
            git_sha=rendered.git_sha,
            dataset_version=rendered.dataset_version,
            configuration=rendered.configuration,
            metrics=result.metrics,
            artifact_uris=locations,
            pipeline_execution_arn=result.arn,
        )
    else:
        record = started_training_run(
            git_sha=rendered.git_sha,
            dataset_version=rendered.dataset_version,
            configuration=rendered.configuration,
            artifact_uris=locations,
            pipeline_execution_arn=result.arn,
        )
    store.save(record)
    return PipelineSubmission(
        started=True,
        reason=(
            f"Started training pipeline execution {result.arn}. "
            f"Training run {record.id} is {record.status}."
        ),
        exit_code=0,
        execution_arn=result.arn,
        training_run=record,
    )


def _missing_pipeline(exc: Exception) -> bool:
    if isinstance(exc, MissingPipeline):
        return True
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    if not isinstance(error, dict):
        return False
    return error.get("Code") == "ResourceNotFound"


def _pipeline_arn(response: Mapping[str, Any]) -> str:
    arn = response.get("PipelineArn", "")
    if not isinstance(arn, str) or not arn.startswith("arn:"):
        raise ValueError("The pipeline service did not return a pipeline ARN.")
    return arn
