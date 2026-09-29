"""Training-run record for one pipeline execution.

A finished execution stores the git SHA, dataset version, resolved
configuration, metric document, and artifact locations.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from forecastops_ml.evaluation.folds import MIN_PUBLISHED_FOLDS

_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHA = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ARTIFACTS = ("channels", "metrics", "model", "quality_report")
_STATUSES = {
    "QUEUED",
    "PREPROCESSING",
    "TRAINING",
    "EVALUATING",
    "REGISTERING",
    "COMPLETED",
    "FAILED",
}
FINISHED_STATUS = "COMPLETED"
STARTED_STATUS = "QUEUED"


@dataclass(frozen=True)
class TrainingRunRecord:
    """Metadata stored for one pipeline execution.

    ``artifact_uri`` is the model location. ``artifact_uris`` is the full set
    of locations for that execution. ``metrics`` is the evaluation document
    and is required once the execution has finished.
    """

    id: str
    dataset_version: str
    model_family: str
    configuration: Mapping[str, object]
    status: str
    artifact_uri: str
    artifact_uris: Mapping[str, str]
    metrics: Mapping[str, object] | None
    git_sha: str
    pipeline_execution_arn: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or self.id.strip() == "":
            raise ValueError("Training run id must be a non-empty string.")
        if self.status not in _STATUSES:
            raise ValueError(f"Training run status {self.status} is not a training status.")
        if self.model_family != "deepar":
            raise ValueError("This training pipeline records a deepar run.")
        object.__setattr__(self, "git_sha", require_git_sha(self.git_sha))
        object.__setattr__(self, "dataset_version", require_dataset_version(self.dataset_version))
        object.__setattr__(self, "configuration", require_configuration(self.configuration))
        locations = require_artifact_uris(self.artifact_uris)
        object.__setattr__(self, "artifact_uris", locations)
        object.__setattr__(self, "artifact_uri", locations["model"])
        arn = self.pipeline_execution_arn
        if not isinstance(arn, str) or not arn.startswith("arn:"):
            raise ValueError("A pipeline execution needs an ARN.")
        if self.status == FINISHED_STATUS:
            if self.metrics is None:
                raise ValueError("A finished training run needs a metric document.")
            object.__setattr__(self, "metrics", require_metric_document(self.metrics))
        elif self.metrics is not None:
            object.__setattr__(self, "metrics", dict(self.metrics))

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this training run."""

        return {
            "artifact_uri": self.artifact_uri,
            "artifact_uris": dict(self.artifact_uris),
            "configuration": dict(self.configuration),
            "dataset_version": self.dataset_version,
            "git_sha": self.git_sha,
            "id": self.id,
            "metrics": None if self.metrics is None else dict(self.metrics),
            "model_family": self.model_family,
            "pipeline_execution_arn": self.pipeline_execution_arn,
            "status": self.status,
        }


class TrainingRunStore(Protocol):
    """Persist one training-run record."""

    def save(self, record: TrainingRunRecord) -> None:
        """Store ``record``."""


class MemoryTrainingRunStore:
    """In-memory store used by tests and local checks."""

    def __init__(self) -> None:
        self.records: list[TrainingRunRecord] = []

    def save(self, record: TrainingRunRecord) -> None:
        """Append ``record``."""

        self.records.append(record)


def require_git_sha(value: str) -> str:
    """Return ``value`` when it can be stored as a git SHA."""

    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError("git SHA must be 1 to 64 letters, digits, dots, underscores, or hyphens.")
    return value


def require_dataset_version(value: str) -> str:
    """Return ``value`` when it can be used as a dataset version and object key."""

    if not isinstance(value, str) or _VERSION.fullmatch(value) is None:
        raise ValueError(
            "Dataset version must be 1 to 64 letters, digits, dots, underscores, or hyphens."
        )
    return value


def require_configuration(configuration: Mapping[str, object]) -> dict[str, object]:
    """Return a resolved configuration object for a deepar training run."""

    if not isinstance(configuration, Mapping):
        raise ValueError("Configuration must be a JSON object.")
    if configuration.get("model_family") != "deepar":
        raise ValueError("The training pipeline configuration uses model_family deepar.")
    return dict(configuration)


def require_artifact_uris(locations: Mapping[str, str]) -> dict[str, str]:
    """Return artifact locations, including the model, metrics, channels, and report."""

    if not isinstance(locations, Mapping):
        raise ValueError("Artifact locations need channels, metrics, model, and quality_report.")
    cleaned: dict[str, str] = {}
    for key, uri in locations.items():
        if not isinstance(key, str) or key.strip() == "":
            raise ValueError("Artifact location names must be non-empty strings.")
        if not isinstance(uri, str) or not uri.startswith("s3://"):
            raise ValueError(f"Artifact location {key} must be an s3 URI.")
        cleaned[key] = uri
    missing = [key for key in _ARTIFACTS if key not in cleaned]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Artifact locations need {names}.")
    return cleaned


def require_metric_document(metrics: Mapping[str, object]) -> dict[str, object]:
    """Return a metric document from the rolling-origin evaluation."""

    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("A finished training run needs a metric document.")
    fold_count = metrics.get("fold_count")
    if type(fold_count) is not int or fold_count < MIN_PUBLISHED_FOLDS:
        minimum = MIN_PUBLISHED_FOLDS
        raise ValueError(f"The metric document needs at least {minimum} rolling-origin folds.")
    if metrics.get("model_family") != "deepar":
        raise ValueError("The metric document must name model_family deepar.")
    nested = metrics.get("metrics")
    if not isinstance(nested, Mapping):
        raise ValueError("The metric document needs global metrics.")
    global_metrics = nested.get("global")
    if not isinstance(global_metrics, Mapping) or "wape" not in global_metrics:
        raise ValueError("The metric document needs a global WAPE.")
    return dict(metrics)


def finished_training_run(
    *,
    git_sha: str,
    dataset_version: str,
    configuration: Mapping[str, object],
    metrics: Mapping[str, object],
    artifact_uris: Mapping[str, str],
    pipeline_execution_arn: str,
    run_id: str | None = None,
) -> TrainingRunRecord:
    """Build the record stored for a finished execution."""

    return TrainingRunRecord(
        id=run_id if run_id is not None else str(uuid4()),
        dataset_version=dataset_version,
        model_family="deepar",
        configuration=configuration,
        status=FINISHED_STATUS,
        artifact_uri="",
        artifact_uris=artifact_uris,
        metrics=metrics,
        git_sha=git_sha,
        pipeline_execution_arn=pipeline_execution_arn,
    )


def started_training_run(
    *,
    git_sha: str,
    dataset_version: str,
    configuration: Mapping[str, object],
    artifact_uris: Mapping[str, str],
    pipeline_execution_arn: str,
    run_id: str | None = None,
) -> TrainingRunRecord:
    """Build the record stored when an execution has been started."""

    return TrainingRunRecord(
        id=run_id if run_id is not None else str(uuid4()),
        dataset_version=dataset_version,
        model_family="deepar",
        configuration=configuration,
        status=STARTED_STATUS,
        artifact_uri="",
        artifact_uris=artifact_uris,
        metrics=None,
        git_sha=git_sha,
        pipeline_execution_arn=pipeline_execution_arn,
    )


def store_finished_execution(
    *,
    git_sha: str,
    dataset_version: str,
    configuration: Mapping[str, object],
    metrics: Mapping[str, object],
    artifact_uris: Mapping[str, str],
    pipeline_execution_arn: str,
    store: TrainingRunStore,
    run_id: str | None = None,
) -> TrainingRunRecord:
    """Validate a finished execution and save it on ``store``."""

    record = finished_training_run(
        git_sha=git_sha,
        dataset_version=dataset_version,
        configuration=configuration,
        metrics=metrics,
        artifact_uris=artifact_uris,
        pipeline_execution_arn=pipeline_execution_arn,
        run_id=run_id,
    )
    store.save(record)
    return record


def artifact_locations(bucket: str, dataset_version: str) -> dict[str, str]:
    """Return the object locations for one dataset version."""

    version = require_dataset_version(dataset_version)
    if not isinstance(bucket, str) or bucket.strip() == "" or "/" in bucket:
        raise ValueError("Artifacts bucket must be a non-empty bucket name.")
    root = f"s3://{bucket.strip()}"
    return {
        "channels": f"{root}/processed/{version}/channels/",
        "metrics": f"{root}/evaluations/{version}/metrics.json",
        "model": f"{root}/models/{version}/",
        "quality_report": f"{root}/processed/{version}/quality/quality_report.json",
    }
