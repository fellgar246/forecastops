"""Command line entry for one manual training pipeline execution."""

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from forecastops_ml.pipelines.definition import TrainingPipeline
from forecastops_ml.pipelines.execution import (
    ObjectTrainingRunStore,
    PipelineClient,
    PipelineSubmission,
    open_pipeline_clients,
    pipeline_block_reason,
    start_pipeline,
)
from forecastops_ml.pipelines.metadata import TrainingRunStore
from forecastops_ml.training.deepar_job import (
    ObjectUploader,
    deepar_training_image,
    gates_from_environ,
    open_aws_clients,
)

ClientFactory = Callable[[str], tuple[PipelineClient, ObjectUploader]]


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    client_factory: ClientFactory | None = None,
    store: TrainingRunStore | None = None,
    started_today: int | None = None,
) -> int:
    """Start one pipeline execution, or stop when a training flag is closed."""

    parser = argparse.ArgumentParser(description="Start one manual training pipeline execution.")
    parser.add_argument("--dataset-version", required=True, help="Dataset version to train on.")
    parser.add_argument("--git-sha", required=True, help="Git revision that produced this run.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON file with the resolved training configuration.",
    )
    args = parser.parse_args(argv)
    env = environ if environ is not None else _process_environ()
    try:
        gates = gates_from_environ(env)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    reason = pipeline_block_reason(gates)
    if reason is not None:
        print(reason)
        return 0
    try:
        configuration = _load_config(args.config)
        deepar_training_image(_region(env))
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    bucket, pipeline_role, training_role, region = _targets(env)
    missing = [
        name
        for name, value in (
            ("ARTIFACTS_BUCKET", bucket),
            ("PIPELINE_ROLE_ARN", pipeline_role),
            ("DEEPAR_ROLE_ARN", training_role),
        )
        if value == ""
    ]
    if missing:
        names = " and ".join(missing)
        print(
            f"The training pipeline was not started because {names} must be set.",
            file=sys.stderr,
        )
        return 1
    count = started_today if started_today is not None else _jobs_started_today(region)
    pipeline = TrainingPipeline(
        dataset_version=args.dataset_version,
        git_sha=args.git_sha,
        configuration=configuration,
        bucket=bucket,
        pipeline_role_arn=pipeline_role,
        training_role_arn=training_role,
        region=region,
        gates=gates,
    )
    active_store = store
    client: PipelineClient | None = None
    if count < gates.max_training_jobs_per_day:
        factory = client_factory if client_factory is not None else open_pipeline_clients
        client, uploader = factory(region)
        if active_store is None:
            active_store = ObjectTrainingRunStore(uploader, bucket)
    try:
        submission = start_pipeline(
            pipeline,
            gates=gates,
            jobs_started_today=count,
            client=client,
            store=active_store,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return _emit(submission)


def _load_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Configuration file {path} was not found.")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Configuration file {path} must be JSON.") from exc
    if not isinstance(loaded, dict) or any(not isinstance(key, str) for key in loaded):
        raise ValueError("Configuration must be a JSON object.")
    return cast(dict[str, object], loaded)


def _targets(environ: Mapping[str, str]) -> tuple[str, str, str, str]:
    bucket = environ.get("ARTIFACTS_BUCKET", "").strip()
    pipeline_role = environ.get("PIPELINE_ROLE_ARN", "").strip()
    training_role = environ.get("DEEPAR_ROLE_ARN", "").strip()
    return bucket, pipeline_role, training_role, _region(environ)


def _region(environ: Mapping[str, str]) -> str:
    region = environ.get("AWS_REGION", "us-east-1").strip()
    return region or "us-east-1"


def _jobs_started_today(region: str) -> int:
    client, _uploader = open_aws_clients(region)
    return client.training_jobs_started_today(day=datetime.now(UTC).date())


def _emit(submission: PipelineSubmission) -> int:
    stream = sys.stdout if submission.exit_code == 0 else sys.stderr
    print(submission.reason, file=stream)
    return submission.exit_code


def _process_environ() -> Mapping[str, str]:
    import os

    return os.environ
