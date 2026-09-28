"""Command line entry for one manual DeepAR training job."""

import argparse
import sys
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

from forecastops_ml.data import load_dataset
from forecastops_ml.training.deepar_export import (
    DEFAULT_GRAIN,
    DEFAULT_PREDICTION_LENGTH,
    export_deepar_channels,
)
from forecastops_ml.training.deepar_job import (
    ObjectUploader,
    TrainingJobClient,
    block_reason,
    gates_from_environ,
    open_aws_clients,
    submit_deepar_training,
)


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    client_factory: Callable[[str], tuple[TrainingJobClient, ObjectUploader]] | None = None,
    now: datetime | None = None,
) -> int:
    """Submit one DeepAR job, or stop when a training flag is closed."""

    parser = argparse.ArgumentParser(description="Submit one manual DeepAR training job.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Directory that contains the retail observation tables.",
    )
    parser.add_argument(
        "--grain",
        choices=("week", "day"),
        default=DEFAULT_GRAIN,
        help="Series grain. Week is the demo default.",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=DEFAULT_PREDICTION_LENGTH,
        help="Forecast horizon in periods of the chosen grain.",
    )
    args = parser.parse_args(argv)
    env = environ if environ is not None else _process_environ()
    try:
        gates = gates_from_environ(env)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    reason = block_reason(gates)
    if reason is not None:
        print(reason)
        return 0
    if type(args.prediction_length) is not int or args.prediction_length < 1:
        print("prediction length must be a positive integer.", file=sys.stderr)
        return 1
    bucket = env.get("ARTIFACTS_BUCKET", "").strip()
    role_arn = env.get("DEEPAR_ROLE_ARN", "").strip()
    region = env.get("AWS_REGION", "us-east-1").strip() or "us-east-1"
    missing = [
        name
        for name, value in (("ARTIFACTS_BUCKET", bucket), ("DEEPAR_ROLE_ARN", role_arn))
        if value == ""
    ]
    if missing:
        names = " and ".join(missing)
        print(f"DeepAR training was not submitted because {names} must be set.", file=sys.stderr)
        return 1
    try:
        frame, _dimensions = load_dataset(args.dataset)
        channels = export_deepar_channels(
            frame,
            grain=args.grain,
            prediction_length=args.prediction_length,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    factory = client_factory if client_factory is not None else open_aws_clients
    client, uploader = factory(region)
    submission = submit_deepar_training(
        channels,
        gates=gates,
        bucket=bucket,
        role_arn=role_arn,
        region=region,
        client=client,
        uploader=uploader,
        now=now,
    )
    stream = sys.stdout if submission.exit_code == 0 else sys.stderr
    print(submission.reason, file=stream)
    return submission.exit_code


def _process_environ() -> Mapping[str, str]:
    import os

    return os.environ
