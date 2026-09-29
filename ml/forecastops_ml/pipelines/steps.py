"""Step functions the training pipeline runs, in order.

Validate checks the dataset. Process writes DeepAR channels. Train uses the
manual DeepAR job settings. Evaluate scores rolling-origin folds. The quality
gate decides whether Register runs. A rejection skips registration.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pyarrow as pa

from forecastops_ml.data.profile import load_dataset
from forecastops_ml.data.quality import (
    DatasetDimensions,
    QualityReport,
    ValidationConfig,
    validate_dataset,
)
from forecastops_ml.evaluation.backtest import Backtester, EvaluationReport, Forecaster
from forecastops_ml.promotion.gate import PromotionDecision, gate
from forecastops_ml.promotion.models import ModelStatus, RegistryStatus
from forecastops_ml.training.deepar_export import (
    DEFAULT_GRAIN,
    DEFAULT_PREDICTION_LENGTH,
    DeepARChannels,
    export_deepar_channels,
)
from forecastops_ml.training.deepar_job import build_training_job_request

GATE_PROPERTY = "gate_status"
GATE_PASS_STATUS = ModelStatus.PENDING_APPROVAL.value
REGISTER_APPROVAL = RegistryStatus.PENDING_MANUAL_APPROVAL.value

ENTRYPOINTS: dict[str, str] = {
    "Validate": "forecastops_ml.pipelines.steps.run_validate",
    "Process": "forecastops_ml.pipelines.steps.run_process",
    "Train": "forecastops_ml.training.deepar_job.build_training_job_request",
    "Evaluate": "forecastops_ml.pipelines.steps.run_evaluate",
    "Quality Gate": "forecastops_ml.pipelines.steps.run_quality_gate",
    "Register": "forecastops_ml.pipelines.steps.register_model",
}


def run_validate(
    frame: pa.Table,
    dimensions: DatasetDimensions,
    config: ValidationConfig,
    *,
    report_dir: Path | None = None,
) -> QualityReport:
    """Validate ``frame`` and reject a dataset that is not valid."""

    report = validate_dataset(frame, dimensions, config, report_dir=report_dir)
    if report.status != "valid":
        codes = ", ".join(finding.code for finding in report.blocking)
        raise ValueError(f"Dataset validation failed: {codes}.")
    return report


def run_process(
    frame: pa.Table,
    *,
    grain: str = DEFAULT_GRAIN,
    prediction_length: int = DEFAULT_PREDICTION_LENGTH,
) -> DeepARChannels:
    """Build the DeepAR train and test channels for ``frame``."""

    return export_deepar_channels(frame, grain=grain, prediction_length=prediction_length)


def run_evaluate(
    forecaster: Forecaster,
    frame: pa.Table,
    folds: int,
    horizon: int,
    *,
    dataset_version: str,
) -> EvaluationReport:
    """Score ``forecaster`` on rolling-origin folds.

    A published score uses at least three folds. The returned report is the
    metric document the quality gate reads.
    """

    return Backtester().evaluate(
        forecaster,
        frame,
        folds,
        horizon,
        dataset_version=dataset_version,
    )


def run_quality_gate(
    candidate: EvaluationReport,
    reference: EvaluationReport,
    *,
    candidate_id: str,
    reference_id: str,
    p90_coverage: float | None = None,
) -> PromotionDecision:
    """Apply the promotion gate. The result is pending approval or rejected."""

    return gate(
        candidate,
        reference,
        candidate_id=candidate_id,
        reference_id=reference_id,
        p90_coverage=p90_coverage,
    )


def evaluation_property(
    report: EvaluationReport,
    decision: PromotionDecision,
) -> dict[str, object]:
    """Return the evaluation document the quality-gate condition reads.

    ``gate_status`` is ``PENDING_APPROVAL`` or ``REJECTED``. Register runs
    only for pending approval.
    """

    document = report.to_dict()
    document[GATE_PROPERTY] = decision.status.value
    return document


def register_model(decision: PromotionDecision) -> dict[str, str] | None:
    """Return the registry request when the gate is waiting for a person.

    A rejected candidate returns none, so Register does not run. The request
    never approves the model.
    """

    if decision.status is ModelStatus.REJECTED:
        return None
    if decision.status is not ModelStatus.PENDING_APPROVAL:
        raise ValueError("Register runs only after the quality gate.")
    return {"ModelApprovalStatus": REGISTER_APPROVAL}


def step_bindings() -> dict[str, object]:
    """Return the function each pipeline step runs."""

    return {
        "Validate": run_validate,
        "Process": run_process,
        "Train": build_training_job_request,
        "Evaluate": run_evaluate,
        "Quality Gate": run_quality_gate,
        "Register": register_model,
    }


def main(argv: list[str] | None = None) -> int:
    """Run validate or process against a dataset directory.

    Evaluate and the quality gate need the trained model and the evaluation
    report, so this entry point does not score or register a candidate.
    """

    parser = argparse.ArgumentParser(description="Run one training pipeline step.")
    parser.add_argument("step", choices=("validate", "process", "evaluate", "quality-gate"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/opt/ml/processing/input/dataset"),
        help="Dataset directory mounted for the step.",
    )
    parser.add_argument("--output", type=Path, help="Directory for DeepAR channel files.")
    parser.add_argument("--as-of", help="Validation date, as YYYY-MM-DD.")
    parser.add_argument("--grain", choices=("week", "day"), default=DEFAULT_GRAIN)
    parser.add_argument("--prediction-length", type=int, default=DEFAULT_PREDICTION_LENGTH)
    args = parser.parse_args(argv)
    try:
        if args.step == "validate":
            return _validate_cli(args.dataset, args.as_of)
        if args.step == "process":
            return _process_cli(
                args.dataset,
                args.output,
                args.grain,
                args.prediction_length,
            )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        "Evaluate and the quality gate run from the trained model and the evaluation report.",
        file=sys.stderr,
    )
    return 1


def _validate_cli(dataset: Path, as_of: str | None) -> int:
    if not dataset.is_dir():
        print(f"Dataset path {dataset} was not found.", file=sys.stderr)
        return 2
    moment = date.fromisoformat(as_of) if as_of else date.today()
    frame, dimensions = load_dataset(dataset)
    report = run_validate(frame, dimensions, ValidationConfig(as_of=moment), report_dir=dataset)
    print(report.status)
    return 0


def _process_cli(
    dataset: Path,
    output: Path | None,
    grain: str,
    prediction_length: int,
) -> int:
    if not dataset.is_dir():
        print(f"Dataset path {dataset} was not found.", file=sys.stderr)
        return 2
    if type(prediction_length) is not int or prediction_length < 1:
        raise ValueError("prediction length must be a positive integer.")
    frame, _dimensions = load_dataset(dataset)
    channels = run_process(frame, grain=grain, prediction_length=prediction_length)
    destination = output if output is not None else Path("/opt/ml/processing/output/channels")
    (destination / "train").mkdir(parents=True, exist_ok=True)
    (destination / "test").mkdir(parents=True, exist_ok=True)
    (destination / "train" / "train.json").write_text(channels.jsonl("train"), encoding="utf-8")
    (destination / "test" / "test.json").write_text(channels.jsonl("test"), encoding="utf-8")
    print(f"Wrote DeepAR channels for {len(channels.train)} series.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
