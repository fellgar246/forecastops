"""Training pipeline definition, skip-on-reject, and execution metadata."""

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from forecastops_ml.data.quality import validate_dataset
from forecastops_ml.data.synthetic import DEFAULT_MAX_ROWS, DEFAULT_SEED, generate_dataset
from forecastops_ml.evaluation import EvaluationReport, MetricSummary, SliceMetrics
from forecastops_ml.evaluation.backtest import Backtester
from forecastops_ml.evaluation.folds import MIN_PUBLISHED_FOLDS
from forecastops_ml.pipelines.command import main
from forecastops_ml.pipelines.definition import (
    STEP_NAMES,
    TrainingPipeline,
    executed_step_names,
    render_pipeline,
)
from forecastops_ml.pipelines.execution import (
    BotoPipelineClient,
    MissingPipeline,
    ObjectTrainingRunStore,
    PipelineExecutionResult,
)
from forecastops_ml.pipelines.metadata import (
    MemoryTrainingRunStore,
    store_finished_execution,
)
from forecastops_ml.pipelines.steps import (
    ENTRYPOINTS,
    register_model,
    run_evaluate,
    run_process,
    run_quality_gate,
    run_validate,
    step_bindings,
)
from forecastops_ml.pipelines.steps import (
    main as step_main,
)
from forecastops_ml.promotion import ModelStatus
from forecastops_ml.training.deepar_export import export_deepar_channels
from forecastops_ml.training.deepar_job import (
    CPU_INSTANCE_TYPE,
    DEEPAR_TRAINING_IMAGE,
    RUNTIME_CEILING_MINUTES,
    TrainingGates,
    build_training_job_request,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET_VERSION = "2026-09-21"
GIT_SHA = "abc123"
EXECUTION_ARN = (
    "arn:aws:sagemaker:us-east-1:000000000000:pipeline/forecastops-training/execution/example"
)
PIPELINE_ARN = "arn:aws:sagemaker:us-east-1:000000000000:pipeline/forecastops-training"


def test_rendered_steps_follow_the_fixed_order_and_skip_register_on_reject() -> None:
    rendered = _rendered()
    document = rendered.document

    assert rendered.step_names == STEP_NAMES
    assert executed_step_names(document, gate_passed=True) == STEP_NAMES
    assert executed_step_names(document, gate_passed=False) == STEP_NAMES[:-1]
    assert "Register" not in executed_step_names(document, gate_passed=False)
    gate = _step(document, "QualityGate")
    arguments = gate["Arguments"]
    assert arguments["ElseSteps"] == []
    assert arguments["IfSteps"][0]["Name"] == "Register"
    assert arguments["IfSteps"][0]["Arguments"]["ModelApprovalStatus"] == "PendingManualApproval"
    assert list(rendered.execution_parameters) == ["DatasetVersion", "GitSha", "Configuration"]
    assert rendered.execution_parameters["DatasetVersion"] == DATASET_VERSION
    assert rendered.execution_parameters["GitSha"] == GIT_SHA
    assert json.loads(rendered.execution_parameters["Configuration"])["model_family"] == "deepar"


def test_steps_use_validation_backtest_and_the_quality_gate() -> None:
    rendered = _rendered()
    document = rendered.document

    expected = [ENTRYPOINTS[name] for name in STEP_NAMES]
    assert [step.entrypoint for step in rendered.steps] == expected
    assert step_bindings()["Validate"] is run_validate
    assert step_bindings()["Process"] is run_process
    assert step_bindings()["Train"] is build_training_job_request
    assert step_bindings()["Evaluate"] is run_evaluate
    assert step_bindings()["Quality Gate"] is run_quality_gate
    assert "validate_dataset" in run_validate.__code__.co_names
    assert "export_deepar_channels" in run_process.__code__.co_names
    assert "evaluate" in run_evaluate.__code__.co_names
    assert "build_folds" in Backtester.evaluate.__code__.co_names
    assert "gate" in run_quality_gate.__code__.co_names
    assert validate_dataset.__name__ == "validate_dataset"
    assert export_deepar_channels.__name__ == "export_deepar_channels"
    train = _step(document, "Train")
    resources = train["Arguments"]["ResourceConfig"]
    assert resources["InstanceType"] == CPU_INSTANCE_TYPE
    assert resources["InstanceCount"] == 1
    assert train["Arguments"]["StoppingCondition"]["MaxRuntimeInSeconds"] == (
        RUNTIME_CEILING_MINUTES * 60
    )
    assert train["Arguments"]["AlgorithmSpecification"]["TrainingImage"] == DEEPAR_TRAINING_IMAGE
    serialized = json.dumps(document)
    assert "HyperParameterTuning" not in serialized
    assert "CreateEndpoint" not in serialized
    evaluate = _step(document, "Evaluate")
    environment = evaluate["Arguments"]["Environment"]
    assert environment["FORECASTOPS_FOLDS"] == "forecastops_ml.evaluation.folds.build_folds"
    assert environment["FORECASTOPS_MIN_PUBLISHED_FOLDS"] == str(MIN_PUBLISHED_FOLDS)
    assert "wape" in environment["FORECASTOPS_METRICS"]
    condition = _step(document, "QualityGate")["Arguments"]["Conditions"][0]
    assert condition["RightValue"] == "PENDING_APPROVAL"
    assert condition["LeftValue"]["Std:JsonGet"]["Path"] == "gate_status"

    rejected = run_quality_gate(
        _report(0.20),
        _report(0.15),
        candidate_id="candidate",
        reference_id="seasonal-naive",
    )
    assert rejected.status is ModelStatus.REJECTED
    assert register_model(rejected) is None
    accepted = run_quality_gate(
        _report(0.10),
        _report(0.15),
        candidate_id="candidate",
        reference_id="seasonal-naive",
    )
    assert register_model(accepted) == {"ModelApprovalStatus": "PendingManualApproval"}


def test_finished_execution_stores_the_training_run_metadata() -> None:
    store = MemoryTrainingRunStore()
    metrics = _report(0.10).to_dict()
    locations = {
        "channels": f"s3://forecastops-artifacts/processed/{DATASET_VERSION}/channels/",
        "metrics": f"s3://forecastops-artifacts/evaluations/{DATASET_VERSION}/metrics.json",
        "model": f"s3://forecastops-artifacts/models/{DATASET_VERSION}/",
        "quality_report": (
            f"s3://forecastops-artifacts/processed/{DATASET_VERSION}/quality/quality_report.json"
        ),
    }
    configuration = _rendered().configuration

    record = store_finished_execution(
        git_sha=GIT_SHA,
        dataset_version=DATASET_VERSION,
        configuration=configuration,
        metrics=metrics,
        artifact_uris=locations,
        pipeline_execution_arn=EXECUTION_ARN,
        store=store,
    )

    assert store.records == [record]
    assert record.git_sha == GIT_SHA
    assert record.dataset_version == DATASET_VERSION
    assert record.configuration == configuration
    assert record.configuration["hyperparameter_search"] is False
    assert record.metrics == metrics
    assert record.metrics is not None
    assert record.metrics["fold_count"] == 3
    assert record.artifact_uris == locations
    assert record.artifact_uri == locations["model"]
    assert record.pipeline_execution_arn == EXECUTION_ARN
    assert record.status == "COMPLETED"


def test_finished_execution_rejects_a_missing_metric_document() -> None:
    with pytest.raises(ValueError, match="metric document"):
        store_finished_execution(
            git_sha=GIT_SHA,
            dataset_version=DATASET_VERSION,
            configuration=_rendered().configuration,
            metrics={},
            artifact_uris={
                "channels": "s3://forecastops-artifacts/channels/",
                "metrics": "s3://forecastops-artifacts/metrics.json",
                "model": "s3://forecastops-artifacts/model/",
                "quality_report": "s3://forecastops-artifacts/quality_report.json",
            },
            pipeline_execution_arn=EXECUTION_ARN,
            store=MemoryTrainingRunStore(),
        )


def test_start_command_refuses_when_training_flags_are_off(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        ["--dataset-version", DATASET_VERSION, "--git-sha", GIT_SHA, "--config", "missing.json"],
        environ={"SAGEMAKER_ENABLED": "false"},
        client_factory=_bomb_factory,
        started_today=0,
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "SAGEMAKER_ENABLED is false" in captured.out
    assert "was not started" in captured.out
    assert captured.err == ""


def test_start_command_persists_a_finished_execution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"grain": "week", "prediction_length": 2}\n', encoding="utf-8")
    store = MemoryTrainingRunStore()
    metrics = _report(0.10).to_dict()
    client = StubPipeline(
        PipelineExecutionResult(
            arn=EXECUTION_ARN,
            status="COMPLETED",
            finished=True,
            metrics=metrics,
        )
    )

    code = main(
        ["--dataset-version", DATASET_VERSION, "--git-sha", GIT_SHA, "--config", str(config)],
        environ=_open_environ(),
        client_factory=lambda _region: (client, MemoryUpload()),
        store=store,
        started_today=0,
    )

    captured = capsys.readouterr()
    assert code == 0
    assert EXECUTION_ARN in captured.out
    assert client.started == [
        {
            "DatasetVersion": DATASET_VERSION,
            "GitSha": GIT_SHA,
            "Configuration": client.definitions[0]["Parameters"][2]["DefaultValue"],
        }
    ]
    record = store.records[0]
    assert record.git_sha == GIT_SHA
    assert record.dataset_version == DATASET_VERSION
    assert record.configuration["model_family"] == "deepar"
    assert record.configuration["instance_type"] == CPU_INSTANCE_TYPE
    assert record.metrics == metrics
    assert set(record.artifact_uris) >= {"channels", "metrics", "model", "quality_report"}
    assert record.pipeline_execution_arn == EXECUTION_ARN
    assert record.status == "COMPLETED"


def test_daily_cap_does_not_start_an_execution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")

    code = main(
        ["--dataset-version", DATASET_VERSION, "--git-sha", GIT_SHA, "--config", str(config)],
        environ=_open_environ(),
        client_factory=_bomb_factory,
        started_today=2,
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "daily limit is 2" in captured.err
    assert captured.out == ""


def test_boto_client_creates_then_updates_and_starts() -> None:
    api = FakePipelineApi()
    client = BotoPipelineClient(api)

    created = client.upsert_pipeline(
        name="forecastops-training",
        definition={"Version": "2020-12-01", "Steps": []},
        role_arn="arn:aws:iam::000000000000:role/pipeline",
    )
    api.exists = True
    updated = client.upsert_pipeline(
        name="forecastops-training",
        definition={"Version": "2020-12-01", "Steps": []},
        role_arn="arn:aws:iam::000000000000:role/pipeline",
    )
    started = client.start_execution(
        pipeline_name="forecastops-training",
        parameters={"DatasetVersion": DATASET_VERSION, "GitSha": GIT_SHA},
    )

    assert created == PIPELINE_ARN
    assert updated == PIPELINE_ARN
    assert len(api.created) == 1
    assert len(api.updated) == 1
    assert api.created[0]["PipelineDefinition"] == '{"Version": "2020-12-01", "Steps": []}'
    assert started.arn.endswith("/execution/1")
    assert started.finished is False
    assert api.started[0]["PipelineParameters"] == [
        {"Name": "DatasetVersion", "Value": DATASET_VERSION},
        {"Name": "GitSha", "Value": GIT_SHA},
    ]


def test_object_store_writes_the_training_run_document() -> None:
    uploads = MemoryUpload()
    store = ObjectTrainingRunStore(uploads, "forecastops-artifacts")
    record = store_finished_execution(
        git_sha=GIT_SHA,
        dataset_version=DATASET_VERSION,
        configuration=_rendered().configuration,
        metrics=_report(0.10).to_dict(),
        artifact_uris={
            "channels": "s3://forecastops-artifacts/channels/",
            "metrics": "s3://forecastops-artifacts/metrics.json",
            "model": "s3://forecastops-artifacts/model/",
            "quality_report": "s3://forecastops-artifacts/quality_report.json",
        },
        pipeline_execution_arn=EXECUTION_ARN,
        store=store,
        run_id="run-1",
    )

    key = ("forecastops-artifacts", "training/runs/run-1/training_run.json")
    body = json.loads(uploads.objects[key])
    assert body["git_sha"] == record.git_sha
    assert body["dataset_version"] == DATASET_VERSION
    assert body["metrics"]["fold_count"] == 3
    assert body["artifact_uris"]["model"] == "s3://forecastops-artifacts/model/"


def test_default_infrastructure_keeps_schedules_disabled() -> None:
    infra = ROOT / "infra"
    for name in ("dev", "demo"):
        text = (infra / "environments" / name / "variables.tf").read_text(encoding="utf-8")
        assert re.search(r'variable "enable_schedules"[\s\S]*?default\s*=\s*false', text)
        assert "enable_schedules = true" not in text
    eventbridge = (infra / "modules" / "eventbridge" / "main.tf").read_text(encoding="utf-8")
    assert "for_each = var.enable_schedules ? local.schedules : {}" in eventbridge
    assert "aws_cloudwatch_event_rule" not in eventbridge
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "ml" / "forecastops_ml" / "pipelines").glob("*.py")
    )
    assert "cron(" not in source
    assert "rate(" not in source


def test_validate_and_process_run_on_a_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    generate_dataset(tmp_path, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    missing = step_main(["validate", "--dataset", str(tmp_path / "missing")])
    assert missing == 2

    code = step_main(["validate", "--dataset", str(tmp_path), "--as-of", "2026-09-30"])
    output = tmp_path / "channels"
    process = step_main(
        [
            "process",
            "--dataset",
            str(tmp_path),
            "--output",
            str(output),
            "--grain",
            "week",
            "--prediction-length",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert process == 0
    assert "valid" in captured.out
    assert (tmp_path / "quality_report.json").is_file()
    assert (output / "train" / "train.json").read_text(encoding="utf-8").startswith("{")
    assert (output / "test" / "test.json").read_text(encoding="utf-8").startswith("{")


def test_ci_does_not_start_the_pipeline() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    lowered = workflow.lower()

    assert "pull_request" in workflow
    assert "push:" in workflow
    assert "start-pipeline" not in lowered
    assert "start_pipeline" not in lowered
    assert "startpipelineexecution" not in lowered


def _rendered() -> Any:
    return render_pipeline(
        TrainingPipeline(
            dataset_version=DATASET_VERSION,
            git_sha=GIT_SHA,
            configuration={"grain": "week", "prediction_length": 2},
            bucket="forecastops-artifacts",
            pipeline_role_arn="arn:aws:iam::000000000000:role/pipeline",
            training_role_arn="arn:aws:iam::000000000000:role/training",
            region="us-east-1",
            gates=_open_gates(),
        )
    )


def _step(document: Mapping[str, Any], name: str) -> dict[str, Any]:
    for step in document["Steps"]:
        if step["Name"] == name:
            return step
    raise AssertionError(name)


def _report(wape: float) -> EvaluationReport:
    summary = MetricSummary(
        row_count=4,
        mae=1.0,
        rmse=1.0,
        wape=wape,
        smape=0.1,
        bias=0.0,
        pinball_loss_p10=None,
        pinball_loss_p90=None,
    )
    return EvaluationReport(
        model_family="deepar",
        fold_count=3,
        horizon=1,
        dataset_version=DATASET_VERSION,
        global_metrics=summary,
        by_category=(SliceMetrics(key="grocery", metrics=summary),),
        by_store=(),
        by_horizon_step=(),
        by_demand_quartile=(),
        runtime_ms=1,
        wape_by_horizon=((1, wape),),
        skipped_series=(),
    )


def _open_gates() -> TrainingGates:
    return TrainingGates(
        sagemaker_enabled=True,
        training_enabled=True,
        aws_ml_enabled=True,
        allow_gpu_training=False,
        max_training_runtime_minutes=RUNTIME_CEILING_MINUTES,
        max_training_jobs_per_day=2,
    )


def _open_environ() -> dict[str, str]:
    return {
        "SAGEMAKER_ENABLED": "true",
        "TRAINING_ENABLED": "true",
        "AWS_ML_ENABLED": "true",
        "ALLOW_GPU_TRAINING": "false",
        "MAX_TRAINING_RUNTIME_MINUTES": "45",
        "MAX_TRAINING_JOBS_PER_DAY": "2",
        "ARTIFACTS_BUCKET": "forecastops-artifacts",
        "PIPELINE_ROLE_ARN": "arn:aws:iam::000000000000:role/pipeline",
        "DEEPAR_ROLE_ARN": "arn:aws:iam::000000000000:role/training",
        "AWS_REGION": "us-east-1",
    }


def _bomb_factory(_region: str) -> tuple[Any, Any]:
    raise AssertionError("The pipeline client must not be opened.")


class StubPipeline:
    def __init__(self, execution: PipelineExecutionResult) -> None:
        self.execution = execution
        self.definitions: list[dict[str, Any]] = []
        self.started: list[dict[str, str]] = []

    def upsert_pipeline(
        self,
        *,
        name: str,
        definition: Mapping[str, Any],
        role_arn: str,
    ) -> str:
        self.definitions.append(dict(definition))
        return PIPELINE_ARN

    def start_execution(
        self,
        *,
        pipeline_name: str,
        parameters: Mapping[str, str],
    ) -> PipelineExecutionResult:
        self.started.append(dict(parameters))
        return self.execution


class MemoryUpload:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_bytes(self, bucket: str, key: str, body: bytes) -> None:
        self.objects[(bucket, key)] = body


class FakePipelineApi:
    def __init__(self) -> None:
        self.exists = False
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []
        self.started: list[dict[str, Any]] = []

    def describe_pipeline(self, **kwargs: Any) -> Mapping[str, Any]:
        if not self.exists:
            raise MissingPipeline(kwargs["PipelineName"])
        return {"PipelineArn": PIPELINE_ARN}

    def create_pipeline(self, **kwargs: Any) -> Mapping[str, Any]:
        self.created.append(kwargs)
        return {"PipelineArn": PIPELINE_ARN}

    def update_pipeline(self, **kwargs: Any) -> Mapping[str, Any]:
        self.updated.append(kwargs)
        return {"PipelineArn": PIPELINE_ARN}

    def start_pipeline_execution(self, **kwargs: Any) -> Mapping[str, Any]:
        self.started.append(kwargs)
        return {"PipelineExecutionArn": f"{PIPELINE_ARN}/execution/1"}
