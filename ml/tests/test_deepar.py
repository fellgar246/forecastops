"""DeepAR channel export, quantile checks, and the manual training command."""

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from forecastops_ml.data import load_dataset
from forecastops_ml.data.synthetic import DEFAULT_MAX_ROWS, DEFAULT_SEED, generate_dataset
from forecastops_ml.evaluation.backtest import make_series_id
from forecastops_ml.features.week import aggregate_store_sku_week
from forecastops_ml.training.deepar_command import main
from forecastops_ml.training.deepar_export import DeepARRecord, export_deepar_channels
from forecastops_ml.training.deepar_forecast import (
    FakeDeepARRunner,
    load_fixture_forecast,
    map_deepar_predictions,
)
from forecastops_ml.training.deepar_job import (
    CPU_INSTANCE_TYPE,
    BotoTrainingJobClient,
    TrainingGates,
    block_reason,
    build_training_job_request,
    require_cpu_instance,
    submit_deepar_training,
)

ROOT = Path(__file__).resolve().parents[2]
START = date(2026, 8, 3)
NOW = datetime(2026, 9, 27, 22, 0, tzinfo=UTC)


class MemoryJobs:
    """In-memory training client."""

    def __init__(self, started: int = 0) -> None:
        self.started = started
        self.requests: list[Mapping[str, object]] = []
        self.listed = 0

    def training_jobs_started_today(self, *, day: date) -> int:
        self.listed += 1
        self.day = day
        return self.started

    def create_training_job(self, request: Mapping[str, object]) -> str:
        self.requests.append(request)
        return "arn:aws:sagemaker:us-east-1:000000000000:training-job/example"


class MemoryUpload:
    """In-memory object uploader."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_bytes(self, bucket: str, key: str, body: bytes) -> None:
        self.objects[(bucket, key)] = body


class BombJobs:
    """Client that fails if the training API is touched."""

    def training_jobs_started_today(self, *, day: date) -> int:
        raise AssertionError("training API was called")

    def create_training_job(self, request: Mapping[str, object]) -> str:
        raise AssertionError("training API was called")


class PagedApi:
    """Two-page training list and one create call."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def list_training_jobs(self, **kwargs: Any) -> dict[str, Any]:
        if "NextToken" not in kwargs:
            return {"TrainingJobSummaries": [{"TrainingJobName": "a"}], "NextToken": "next"}
        return {"TrainingJobSummaries": [{"TrainingJobName": "b"}, {"TrainingJobName": "c"}]}

    def create_training_job(self, **kwargs: Any) -> dict[str, str]:
        self.created.append(kwargs)
        return {"TrainingJobArn": "arn:example"}


def test_test_dataset_export_locks_channels_categories_and_features(tmp_path: Path) -> None:
    generate_dataset(tmp_path, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    daily, _dimensions = load_dataset(tmp_path)
    weekly = aggregate_store_sku_week(daily)
    channels = export_deepar_channels(daily, grain="week", prediction_length=2)

    assert channels.frequency == "1W"
    assert channels.grain == "week"
    assert len(channels.train) >= 2
    assert len(channels.train) == len(channels.test)
    assert channels.train[0].start.weekday() == 0

    expected = _weekly_features(weekly)
    store_codes = _codes(weekly, "store_id")
    category_codes = _codes(weekly, "category_id")
    for trained, held_out in zip(channels.train, channels.test, strict=True):
        assert trained.series_id == held_out.series_id
        assert trained.start == held_out.start
        assert trained.target_end < held_out.target_end
        assert len(held_out.target) <= 12
        assert len(trained.target) + channels.prediction_length == len(held_out.target)
        history = expected[trained.series_id]
        assert list(trained.target) == history["target"][:-2]
        assert list(held_out.target) == history["target"]
        assert list(trained.dynamic_feat[0]) == history["promotion"]
        assert list(trained.dynamic_feat[1]) == history["holiday"]
        assert len(trained.dynamic_feat[0]) == len(trained.target) + 2
        assert list(held_out.dynamic_feat[0]) == history["promotion"]
        assert list(held_out.dynamic_feat[1]) == history["holiday"]
        assert trained.cat == (
            store_codes[history["store_id"]],
            category_codes[history["category_id"]],
        )
        assert held_out.cat == trained.cat

    parsed = _json_lines(channels.jsonl("train"))
    assert len(parsed) == len(channels.train)
    first = parsed[0]
    assert set(first) == {"start", "target", "cat", "dynamic_feat"}
    assert first["start"].endswith(" 00:00:00")
    assert len(first["cat"]) == 2
    assert len(first["dynamic_feat"]) == 2
    assert len(first["dynamic_feat"][0]) == len(first["target"]) + 2


def test_day_grain_uses_daily_frequency_and_keeps_train_before_test() -> None:
    channels = export_deepar_channels(_daily_frame(), grain="day", prediction_length=2)

    assert channels.frequency == "1D"
    trained = channels.train[0]
    held_out = channels.test[0]
    assert trained.target_end == START + timedelta(days=5)
    assert held_out.target_end == START + timedelta(days=7)
    assert trained.target_end < held_out.target_end
    assert trained.cat == (0, 0)
    assert trained.dynamic_feat[0][-2:] == (1.0, 0.0)
    assert trained.dynamic_feat[1][-2:] == (1.0, 0.0)


def test_horizon_plan_extends_test_dynamic_features() -> None:
    frame = _daily_frame()
    plan_rows = []
    for store, category in (("store-01", "beverages"), ("store-02", "snacks")):
        for offset in (8, 9):
            plan_rows.append(
                {
                    "date": START + timedelta(days=offset),
                    "store_id": store,
                    "sku_id": "sku-001",
                    "category_id": category,
                    "promotion": True,
                    "holiday": False,
                }
            )
    channels = export_deepar_channels(
        frame,
        grain="day",
        prediction_length=2,
        horizon_plan=pa.Table.from_pylist(plan_rows),
    )

    held_out = channels.test[0]
    trained = channels.train[0]
    assert len(trained.dynamic_feat[0]) == len(trained.target) + 2
    assert len(held_out.dynamic_feat[0]) == len(held_out.target) + 2
    assert held_out.dynamic_feat[0][-2:] == (1.0, 1.0)
    assert held_out.dynamic_feat[1][-2:] == (0.0, 0.0)
    assert len(held_out.target) == 8


def test_single_series_and_gaps_are_rejected() -> None:
    rows = _rows_for(("store-01", "beverages"), days=8)
    with pytest.raises(ValueError, match="at least two series"):
        export_deepar_channels(pa.Table.from_pylist(rows), grain="day", prediction_length=2)

    gapped = _rows_for(("store-01", "beverages"), ("store-02", "snacks"), days=8)
    gapped = [row for row in gapped if row["date"] != START + timedelta(days=3)]
    with pytest.raises(ValueError, match="skips from"):
        export_deepar_channels(pa.Table.from_pylist(gapped), grain="day", prediction_length=2)


def test_fixture_quantiles_are_finite_and_ordered() -> None:
    points = load_fixture_forecast()

    assert len(points) == 4
    assert {point.series_id for point in points} == {"store-01|sku-001", "store-02|sku-001"}
    assert {point.date for point in points} == {date(2026, 9, 7), date(2026, 9, 14)}
    for point in points:
        assert point.p10 <= point.p50 <= point.p90
        assert point.actual is None


def test_unordered_and_non_finite_quantiles_are_rejected() -> None:
    record = DeepARRecord(
        series_id="store-01|sku-001",
        start=START,
        target=(1.0, 2.0),
        cat=(0, 0),
        dynamic_feat=((0.0, 0.0), (0.0, 0.0)),
        target_end=START + timedelta(days=1),
    )
    with pytest.raises(ValueError, match="unordered quantiles"):
        map_deepar_predictions(
            [record],
            [{"quantiles": {"0.1": [3.0], "0.5": [1.0], "0.9": [4.0]}}],
            grain="day",
            prediction_length=1,
        )
    with pytest.raises(ValueError, match="non-finite"):
        map_deepar_predictions(
            [record],
            [{"quantiles": {"0.1": [float("nan")], "0.5": [1.0], "0.9": [2.0]}}],
            grain="day",
            prediction_length=1,
        )


def test_fake_runner_writes_training_and_model_artifacts(tmp_path: Path) -> None:
    points = FakeDeepARRunner().write_artifacts(tmp_path)

    metrics = json.loads((tmp_path / "training" / "deepar-fixture" / "metrics.json").read_text())
    forecast = json.loads((tmp_path / "models" / "deepar-fixture" / "forecast.json").read_text())
    assert metrics["source"] == "fixture"
    assert metrics["quantiles"] == [0.1, 0.5, 0.9]
    assert len(forecast["points"]) == len(points)
    assert (
        forecast["points"][0]["p10"] <= forecast["points"][0]["p50"] <= forecast["points"][0]["p90"]
    )


def test_closed_flags_do_not_call_the_training_client() -> None:
    submission = submit_deepar_training(
        export_deepar_channels(_daily_frame(), grain="day"),
        gates=TrainingGates(False, True, True, False, 45, 2),
        bucket="forecastops-artifacts",
        role_arn="arn:aws:iam::000000000000:role/training",
        region="us-east-1",
        client=BombJobs(),
        uploader=MemoryUpload(),
        now=NOW,
    )

    assert submission.submitted is False
    assert submission.exit_code == 0
    assert "SAGEMAKER_ENABLED is false" in submission.reason


def test_command_with_default_flags_does_not_construct_a_client(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        ["--dataset", "missing"],
        environ={
            "SAGEMAKER_ENABLED": "false",
            "TRAINING_ENABLED": "true",
            "AWS_ML_ENABLED": "false",
            "ALLOW_GPU_TRAINING": "false",
        },
        client_factory=_bomb_factory,
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "was not submitted" in captured.out
    assert "SAGEMAKER_ENABLED is false" in captured.out
    assert "AWS_ML_ENABLED is false" in captured.out
    assert captured.err == ""


def test_gpu_runtime_and_daily_ceilings_block_submission() -> None:
    gpu = TrainingGates(True, True, True, True, 45, 2)
    assert "ALLOW_GPU_TRAINING is true" in (block_reason(gpu) or "")
    long_run = TrainingGates(True, True, True, False, 90, 2)
    assert "above 45" in (block_reason(long_run) or "")
    with pytest.raises(ValueError, match="CPU"):
        require_cpu_instance("ml.p3.2xlarge")
    with pytest.raises(ValueError, match="CPU"):
        require_cpu_instance("ml.g4dn.xlarge")


def test_job_request_is_one_cpu_job_without_hyperparameter_search() -> None:
    channels = export_deepar_channels(_daily_frame(), grain="day")
    gates = _open_gates()
    request = build_training_job_request(
        channels,
        gates=gates,
        bucket="forecastops-artifacts",
        role_arn="arn:aws:iam::000000000000:role/training",
        region="us-east-1",
        job_name="forecastops-deepar-20260927220000",
    )

    encoded = json.dumps(request)
    resource = request["ResourceConfig"]
    assert isinstance(resource, dict)
    assert resource["InstanceType"] == CPU_INSTANCE_TYPE
    assert resource["InstanceCount"] == 1
    stopping = request["StoppingCondition"]
    assert isinstance(stopping, dict)
    assert stopping["MaxRuntimeInSeconds"] == 45 * 60
    hyperparameters = request["HyperParameters"]
    assert isinstance(hyperparameters, dict)
    assert all(isinstance(value, str) for value in hyperparameters.values())
    assert hyperparameters["test_quantiles"] == "[0.1, 0.5, 0.9]"
    assert hyperparameters["time_freq"] == "1D"
    assert hyperparameters["num_dynamic_feat"] == "2"
    assert "HyperParameterRanges" not in encoded
    assert "EndpointConfigName" not in encoded
    assert "CreateEndpoint" not in encoded
    output = request["OutputDataConfig"]
    assert isinstance(output, dict)
    assert (
        output["S3OutputPath"]
        == "s3://forecastops-artifacts/models/forecastops-deepar-20260927220000/"
    )
    inputs = request["InputDataConfig"]
    assert isinstance(inputs, list)
    train_uri = inputs[0]["DataSource"]["S3DataSource"]["S3Uri"]
    assert train_uri.startswith("s3://forecastops-artifacts/training/")
    assert train_uri.endswith("/train/")


def test_daily_cap_lists_jobs_and_does_not_create_one() -> None:
    jobs = MemoryJobs(started=2)
    uploads = MemoryUpload()
    submission = submit_deepar_training(
        export_deepar_channels(_daily_frame(), grain="day"),
        gates=_open_gates(),
        bucket="forecastops-artifacts",
        role_arn="arn:aws:iam::000000000000:role/training",
        region="us-east-1",
        client=jobs,
        uploader=uploads,
        now=NOW,
    )

    assert submission.submitted is False
    assert submission.exit_code == 1
    assert "daily limit is 2" in submission.reason
    assert jobs.listed == 1
    assert jobs.requests == []
    assert uploads.objects == {}


def test_open_command_uploads_channels_and_submits_one_job(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    generate_dataset(tmp_path, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    jobs = MemoryJobs()
    uploads = MemoryUpload()
    code = main(
        ["--dataset", str(tmp_path), "--grain", "week", "--prediction-length", "2"],
        environ=_open_environ(),
        client_factory=lambda _region: (jobs, uploads),
        now=NOW,
    )

    captured = capsys.readouterr()
    assert code == 0
    assert jobs.requests
    assert "Submitted DeepAR training job forecastops-deepar-20260927220000" in captured.out
    train_key = (
        "forecastops-artifacts",
        "training/forecastops-deepar-20260927220000/train/train.json",
    )
    limits = json.loads(
        uploads.objects[
            ("forecastops-artifacts", "training/forecastops-deepar-20260927220000/limits.json")
        ].decode()
    )
    assert train_key in uploads.objects
    assert uploads.objects[train_key].decode().startswith("{")
    assert limits["hyperparameter_search"] is False
    assert limits["instance_type"] == "ml.c5.xlarge"
    assert limits["quantiles"] == [0.1, 0.5, 0.9]
    request = jobs.requests[0]
    assert "EndpointConfigName" not in json.dumps(request)


def test_boto_client_counts_every_page_and_returns_the_job_arn() -> None:
    api = PagedApi()
    client = BotoTrainingJobClient(api)

    assert client.training_jobs_started_today(day=date(2026, 9, 27)) == 3
    arn = client.create_training_job({"TrainingJobName": "forecastops-deepar-example"})
    assert arn == "arn:example"
    assert api.created[0]["TrainingJobName"] == "forecastops-deepar-example"


def test_unknown_region_does_not_call_the_training_client() -> None:
    submission = submit_deepar_training(
        export_deepar_channels(_daily_frame(), grain="day"),
        gates=_open_gates(),
        bucket="forecastops-artifacts",
        role_arn="arn:aws:iam::000000000000:role/training",
        region="eu-west-1",
        client=BombJobs(),
        uploader=MemoryUpload(),
        now=NOW,
    )

    assert submission.submitted is False
    assert "us-east-1" in submission.reason


def test_ci_workflow_does_not_submit_a_training_job() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    lowered = workflow.lower()

    assert "pull_request" in workflow
    assert "train-deepar" not in lowered
    assert "train_deepar" not in lowered
    assert "createtrainingjob" not in lowered
    assert "sagemaker_enabled=true" not in lowered


def test_deepar_modules_do_not_create_an_endpoint() -> None:
    root = ROOT / "ml" / "forecastops_ml" / "training"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("deepar*.py"))

    assert "create_endpoint" not in text
    assert "CreateEndpoint" not in text
    assert "HyperParameterTuning" not in text


def _open_gates() -> TrainingGates:
    return TrainingGates(
        sagemaker_enabled=True,
        training_enabled=True,
        aws_ml_enabled=True,
        allow_gpu_training=False,
        max_training_runtime_minutes=45,
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
        "DEEPAR_ROLE_ARN": "arn:aws:iam::000000000000:role/forecastops-demo-training",
        "AWS_REGION": "us-east-1",
    }


def _bomb_factory(_region: str) -> tuple[BombJobs, MemoryUpload]:
    raise AssertionError("training client was constructed")


def _daily_frame() -> pa.Table:
    rows = _rows_for(("store-01", "beverages"), ("store-02", "snacks"), days=8)
    return pa.Table.from_pylist(rows)


def _rows_for(*series: tuple[str, str], days: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for store_id, category_id in series:
        for offset in range(days):
            rows.append(
                {
                    "date": START + timedelta(days=offset),
                    "store_id": store_id,
                    "sku_id": "sku-001",
                    "category_id": category_id,
                    "units_sold": 10 + offset,
                    "price": 2.0,
                    "promotion": offset % 2 == 0,
                    "holiday": offset % 3 == 0,
                }
            )
    return rows


def _weekly_features(frame: pa.Table) -> dict[str, dict[str, object]]:
    dates = frame.column("date").to_pylist()
    stores = frame.column("store_id").to_pylist()
    skus = frame.column("sku_id").to_pylist()
    categories = frame.column("category_id").to_pylist()
    units = frame.column("units_sold").to_pylist()
    promotions = frame.column("promotion").to_pylist()
    holidays = frame.column("holiday").to_pylist()
    grouped: dict[str, list[int]] = {}
    for index, store_id in enumerate(stores):
        series_id = make_series_id(str(store_id), str(skus[index]))
        grouped.setdefault(series_id, []).append(index)
    features: dict[str, dict[str, object]] = {}
    for series_id, indexes in grouped.items():
        ordered = sorted(indexes, key=lambda index: dates[index])
        features[series_id] = {
            "store_id": str(stores[ordered[0]]),
            "category_id": str(categories[ordered[0]]),
            "target": [float(units[index]) for index in ordered],
            "promotion": [1.0 if promotions[index] else 0.0 for index in ordered],
            "holiday": [1.0 if holidays[index] else 0.0 for index in ordered],
        }
    return features


def _codes(frame: pa.Table, column: str) -> dict[str, int]:
    names = sorted({str(value) for value in frame.column(column).to_pylist()})
    return {name: index for index, name in enumerate(names)}


def _json_lines(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line]
