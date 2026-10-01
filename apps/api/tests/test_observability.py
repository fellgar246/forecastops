"""Correlation ids, metric names, and logs that omit dataset bodies."""

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from forecastops_api.artifacts import LocalArtifactStore
from forecastops_api.db import Base
from forecastops_api.logging import configure_logging
from forecastops_api.main import create_app
from forecastops_api.observability import (
    ALARMS,
    METRIC_NAMES,
    CloudWatchPublisher,
    MetricPoint,
    configure_metrics,
    notifications_enabled,
    record_data_quality,
    record_explanation_usage,
    record_explanation_validation_failure,
    record_forecast_job,
    record_metric,
    record_model_metrics,
    reset_metrics,
    snapshot,
)
from forecastops_api.settings import ExecutionMode, Settings, get_settings
from forecastops_ml.evaluation.backtest import EvaluationReport, MetricSummary, SliceMetrics

SENTINEL_UNITS = "999001"


@pytest.fixture(autouse=True)
def _isolated_metrics() -> None:
    reset_metrics()


def test_request_log_includes_the_correlation_id_and_omits_the_dataset(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    configure_logging()
    client = _client(tmp_path)
    body = f"date,store_id,sku_id,units_sold\n2026-01-01,store-1,sku-1,{SENTINEL_UNITS}\n"
    correlation_id = "request-19"

    with caplog.at_level(logging.INFO):
        response = client.post(
            "/datasets/uploads",
            files={"file": ("demand.csv", body.encode(), "text/csv")},
            headers={"X-Correlation-Id": correlation_id},
        )

    assert response.status_code == 201
    assert response.headers["x-correlation-id"] == correlation_id
    request_logs = _request_logs(caplog, correlation_id)
    assert request_logs
    assert all(item["correlation_id"] == correlation_id for item in request_logs)
    assert any(item["event"] == "http.request" for item in request_logs)
    captured = "\n".join(record.getMessage() for record in caplog.records)
    assert SENTINEL_UNITS not in captured
    assert "units_sold" not in captured


def test_missing_correlation_id_is_created_and_returned(tmp_path: Path) -> None:
    client = _client(tmp_path)

    first = client.get("/health")
    second = client.get("/health", headers={"X-Correlation-Id": "trace-2"})

    assert first.status_code == 200
    assert first.headers["x-correlation-id"]
    assert first.headers["x-correlation-id"] != second.headers["x-correlation-id"]
    assert second.headers["x-correlation-id"] == "trace-2"


def test_metrics_endpoint_lists_every_catalogued_name(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    names = {item["name"] for item in response.json()["metrics"]}
    assert names == METRIC_NAMES
    counts = {
        item["name"]: item["value"]
        for item in response.json()["metrics"]
        if item["dimensions"] == {}
    }
    assert counts["api_request_count"] >= 1
    assert counts["api_latency_ms"] > 0


def test_client_errors_increment_the_error_count(tmp_path: Path) -> None:
    client = _client(tmp_path)

    missing = client.get("/missing-route")
    metrics = client.get("/metrics")

    assert missing.status_code == 404
    values = {
        item["name"]: item["value"]
        for item in metrics.json()["metrics"]
        if item["name"] == "api_error_count" and item["dimensions"] == {}
    }
    assert values["api_error_count"] >= 1


def test_owned_metrics_use_the_catalogued_names() -> None:
    summary = MetricSummary(
        row_count=2,
        mae=1.5,
        rmse=2.0,
        wape=0.2,
        smape=0.1,
        bias=-0.05,
        pinball_loss_p10=None,
        pinball_loss_p90=None,
    )
    report = EvaluationReport(
        model_family="seasonal_naive",
        fold_count=3,
        horizon=7,
        dataset_version="v1",
        global_metrics=summary,
        by_category=(SliceMetrics(key="beverages", metrics=summary),),
        by_store=(),
        by_horizon_step=(),
        by_demand_quartile=(),
        runtime_ms=10,
        wape_by_horizon=((1, 0.2),),
        skipped_series=(),
    )
    record_model_metrics(
        model_version="3",
        forecast_horizon=7,
        report=report,
        p90_coverage=0.91,
    )
    record_data_quality(
        latest_data_age_hours=48,
        missing_value_rate=0.0,
        duplicate_rate=0.01,
        stockout_rate=0.2,
        unknown_category_count=1,
    )
    record_forecast_job(outcome="failed", latency_seconds=1.25)
    record_explanation_usage(
        input_tokens=20,
        output_tokens=8,
        latency_ms=40,
        cache_hit=False,
        provider_failed=True,
    )
    record_explanation_validation_failure()

    recorded = {(point.name, tuple(sorted(point.dimensions.items()))) for point in snapshot()}
    beverages = (
        "wape",
        (("category", "beverages"), ("forecast_horizon", "7"), ("model_version", "3")),
    )
    overall = (
        "p90_coverage",
        (("category", "all"), ("forecast_horizon", "7"), ("model_version", "3")),
    )
    assert beverages in recorded
    assert overall in recorded
    assert ("latest_data_age_hours", ()) in recorded
    assert ("forecast_job_count", (("outcome", "failed"),)) in recorded
    assert ("bedrock_calls", ()) in recorded
    assert ("bedrock_failures", ()) in recorded
    assert ("explanation_validation_failures", ()) in recorded
    assert {point.name for point in snapshot()} == METRIC_NAMES


def test_cloudwatch_publisher_sends_the_metric_name() -> None:
    class FakeCloudWatch:
        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def put_metric_data(
            self,
            *,
            Namespace: str,
            MetricData: list[dict[str, object]],
        ) -> dict[str, str]:
            self.payloads.append({"Namespace": Namespace, "MetricData": MetricData})
            return {}

    client = FakeCloudWatch()
    CloudWatchPublisher(client).publish(
        MetricPoint(name="training_failures", value=1, unit="Count", dimensions={})
    )

    assert client.payloads[0]["Namespace"] == "ForecastOps"
    metric_data = client.payloads[0]["MetricData"]
    assert isinstance(metric_data, list)
    assert metric_data[0]["MetricName"] == "training_failures"


def test_local_mode_does_not_open_a_cloudwatch_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_client(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("local mode must not open a CloudWatch client")

    monkeypatch.setattr("boto3.client", refuse_client)
    for key, value in _example_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("EXECUTION_MODE", "local")
    get_settings.cache_clear()

    configure_metrics(Settings(_env_file=None))
    record_metric("api_request_count", 1)

    assert Settings(_env_file=None).execution_mode is ExecutionMode.LOCAL


def test_alarm_notifications_stay_off_until_demo_requests_them() -> None:
    assert [alarm.name for alarm in ALARMS] == [
        "training-failure",
        "repeated-forecast-failure",
        "stale-dataset",
        "high-forecast-error",
        "explanation-error-spike",
    ]
    assert notifications_enabled("dev", True) is False
    assert notifications_enabled("demo", False) is False
    assert notifications_enabled("demo", True) is True


def _client(tmp_path: Path) -> TestClient:
    database = tmp_path / "api.db"
    get_settings.cache_clear()
    application = create_app()
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite+pysqlite:///{database}")
    application.state.engine = engine
    application.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    application.state.artifact_dir = tmp_path / "artifacts"
    application.state.artifact_store = LocalArtifactStore(application.state.artifact_dir)
    Base.metadata.create_all(engine)
    return TestClient(application)


def _request_logs(
    caplog: pytest.LogCaptureFixture,
    correlation_id: str,
) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for record in caplog.records:
        try:
            payload = json.loads(record.getMessage())
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("correlation_id") == correlation_id:
            found.append(payload)
    return found


def _example_env() -> dict[str, str]:
    from envfile import ROOT, parse_env_file

    return parse_env_file(ROOT / ".env.example")
