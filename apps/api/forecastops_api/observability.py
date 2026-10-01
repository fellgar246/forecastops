"""Correlation ids, process metrics, and operational alarm definitions.

Local mode records the metric names and writes them to the log. ``GET /metrics``
returns the same names. Cloud mode also publishes them to CloudWatch. Alarm
notifications stay off until the demo environment turns them on. Logs carry
ids, statuses, artifact URIs, and durations. They do not carry dataset rows.
"""

import math
import re
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from forecastops_api.settings import ExecutionMode, Settings
from forecastops_ml.evaluation.backtest import EvaluationReport, MetricSummary

CORRELATION_HEADER = "X-Correlation-Id"
METRIC_NAMESPACE = "ForecastOps"
_CORRELATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_STALE_DATASET_HOURS = 30 * 24

METRIC_NAMES: frozenset[str] = frozenset(
    {
        "api_request_count",
        "api_error_count",
        "api_latency_ms",
        "training_job_count",
        "training_duration_seconds",
        "training_failures",
        "forecast_job_count",
        "forecast_latency_seconds",
        "wape",
        "smape",
        "mae",
        "rmse",
        "bias",
        "p90_coverage",
        "bedrock_calls",
        "bedrock_input_tokens",
        "bedrock_output_tokens",
        "bedrock_latency",
        "bedrock_failures",
        "explanation_validation_failures",
        "latest_data_age_hours",
        "missing_value_rate",
        "duplicate_rate",
        "stockout_rate",
        "unknown_category_count",
    }
)

_COUNTERS = frozenset(
    {
        "api_request_count",
        "api_error_count",
        "training_job_count",
        "training_failures",
        "forecast_job_count",
        "bedrock_calls",
        "bedrock_input_tokens",
        "bedrock_output_tokens",
        "bedrock_failures",
        "explanation_validation_failures",
        "unknown_category_count",
    }
)

_UNITS: dict[str, str] = {
    "api_latency_ms": "Milliseconds",
    "training_duration_seconds": "Seconds",
    "forecast_latency_seconds": "Seconds",
    "bedrock_latency": "Milliseconds",
    "wape": "None",
    "smape": "None",
    "mae": "None",
    "rmse": "None",
    "bias": "None",
    "p90_coverage": "None",
    "latest_data_age_hours": "None",
    "missing_value_rate": "None",
    "duplicate_rate": "None",
    "stockout_rate": "None",
}


class MetricPoint(BaseModel):
    """One aggregated metric sample."""

    name: str
    value: float
    unit: str
    dimensions: dict[str, str] = Field(default_factory=dict)


class MetricsSnapshot(BaseModel):
    """Every catalogued metric, including names that have not been emitted yet."""

    metrics: list[MetricPoint]


@dataclass(frozen=True)
class AlarmDefinition:
    """One operational alarm. Notifications are a separate switch."""

    name: str
    description: str
    metric_names: tuple[str, ...]
    threshold: float
    statistic: str


ALARMS: tuple[AlarmDefinition, ...] = (
    AlarmDefinition(
        name="training-failure",
        description="A training job failed.",
        metric_names=("training_failures",),
        threshold=1,
        statistic="Sum",
    ),
    AlarmDefinition(
        name="repeated-forecast-failure",
        description="More than one forecast job failed in the window.",
        metric_names=("forecast_job_count",),
        threshold=2,
        statistic="Sum",
    ),
    AlarmDefinition(
        name="stale-dataset",
        description="The newest observation is older than 30 days.",
        metric_names=("latest_data_age_hours",),
        threshold=float(_STALE_DATASET_HOURS),
        statistic="Maximum",
    ),
    AlarmDefinition(
        name="high-forecast-error",
        description="Forecast WAPE is at or above 0.5.",
        metric_names=("wape",),
        threshold=0.5,
        statistic="Maximum",
    ),
    AlarmDefinition(
        name="explanation-error-spike",
        description="Explanation validation or provider errors spiked.",
        metric_names=("explanation_validation_failures", "bedrock_failures"),
        threshold=5,
        statistic="Sum",
    ),
)


class MetricSink(Protocol):
    """CloudWatch client method used to publish one batch."""

    def put_metric_data(
        self,
        *,
        Namespace: str,
        MetricData: list[dict[str, object]],
    ) -> object:
        """Publish ``MetricData`` into ``Namespace``."""


class MetricPublisher(Protocol):
    """Where a recorded sample is delivered."""

    def publish(self, point: MetricPoint) -> None:
        """Deliver ``point``."""


class LogMetricPublisher:
    """Write the metric name and value to the structured log."""

    def publish(self, point: MetricPoint) -> None:
        structlog.get_logger().info(
            "metric.emitted",
            metric_name=point.name,
            value=point.value,
            unit=point.unit,
            dimensions=point.dimensions,
        )


class CloudWatchPublisher:
    """Publish a sample to CloudWatch and keep a copy in the log."""

    def __init__(self, client: MetricSink, namespace: str = METRIC_NAMESPACE) -> None:
        self._client = client
        self._namespace = namespace
        self._log = LogMetricPublisher()

    def publish(self, point: MetricPoint) -> None:
        payload: dict[str, object] = {
            "MetricName": point.name,
            "Value": point.value,
            "Unit": point.unit,
        }
        if point.dimensions:
            payload["Dimensions"] = [
                {"Name": key, "Value": value} for key, value in point.dimensions.items()
            ]
        try:
            self._client.put_metric_data(Namespace=self._namespace, MetricData=[payload])
        except Exception:
            structlog.get_logger().warning("metric.publish_failed", metric_name=point.name)
        self._log.publish(point)


router = APIRouter()
_lock = threading.Lock()
_values: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
_publisher: MetricPublisher = LogMetricPublisher()


def notifications_enabled(environment: str, requested: bool) -> bool:
    """Return whether alarm notifications may be subscribed.

    Dev stays unsubscribed. Demo subscribes only when that environment asks.
    """

    return environment == "demo" and requested


def configure_metrics(
    settings: Settings,
    publisher: MetricPublisher | None = None,
) -> None:
    """Select the local log publisher, or CloudWatch when execution is in AWS."""

    global _publisher
    if publisher is not None:
        _publisher = publisher
        return
    if settings.execution_mode is ExecutionMode.AWS:
        _publisher = CloudWatchPublisher(cloudwatch_client(settings.aws_region))
        return
    _publisher = LogMetricPublisher()


def cloudwatch_client(region: str) -> MetricSink:
    """Open a CloudWatch client. Call this only when execution is in AWS."""

    import boto3

    client: MetricSink = boto3.client("cloudwatch", region_name=region)
    return client


def reset_metrics() -> None:
    """Drop aggregated samples. Tests use this to isolate a case."""

    with _lock:
        _values.clear()


def record_metric(
    name: str,
    value: float,
    dimensions: Mapping[str, str] | None = None,
) -> None:
    """Record one sample under a catalogued metric name."""

    if name not in METRIC_NAMES:
        raise ValueError(f"Unknown metric {name}.")
    numeric = _finite(value)
    labels = _dimensions(dimensions or {})
    key = (name, tuple(labels.items()))
    with _lock:
        if name in _COUNTERS:
            _values[key] = _values.get(key, 0.0) + numeric
        else:
            _values[key] = numeric
    _publisher.publish(
        MetricPoint(name=name, value=numeric, unit=_unit(name), dimensions=dict(labels))
    )


def snapshot() -> list[MetricPoint]:
    """Return aggregated samples, plus a zero for every name not yet seen."""

    with _lock:
        stored = dict(_values)
    points = [
        MetricPoint(name=name, value=value, unit=_unit(name), dimensions=dict(labels))
        for (name, labels), value in stored.items()
    ]
    seen = {point.name for point in points}
    points.extend(
        MetricPoint(name=name, value=0, unit=_unit(name), dimensions={})
        for name in sorted(METRIC_NAMES - seen)
    )
    points.sort(key=lambda point: (point.name, tuple(sorted(point.dimensions.items()))))
    return points


def record_model_metrics(
    *,
    model_version: str,
    forecast_horizon: int,
    report: EvaluationReport,
    p90_coverage: float | None,
) -> None:
    """Emit score metrics with model version, category, and horizon labels."""

    horizon = str(forecast_horizon)
    _emit_summary(
        report.global_metrics,
        model_version=model_version,
        category="all",
        forecast_horizon=horizon,
        p90_coverage=p90_coverage,
    )
    for item in report.by_category:
        _emit_summary(
            item.metrics,
            model_version=model_version,
            category=item.key,
            forecast_horizon=horizon,
            p90_coverage=None,
        )


def record_data_quality(
    *,
    latest_data_age_hours: float,
    missing_value_rate: float,
    duplicate_rate: float,
    stockout_rate: float,
    unknown_category_count: int,
) -> None:
    """Emit dataset freshness and quality rates."""

    record_metric("latest_data_age_hours", latest_data_age_hours)
    record_metric("missing_value_rate", missing_value_rate)
    record_metric("duplicate_rate", duplicate_rate)
    record_metric("stockout_rate", stockout_rate)
    record_metric("unknown_category_count", unknown_category_count)


def record_forecast_job(*, outcome: str, latency_seconds: float) -> None:
    """Count one finished forecast and record how long it took."""

    labels = {"outcome": outcome}
    record_metric("forecast_job_count", 1, labels)
    record_metric("forecast_latency_seconds", latency_seconds, labels)


def record_explanation_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    cache_hit: bool,
    provider_failed: bool,
) -> None:
    """Count a language-model call. A cache hit does not count as a call."""

    if not cache_hit:
        record_metric("bedrock_calls", 1)
        record_metric("bedrock_input_tokens", input_tokens)
        record_metric("bedrock_output_tokens", output_tokens)
        record_metric("bedrock_latency", latency_ms)
    if provider_failed:
        record_metric("bedrock_failures", 1)


def record_explanation_validation_failure() -> None:
    """Count one explanation draft that failed a deterministic check."""

    record_metric("explanation_validation_failures", 1)


@router.get("/metrics", response_model=MetricsSnapshot)
def read_metrics() -> MetricsSnapshot:
    """Return the metric names and the values collected in this process."""

    return MetricsSnapshot(metrics=snapshot())


class CorrelationIdMiddleware:
    """Create or reuse a correlation id and attach it to the response and logs."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        correlation_id = _correlation_id(Headers(scope=scope))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
        started = time.perf_counter()
        status_code = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = MutableHeaders(scope=message)
                headers[CORRELATION_HEADER] = correlation_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            _record_http(scope, 500, started)
            raise
        else:
            _record_http(scope, status_code, started)
        finally:
            structlog.contextvars.clear_contextvars()


def _record_http(scope: Scope, status_code: int, started: float) -> None:
    elapsed_ms = (time.perf_counter() - started) * 1000
    record_metric("api_request_count", 1)
    if status_code >= 400:
        record_metric("api_error_count", 1)
    record_metric("api_latency_ms", elapsed_ms)
    structlog.get_logger().info(
        "http.request",
        http_method=str(scope.get("method", "")),
        http_path=str(scope.get("path", "")),
        status_code=status_code,
        duration_ms=round(elapsed_ms, 3),
    )


def _emit_summary(
    summary: MetricSummary,
    *,
    model_version: str,
    category: str,
    forecast_horizon: str,
    p90_coverage: float | None,
) -> None:
    labels = {
        "model_version": model_version,
        "category": category,
        "forecast_horizon": forecast_horizon,
    }
    record_metric("wape", summary.wape, labels)
    record_metric("smape", summary.smape, labels)
    record_metric("mae", summary.mae, labels)
    record_metric("rmse", summary.rmse, labels)
    record_metric("bias", summary.bias, labels)
    if p90_coverage is not None:
        record_metric("p90_coverage", p90_coverage, labels)


def _correlation_id(headers: Headers) -> str:
    incoming = headers.get(CORRELATION_HEADER, "").strip()
    if _CORRELATION_ID.fullmatch(incoming):
        return incoming
    return str(uuid.uuid4())


def _dimensions(dimensions: Mapping[str, str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for key, value in dimensions.items():
        name = key.strip()
        label = value.strip()
        if name == "" or label == "" or len(name) > 255 or len(label) > 255:
            raise ValueError("Metric dimensions need a short name and value.")
        labels[name] = label
    return dict(sorted(labels.items()))


def _finite(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("A metric value must be a real number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("A metric value must be finite.")
    return numeric


def _unit(name: str) -> str:
    return _UNITS.get(name, "Count")
