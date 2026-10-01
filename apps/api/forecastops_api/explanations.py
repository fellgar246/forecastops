"""Request, validate, and store a grounded forecast explanation."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import structlog

from forecastops_api.errors import ErrorBody
from forecastops_api.observability import (
    record_explanation_usage,
    record_explanation_validation_failure,
)
from forecastops_api.persistence import AIExplanationRow, ForecastRunRow, ModelVersionRow
from forecastops_api.repositories import MetadataRepository
from forecastops_api.schemas import ExplanationResponse, ExplanationSignal
from forecastops_api.series import Observation, SeriesPoint, assemble_series, point_matches
from forecastops_api.settings import Settings
from forecastops_ml.data import load_dataset
from forecastops_ml.explanations import (
    PROMPT_VERSION,
    ExplanationClient,
    ExplanationDraft,
    ExplanationPackage,
    ExplanationValidationError,
    build_explanation_package,
    explanation_validation_failures,
    render_prompt,
    select_explanation_client,
    validate_explanation,
)
from forecastops_ml.explanations.clients import estimate_tokens

EXPLANATIONS_DISABLED = "Explanations are not enabled."
_RESET = "00:00 UTC"


@dataclass(frozen=True)
class ExplanationResult:
    """HTTP status and JSON body for one explanation route."""

    status_code: int
    content: dict[str, Any]


class ExplanationService:
    """Serve cached explanations and call the selected adapter on a miss."""

    def __init__(
        self,
        repository: MetadataRepository,
        settings: Settings,
        client: ExplanationClient | None = None,
    ) -> None:
        self._repository = repository
        self._settings = settings
        self._client = client

    def request(
        self,
        forecast_id: str,
        *,
        category: str | None,
        store: str | None,
        sku: str | None,
    ) -> ExplanationResult:
        """Return a cached explanation or generate one from the forecast package."""

        if not self._settings.ai_enabled:
            return _error(409, "explanations_disabled", EXPLANATIONS_DISABLED)
        forecast = self._repository.get_forecast(forecast_id)
        if forecast is None:
            return _error(404, "not_found", "Forecast was not found.")
        if forecast.status != "SUCCEEDED":
            return _error(
                409,
                "forecast_not_ready",
                "The forecast is not ready for an explanation.",
            )
        scope = _scope(category, store, sku)
        key = _scope_key(scope)
        cached = self._repository.find_explanation(forecast.id, key, PROMPT_VERSION)
        if cached is not None:
            self._log(cached, cache_hit=True, accepted=True)
            return ExplanationResult(200, _present(cached))
        if self._over_daily_limit():
            return self._quota("calls")
        try:
            package = self._package(forecast, scope, category=category, store=store, sku=sku)
        except _PackageError as exc:
            return _error(422, "explanation_unavailable", str(exc))
        if estimate_tokens(render_prompt(package)) > self._settings.max_bedrock_input_tokens:
            return self._quota("input")
        client = self._adapter()
        try:
            draft = client.explain(package)
        except ExplanationValidationError as exc:
            self._record(forecast.id, scope, key, client, package, draft=None, status="invalid")
            self._log_usage(forecast.id, client, cache_hit=False, accepted=False)
            _log_validation(forecast.id, exc)
            return _invalid(exc)
        except Exception:
            # Provider failures are not one exception type. Count the call, then surface it.
            record_explanation_usage(
                input_tokens=client.usage.input_tokens,
                output_tokens=client.usage.output_tokens,
                latency_ms=client.usage.latency_ms,
                cache_hit=False,
                provider_failed=True,
            )
            raise
        usage = client.usage
        if usage.input_tokens > self._settings.max_bedrock_input_tokens:
            self._record(forecast.id, scope, key, client, package, draft=None, status="invalid")
            self._log_usage(forecast.id, client, cache_hit=False, accepted=False)
            return self._quota("input")
        if usage.output_tokens > self._settings.max_bedrock_output_tokens:
            self._record(forecast.id, scope, key, client, package, draft=None, status="invalid")
            self._log_usage(forecast.id, client, cache_hit=False, accepted=False)
            return self._quota("output")
        try:
            validate_explanation(package, draft)
        except ExplanationValidationError as exc:
            self._record(forecast.id, scope, key, client, package, draft=None, status="invalid")
            self._log_usage(forecast.id, client, cache_hit=False, accepted=False)
            _log_validation(forecast.id, exc)
            return _invalid(exc)
        row = self._record(forecast.id, scope, key, client, package, draft=draft, status="valid")
        self._log(row, cache_hit=False, accepted=True)
        return ExplanationResult(202, _present(row))

    def read(
        self,
        forecast_id: str,
        *,
        category: str | None,
        store: str | None,
        sku: str | None,
    ) -> ExplanationResult:
        """Return the stored explanation for this forecast and scope."""

        if not self._settings.ai_enabled:
            return _error(409, "explanations_disabled", EXPLANATIONS_DISABLED)
        forecast = self._repository.get_forecast(forecast_id)
        if forecast is None:
            return _error(404, "not_found", "Forecast was not found.")
        row = self._repository.latest_explanation(
            forecast.id,
            _scope_key(_scope(category, store, sku)),
        )
        if row is None:
            return _error(404, "not_found", "No explanation has been generated yet.")
        return ExplanationResult(200, _present(row))

    def _adapter(self) -> ExplanationClient:
        if self._client is not None:
            return self._client
        return select_explanation_client(
            bedrock_enabled=self._settings.bedrock_enabled,
            model_id=self._settings.bedrock_model_id,
            max_output_tokens=self._settings.max_bedrock_output_tokens,
            region=self._settings.aws_region,
        )

    def _over_daily_limit(self) -> bool:
        used = self._repository.count_explanation_calls_on(datetime.now(UTC).date())
        return used >= self._settings.max_bedrock_calls_per_day

    def _quota(self, kind: str) -> ExplanationResult:
        limit = {
            "calls": self._settings.max_bedrock_calls_per_day,
            "input": self._settings.max_bedrock_input_tokens,
            "output": self._settings.max_bedrock_output_tokens,
        }[kind]
        if kind == "calls":
            message = (
                f"The daily explanation limit of {limit} calls has been reached. "
                f"It resets at {_RESET}."
            )
        elif kind == "input":
            message = (
                f"The explanation exceeds the input token ceiling of {limit}. "
                f"It resets at {_RESET}."
            )
        else:
            message = (
                f"The explanation exceeds the output token ceiling of {limit}. "
                f"It resets at {_RESET}."
            )
        return _error(429, "explanation_quota", message)

    def _package(
        self,
        forecast: ForecastRunRow,
        scope: dict[str, str],
        *,
        category: str | None,
        store: str | None,
        sku: str | None,
    ) -> ExplanationPackage:
        model = self._repository.get_model(forecast.model_version_id)
        if model is None:
            raise _PackageError("The forecast model was not found.")
        points = self._repository.list_points(forecast.id)
        categories = self._sku_categories(forecast.dataset_id) if category else {}
        selected = [
            point
            for point in points
            if point_matches(
                point.series_id,
                store=store,
                sku=sku,
                category=category,
                sku_categories=categories,
            )
        ]
        views = [
            SeriesPoint(
                series_id=point.series_id,
                day=point.date,
                p10=point.p10,
                p50=point.p50,
                p90=point.p90,
                actual=point.actual,
            )
            for point in selected
        ]
        horizon_days = forecast.horizon * 7 if forecast.granularity == "week" else forecast.horizon
        observations = self._observations(forecast.dataset_id) if views else []
        assembled = assemble_series(
            views,
            observations,
            horizon_days=horizon_days,
            category=category,
        )
        if assembled.p50_total is None or assembled.cutoff is None:
            raise _PackageError("This forecast has no points to explain.")
        previous, year_over_year = _history(
            observations,
            {point.series_id for point in views},
            assembled.cutoff,
            horizon_days,
        )
        weeks = float(forecast.horizon) if forecast.granularity == "week" else forecast.horizon / 7
        return build_explanation_package(
            scope=scope,
            model_family=model.model_family,
            horizon_weeks=weeks,
            p10=assembled.p10_total,
            p50=assembled.p50_total,
            p90=assembled.p90_total,
            previous_period_units=previous,
            year_over_year_pct=year_over_year,
            wape=_metric(model, "wape"),
            bias=_metric(model, "bias"),
        )

    def _record(
        self,
        forecast_id: str,
        scope: dict[str, str],
        key: str,
        client: ExplanationClient,
        package: ExplanationPackage,
        *,
        draft: ExplanationDraft | None,
        status: str,
    ) -> AIExplanationRow:
        usage = client.usage
        row = AIExplanationRow(
            id=str(uuid4()),
            forecast_run_id=forecast_id,
            scope_key=key,
            scope=scope,
            model_id=client.model_id,
            prompt_version=PROMPT_VERSION,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=usage.latency_ms,
            explanation=(
                None if draft is None else cast(dict[str, object], draft.model_dump(mode="json"))
            ),
            package=cast(dict[str, object], package.model_dump(mode="json")),
            status=status,
            created_at=datetime.now(UTC),
        )
        self._repository.add(row)
        return row

    def _log(self, row: AIExplanationRow, *, cache_hit: bool, accepted: bool) -> None:
        record_explanation_usage(
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            latency_ms=row.latency_ms,
            cache_hit=cache_hit,
            provider_failed=False,
        )
        structlog.get_logger().info(
            "explanation.completed",
            forecast_run_id=row.forecast_run_id,
            model_id=row.model_id,
            prompt_version=row.prompt_version,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            latency_ms=row.latency_ms,
            cache_hit=cache_hit,
            accepted=accepted,
        )

    def _log_usage(
        self,
        forecast_id: str,
        client: ExplanationClient,
        *,
        cache_hit: bool,
        accepted: bool,
    ) -> None:
        usage = client.usage
        record_explanation_usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=usage.latency_ms,
            cache_hit=cache_hit,
            provider_failed=False,
        )
        structlog.get_logger().info(
            "explanation.completed",
            forecast_run_id=forecast_id,
            model_id=client.model_id,
            prompt_version=PROMPT_VERSION,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=usage.latency_ms,
            cache_hit=cache_hit,
            accepted=accepted,
        )

    def _sku_categories(self, dataset_id: str) -> dict[str, str]:
        dataset = self._repository.get_dataset(dataset_id)
        if dataset is None:
            return {}
        try:
            _frame, dimensions = load_dataset(Path(dataset.uri))
        except ValueError:
            return {}
        return {
            str(sku_id): str(category_id)
            for sku_id, category_id in zip(
                dimensions.skus.column("sku_id").to_pylist(),
                dimensions.skus.column("category_id").to_pylist(),
                strict=True,
            )
        }

    def _observations(self, dataset_id: str) -> list[Observation]:
        dataset = self._repository.get_dataset(dataset_id)
        if dataset is None:
            return []
        try:
            frame, _dimensions = load_dataset(Path(dataset.uri))
        except ValueError:
            return []
        rows: list[Observation] = []
        columns = zip(
            frame.column("date").to_pylist(),
            frame.column("store_id").to_pylist(),
            frame.column("sku_id").to_pylist(),
            frame.column("category_id").to_pylist(),
            frame.column("units_sold").to_pylist(),
            strict=True,
        )
        for day, store_id, sku_id, category_id, sold in columns:
            if not isinstance(day, date) or sold is None:
                continue
            rows.append(
                Observation(
                    day=day,
                    series_id=f"{store_id}|{sku_id}",
                    category_id=str(category_id),
                    units=float(sold),
                )
            )
        return rows


class _PackageError(Exception):
    """The forecast cannot be turned into a package."""


def _present(row: AIExplanationRow) -> dict[str, Any]:
    if row.explanation is None or row.package is None:
        raise ValueError("A valid explanation is missing its package.")
    draft = ExplanationDraft.model_validate(row.explanation)
    package = ExplanationPackage.model_validate(row.package)
    named = set(draft.drivers)
    signals = [
        ExplanationSignal(
            name=signal.name,
            direction=signal.direction,
            value=signal.value,
            kind=signal.kind,
        )
        for signal in package.signals
        if signal.name in named
    ]
    body = ExplanationResponse(
        id=row.id,
        model_id=row.model_id,
        prompt_version=row.prompt_version,
        summary=draft.summary,
        signals=signals,
        risks=draft.risks,
        uncertainty=draft.uncertainty_note,
        checks=draft.recommended_checks,
        generated_at=row.created_at,
        package=cast(dict[str, object], package.model_dump(mode="json")),
    )
    dumped: dict[str, Any] = body.model_dump(mode="json", exclude_none=True)
    return dumped


def _history(
    observations: Sequence[Observation],
    series_ids: set[str],
    cutoff: date,
    horizon_days: int,
) -> tuple[float | None, float | None]:
    recent_start = cutoff - timedelta(days=horizon_days)
    prior_end = cutoff - timedelta(days=365)
    prior_start = prior_end - timedelta(days=horizon_days)
    recent = 0.0
    recent_seen = False
    prior = 0.0
    prior_seen = False
    for row in observations:
        if row.series_id not in series_ids:
            continue
        if recent_start <= row.day < cutoff:
            recent += row.units
            recent_seen = True
        elif prior_start <= row.day < prior_end:
            prior += row.units
            prior_seen = True
    previous = recent if recent_seen else None
    year_over_year = None
    if recent_seen and prior_seen and prior != 0:
        year_over_year = (recent - prior) / prior * 100.0
    return previous, year_over_year


def _metric(model: ModelVersionRow, key: str) -> float | None:
    metrics = model.metrics
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _scope(category: str | None, store: str | None, sku: str | None) -> dict[str, str]:
    scope: dict[str, str] = {}
    if category:
        scope["category"] = category
    if store:
        scope["store"] = store
    if sku:
        scope["sku"] = sku
    if not scope:
        scope["series"] = "all"
    return scope


def _scope_key(scope: dict[str, str]) -> str:
    return json.dumps(scope, sort_keys=True, separators=(",", ":"))


def _invalid(exc: ExplanationValidationError) -> ExplanationResult:
    return _error(
        422,
        "explanation_invalid",
        exc.message,
        {"check": exc.check},
    )


def _log_validation(forecast_id: str, exc: ExplanationValidationError) -> None:
    record_explanation_validation_failure()
    structlog.get_logger().info(
        "explanation.validation_failed",
        forecast_run_id=forecast_id,
        check=exc.check,
        explanation_validation_failures=explanation_validation_failures(),
    )


def _error(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, object] | None = None,
) -> ExplanationResult:
    body = ErrorBody(code=code, message=message, details=details or {})
    return ExplanationResult(status_code, body.model_dump())
