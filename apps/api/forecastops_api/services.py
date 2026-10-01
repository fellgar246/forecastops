"""Forecast lifecycle operations. SQL stays in the repository."""

import time
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import structlog

from forecastops_api.artifacts import (
    PRESIGNED_UPLOAD_SECONDS,
    ArtifactStore,
    InvalidArtifactPrefix,
    PresigningStore,
    upload_object_name,
)
from forecastops_api.errors import ApiError
from forecastops_api.forecast_jobs import ForecastExecutor, ForecastPredictor
from forecastops_api.observability import record_data_quality
from forecastops_api.persistence import (
    DatasetRow,
    ForecastRunRow,
    ModelVersionRow,
    PromotionDecisionRow,
    TrainingRunRow,
)
from forecastops_api.repositories import MetadataRepository
from forecastops_api.runner import LocalJobRunner
from forecastops_api.schemas import (
    ApproveRequest,
    CatalogEntry,
    DataQualityItem,
    DatasetCatalog,
    DatasetCreate,
    DatasetResponse,
    DatasetUploadResponse,
    DatasetValidate,
    ForecastCreate,
    ForecastResponse,
    ForecastSeries,
    GateChecksBody,
    ModelResponse,
    PresignedUploadResponse,
    PromotionBody,
    RejectRequest,
    SegmentRegressionBody,
    SkuEntry,
    TrainingCreate,
    TrainingResponse,
)
from forecastops_api.series import (
    Observation,
    SeriesPoint,
    assemble_series,
    empty_series,
    point_matches,
)
from forecastops_api.settings import ExecutionMode, Settings
from forecastops_contracts import SCHEMA_VERSION
from forecastops_ml.data import load_dataset, validate_dataset
from forecastops_ml.data.quality import ValidationConfig, unexpected_category_count
from forecastops_ml.inference import BatchCapacityError, BatchInference, admit_forecast
from forecastops_ml.promotion.models import ModelStatus, RegistryStatus
from forecastops_ml.registry import (
    ForecastSelectionError,
    ModelRegistry,
    RegistryEntry,
    RegistryError,
    registry_status_for,
    select_forecast_model,
)

_CLOUD_FAMILIES = {"deepar"}


class ForecastService:
    """Register data, train local models, and serve stored forecasts."""

    def __init__(
        self,
        repository: MetadataRepository,
        settings: Settings,
        artifacts: ArtifactStore,
        registry: ModelRegistry | None = None,
        *,
        batch: BatchInference | None = None,
        predictor: ForecastPredictor | None = None,
    ) -> None:
        self._repository = repository
        self._settings = settings
        self._artifacts = artifacts
        self._registry = registry
        self._runner = LocalJobRunner(repository, artifacts)
        self._forecasts = ForecastExecutor(repository, artifacts, batch, predictor)

    def register_dataset(self, body: DatasetCreate) -> DatasetRow:
        """Record a dataset directory and its observation window."""

        path = Path(body.uri)
        if not path.is_dir():
            raise ApiError(422, "invalid_dataset", "Dataset uri must be a directory.")
        try:
            frame, _dimensions = load_dataset(path)
        except ValueError as exc:
            raise ApiError(422, "invalid_dataset", str(exc)) from exc
        dates = [value for value in frame.column("date").to_pylist() if isinstance(value, date)]
        row = DatasetRow(
            id=str(uuid4()),
            name=body.name,
            version=self._repository.next_dataset_version(body.name),
            source=body.source,
            status="registered",
            uri=str(path),
            schema_version=SCHEMA_VERSION,
            row_count=frame.num_rows,
            date_min=min(dates) if dates else None,
            date_max=max(dates) if dates else None,
            quality_report=None,
            created_at=_now(),
        )
        self._repository.add(row)
        return row

    def store_uploaded_dataset(self, filename: str, body: bytes) -> DatasetUploadResponse:
        """Store dataset bytes. Local mode writes them through the artifact store."""

        if self._settings.aws_enabled:
            raise ApiError(
                409,
                "presigned_upload_required",
                "Upload the dataset with the pre-signed URL when AWS is enabled.",
            )
        try:
            name = upload_object_name(filename)
            uri = self._artifacts.put("raw", name, body)
        except InvalidArtifactPrefix as exc:
            raise ApiError(422, "invalid_artifact", str(exc)) from exc
        return DatasetUploadResponse(uri=uri, name=name)

    def issue_presigned_upload(self, filename: str) -> PresignedUploadResponse:
        """Issue a pre-signed upload URL when the remote store is selected."""

        if not self._settings.aws_enabled or not isinstance(self._artifacts, PresigningStore):
            raise ApiError(
                409,
                "aws_disabled",
                "Pre-signed dataset uploads are available only when AWS is enabled.",
            )
        try:
            name = upload_object_name(filename)
            url, uri = self._artifacts.presign_put(
                "raw",
                name,
                expires_in=PRESIGNED_UPLOAD_SECONDS,
            )
        except InvalidArtifactPrefix as exc:
            raise ApiError(422, "invalid_artifact", str(exc)) from exc
        return PresignedUploadResponse(url=url, uri=uri, expires_in=PRESIGNED_UPLOAD_SECONDS)

    def list_datasets(self) -> list[DatasetRow]:
        """Return every registered dataset."""

        return self._repository.list_datasets()

    def get_dataset(self, dataset_id: str) -> DatasetRow:
        """Return one dataset or raise a not-found error."""

        return _require_dataset(self._repository, dataset_id)

    def validate_dataset(self, dataset_id: str, body: DatasetValidate) -> DatasetRow:
        """Run the quality checks and store the report on the dataset."""

        dataset = _require_dataset(self._repository, dataset_id)
        started = time.perf_counter()
        frame, dimensions = load_dataset(Path(dataset.uri))
        as_of = body.as_of or dataset.date_max or date.today()
        report = validate_dataset(frame, dimensions, ValidationConfig(as_of=as_of))
        dataset.status = report.status
        dataset.quality_report = report.to_dict()
        self._repository.save()
        age_hours = 0.0
        if dataset.date_max is not None:
            age_hours = max((as_of - dataset.date_max).total_seconds() / 3600, 0.0)
        record_data_quality(
            latest_data_age_hours=age_hours,
            missing_value_rate=report.missing_value_rate,
            duplicate_rate=report.duplicate_rate,
            stockout_rate=report.stockout_rate,
            unknown_category_count=unexpected_category_count(frame, dimensions.categories),
        )
        structlog.get_logger().info(
            "dataset.validated",
            dataset_id=dataset.id,
            status=dataset.status,
            artifact_uri=dataset.uri,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return dataset

    def start_training(self, body: TrainingCreate) -> TrainingRunRow:
        """Queue a training run and execute it in this process."""

        if not self._settings.training_enabled:
            raise ApiError(409, "training_disabled", "Training is disabled.")
        if body.model_family in _CLOUD_FAMILIES:
            raise ApiError(
                409,
                "cloud_model_unavailable",
                f"{body.model_family} training is not enabled in the local profile.",
            )
        dataset = _require_dataset(self._repository, body.dataset_id)
        if dataset.status != "valid":
            raise ApiError(409, "dataset_not_valid", "Train only a dataset whose status is valid.")
        run = TrainingRunRow(
            id=str(uuid4()),
            dataset_id=dataset.id,
            dataset_version=dataset.version,
            model_family=body.model_family,
            configuration=dict(body.configuration),
            status="QUEUED",
            started_at=None,
            finished_at=None,
            artifact_uri="",
            metrics=None,
            git_sha="",
            pipeline_execution_arn="",
            error_message=None,
            created_at=_now(),
        )
        self._repository.add(run)
        frame, _dimensions = load_dataset(Path(dataset.uri))
        model = self._runner.train(run, frame)
        if model is not None:
            self._register_gated_model(model)
        return run

    def list_training_runs(self) -> list[TrainingRunRow]:
        """Return every training run."""

        return self._repository.list_training_runs()

    def get_training_run(self, run_id: str) -> TrainingRunRow:
        """Return one training run."""

        run = self._repository.get_training_run(run_id)
        if run is None:
            raise ApiError(404, "not_found", "Training run was not found.")
        return run

    def list_models(self) -> list[ModelVersionRow]:
        """Return every model version."""

        return self._repository.list_models()

    def get_model(self, model_id: str) -> ModelVersionRow:
        """Return one model version."""

        return _require_model(self._repository, model_id)

    def approve(self, model_id: str, body: ApproveRequest) -> ModelVersionRow:
        """Approve a pending model. Optionally place it in production."""

        model = _require_model(self._repository, model_id)
        if model.status != "PENDING_APPROVAL":
            raise ApiError(
                409,
                "invalid_transition",
                f"Model cannot be approved from {model.status}.",
            )
        entry = self._record_human_decision(model, body.actor_id, approve=True)
        model.status = "APPROVED"
        model.approved_at = _now()
        model.approved_by = body.actor_id
        _apply_registry_entry(model, entry)
        self._repository.add(
            PromotionDecisionRow(
                id=str(uuid4()),
                candidate_id=model.id,
                reference_id=model.id,
                thresholds={},
                checks={},
                status="APPROVED",
                reason=None,
                p90_coverage=None,
                regressed_categories=[],
                actor_id=body.actor_id,
                created_at=_now(),
            )
        )
        if body.promote:
            self._promote(model, body.actor_id)
        self._repository.save()
        return model

    def reject(self, model_id: str, body: RejectRequest) -> ModelVersionRow:
        """Reject a pending model. The reason must be human."""

        if body.reason != "human":
            raise ApiError(422, "invalid_reason", "Human rejection uses reason 'human'.")
        model = _require_model(self._repository, model_id)
        if model.status != "PENDING_APPROVAL":
            raise ApiError(
                409,
                "invalid_transition",
                f"Model cannot be rejected from {model.status}.",
            )
        entry = self._record_human_decision(model, body.actor_id, approve=False)
        model.status = "REJECTED"
        model.rejected_by = body.actor_id
        model.rejection_reason = "human"
        _apply_registry_entry(model, entry)
        self._repository.add(
            PromotionDecisionRow(
                id=str(uuid4()),
                candidate_id=model.id,
                reference_id=model.id,
                thresholds={},
                checks={},
                status="REJECTED",
                reason="human",
                p90_coverage=None,
                regressed_categories=[],
                actor_id=body.actor_id,
                created_at=_now(),
            )
        )
        self._repository.save()
        return model

    def start_forecast(self, body: ForecastCreate, *, execute: bool = True) -> ForecastRunRow:
        """Queue a forecast. Execution follows when ``execute`` is true."""

        key = _idempotency_key(body.idempotency_key)
        if key is not None:
            existing = self._repository.get_forecast_by_idempotency_key(key)
            if existing is not None:
                return existing
        self._require_horizon(body.horizon, body.granularity)
        model = _require_model(self._repository, body.model_id)
        self._require_forecastable(model)
        dataset_id = body.dataset_id or model.dataset_id
        dataset = _require_dataset(self._repository, dataset_id)
        self._require_batch_capacity()
        forecast = ForecastRunRow(
            id=str(uuid4()),
            model_version_id=model.id,
            dataset_id=dataset.id,
            dataset_version=dataset.version,
            horizon=body.horizon,
            granularity=body.granularity,
            status="QUEUED",
            output_uri="",
            error_message=None,
            idempotency_key=key,
            created_at=_now(),
        )
        self._repository.add(forecast)
        self._repository.save()
        if execute:
            self.execute_forecast(forecast.id)
        return forecast

    def execute_forecast(self, forecast_id: str) -> ForecastRunRow:
        """Run a queued forecast and return the updated row."""

        forecast = self.get_forecast(forecast_id)
        self._forecasts.run(forecast)
        return forecast

    def list_forecasts(self) -> list[ForecastRunRow]:
        """Return every forecast run."""

        return self._repository.list_forecasts()

    def get_forecast(self, forecast_id: str) -> ForecastRunRow:
        """Return one forecast run."""

        forecast = self._repository.get_forecast(forecast_id)
        if forecast is None:
            raise ApiError(404, "not_found", "Forecast was not found.")
        return forecast

    def forecast_series(
        self,
        forecast_id: str,
        *,
        category: str | None = None,
        sku: str | None = None,
        store: str | None = None,
        horizon: int | None = None,
    ) -> ForecastSeries:
        """Return stored points for one forecast, limited to the requested filters."""

        forecast = self.get_forecast(forecast_id)
        if horizon is not None and forecast.horizon != horizon:
            return empty_series()
        points = self._repository.list_points(forecast_id)
        sku_categories = self._sku_categories(forecast.dataset_id) if category else {}
        selected = [
            point
            for point in points
            if point_matches(
                point.series_id,
                store=store,
                sku=sku,
                category=category,
                sku_categories=sku_categories,
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
        observations = self._observations(forecast.dataset_id) if views else []
        horizon_days = forecast.horizon * 7 if forecast.granularity == "week" else forecast.horizon
        return assemble_series(
            views,
            observations,
            horizon_days=horizon_days,
            category=category,
        )

    def catalog(self, dataset_id: str) -> DatasetCatalog:
        """Return stores, categories, and SKUs from a dataset directory."""

        dataset = _require_dataset(self._repository, dataset_id)
        try:
            _frame, dimensions = load_dataset(Path(dataset.uri))
        except ValueError as exc:
            raise ApiError(422, "invalid_dataset", str(exc)) from exc
        stores = [
            CatalogEntry(id=str(store_id), name=str(store_id))
            for store_id in dimensions.stores.column("store_id").to_pylist()
        ]
        categories = [
            CatalogEntry(id=str(category_id), name=str(name))
            for category_id, name in zip(
                dimensions.categories.column("category_id").to_pylist(),
                dimensions.categories.column("category_name").to_pylist(),
                strict=True,
            )
        ]
        skus = [
            SkuEntry(id=str(sku_id), category_id=str(category_id))
            for sku_id, category_id in zip(
                dimensions.skus.column("sku_id").to_pylist(),
                dimensions.skus.column("category_id").to_pylist(),
                strict=True,
            )
        ]
        return DatasetCatalog(
            stores=sorted(stores, key=lambda item: item.id),
            categories=sorted(categories, key=lambda item: item.id),
            skus=sorted(skus, key=lambda item: item.id),
        )

    def present_model(self, row: ModelVersionRow) -> ModelResponse:
        """Map a model row, including its quality-gate record."""

        decisions = self._repository.list_decisions(row.id)
        promotion = _promotion_body(decisions)
        return model_response(row, promotion)

    def present_forecast(self, row: ForecastRunRow) -> ForecastResponse:
        """Map a forecast row together with its model identity."""

        model = self.get_model(row.model_version_id)
        return forecast_response(row, model)

    def _sku_categories(self, dataset_id: str) -> dict[str, str]:
        dataset = _require_dataset(self._repository, dataset_id)
        try:
            _frame, dimensions = load_dataset(Path(dataset.uri))
        except ValueError as exc:
            raise ApiError(422, "invalid_dataset", str(exc)) from exc
        return {
            str(sku_id): str(category_id)
            for sku_id, category_id in zip(
                dimensions.skus.column("sku_id").to_pylist(),
                dimensions.skus.column("category_id").to_pylist(),
                strict=True,
            )
        }

    def _observations(self, dataset_id: str) -> list[Observation]:
        dataset = _require_dataset(self._repository, dataset_id)
        try:
            frame, _dimensions = load_dataset(Path(dataset.uri))
        except ValueError:
            return []
        dates = frame.column("date").to_pylist()
        stores = frame.column("store_id").to_pylist()
        skus = frame.column("sku_id").to_pylist()
        categories = frame.column("category_id").to_pylist()
        units = frame.column("units_sold").to_pylist()
        rows: list[Observation] = []
        for day, store_id, sku_id, category_id, sold in zip(
            dates, stores, skus, categories, units, strict=True
        ):
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

    def model_performance(self) -> list[ModelVersionRow]:
        """Return models and the metrics used for promotion."""

        return self._repository.list_models()

    def data_quality(self) -> list[DataQualityItem]:
        """Return the quality report stored on each dataset."""

        return [
            DataQualityItem(
                dataset_id=row.id,
                dataset_version=row.version,
                status=row.status,  # type: ignore[arg-type]
                quality_report=row.quality_report,
            )
            for row in self._repository.list_datasets()
        ]

    def retrain(self) -> TrainingRunRow:
        """Train the production model's family again. Do not approve the result."""

        production = self._repository.production_model()
        if production is None:
            raise ApiError(
                409,
                "no_production_model",
                "No production model is available to retrain.",
            )
        return self.start_training(
            TrainingCreate(
                dataset_id=production.dataset_id,
                model_family=production.model_family,  # type: ignore[arg-type]
            )
        )

    def _cloud_forecast(self) -> bool:
        return self._registry is not None or self._settings.execution_mode is ExecutionMode.AWS

    def _register_gated_model(self, model: ModelVersionRow) -> None:
        """Record a gated candidate when the cloud registry is enabled."""

        registry = self._registry
        if registry is None:
            return
        try:
            internal = ModelStatus(model.status)
        except ValueError:
            return
        if internal not in {ModelStatus.PENDING_APPROVAL, ModelStatus.REJECTED}:
            return
        try:
            entry = registry.register(
                model_id=model.id,
                model_family=model.model_family,
                version=model.version,
                status=registry_status_for(internal),
            )
        except RegistryError as exc:
            raise ApiError(409, "registry_rejected", str(exc)) from exc
        _apply_registry_entry(model, entry)
        self._repository.save()

    def _record_human_decision(
        self,
        model: ModelVersionRow,
        actor_id: str,
        *,
        approve: bool,
    ) -> RegistryEntry | None:
        """Update the registry when one is attached."""

        registry = self._registry
        if registry is None:
            return None
        if model.registry_arn == "":
            raise ApiError(409, "not_registered", "A cloud decision requires a registered model.")
        try:
            if approve:
                return registry.approve(model.registry_arn, actor_id=actor_id)
            return registry.reject(model.registry_arn, actor_id=actor_id)
        except RegistryError as exc:
            raise ApiError(409, "invalid_transition", str(exc)) from exc

    def _require_forecastable(self, model: ModelVersionRow) -> None:
        cloud = self._cloud_forecast()
        try:
            select_forecast_model(
                internal_status=ModelStatus(model.status),
                registry_status=self._observed_registry_status(model) if cloud else None,
                cloud=cloud,
            )
        except (ForecastSelectionError, ValueError) as exc:
            raise ApiError(409, "model_not_approved", str(exc)) from exc

    def _observed_registry_status(self, model: ModelVersionRow) -> RegistryStatus | None:
        registry = self._registry
        if registry is not None and model.registry_arn:
            try:
                return registry.get(model.registry_arn).status
            except RegistryError as exc:
                raise ApiError(409, "model_not_approved", str(exc)) from exc
        if model.registry_status == "":
            return None
        try:
            return RegistryStatus(model.registry_status)
        except ValueError:
            return None

    def _promote(self, model: ModelVersionRow, actor_id: str) -> None:
        for other in self._repository.production_for_family(model.model_family):
            if other.id == model.id:
                continue
            other.status = "APPROVED"
        model.status = "PRODUCTION"
        self._repository.add(
            PromotionDecisionRow(
                id=str(uuid4()),
                candidate_id=model.id,
                reference_id=model.id,
                thresholds={},
                checks={},
                status="PRODUCTION",
                reason=None,
                p90_coverage=None,
                regressed_categories=[],
                actor_id=actor_id,
                created_at=_now(),
            )
        )

    def _require_batch_capacity(self) -> None:
        limit = self._settings.max_batch_inference_jobs_per_day
        started = self._repository.count_forecasts_created_on(_now().date())
        try:
            admit_forecast(jobs_started_today=started, daily_limit=limit, replay=False)
        except BatchCapacityError as exc:
            raise ApiError(
                429,
                "batch_inference_limit",
                str(exc),
                {"max_batch_inference_jobs_per_day": limit},
            ) from exc

    def _require_horizon(self, horizon: int, granularity: str) -> None:
        days = horizon * 7 if granularity == "week" else horizon
        limit = self._settings.max_forecast_horizon_days
        if days > limit:
            raise ApiError(
                422,
                "horizon_too_long",
                f"Forecast horizon cannot exceed {limit} days.",
                {"horizon_days": days, "max_forecast_horizon_days": limit},
            )


def _idempotency_key(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.strip()
    if key == "":
        return None
    return key


def dataset_response(row: DatasetRow) -> DatasetResponse:
    """Map a dataset row to the API schema."""

    return DatasetResponse(
        id=row.id,
        name=row.name,
        version=row.version,
        source=row.source,  # type: ignore[arg-type]
        status=row.status,  # type: ignore[arg-type]
        uri=row.uri,
        schema_version=row.schema_version,
        row_count=row.row_count,
        date_min=row.date_min,
        date_max=row.date_max,
        quality_report=row.quality_report,
        created_at=row.created_at,
    )


def training_response(row: TrainingRunRow) -> TrainingResponse:
    """Map a training row to the API schema."""

    return TrainingResponse(
        id=row.id,
        dataset_id=row.dataset_id,
        dataset_version=row.dataset_version,
        model_family=row.model_family,
        configuration=row.configuration,
        status=row.status,  # type: ignore[arg-type]
        started_at=row.started_at,
        finished_at=row.finished_at,
        artifact_uri=row.artifact_uri,
        metrics=row.metrics,
        error_message=row.error_message,
        git_sha=row.git_sha,
        pipeline_execution_arn=row.pipeline_execution_arn,
        created_at=row.created_at,
    )


def model_response(row: ModelVersionRow, promotion: PromotionBody | None = None) -> ModelResponse:
    """Map a model row to the API schema."""

    return ModelResponse(
        id=row.id,
        model_family=row.model_family,
        version=row.version,
        training_run_id=row.training_run_id,
        dataset_id=row.dataset_id,
        dataset_version=row.dataset_version,
        registry_arn=row.registry_arn,
        registry_status=row.registry_status,
        status=row.status,  # type: ignore[arg-type]
        metrics=row.metrics,
        rejection_reason=row.rejection_reason,
        promotion=promotion,
        approved_at=row.approved_at,
        created_at=row.created_at,
    )


def forecast_response(row: ForecastRunRow, model: ModelVersionRow) -> ForecastResponse:
    """Map a forecast row to the API schema."""

    return ForecastResponse(
        id=row.id,
        model_id=model.id,
        model_family=model.model_family,
        model_version=model.version,
        dataset_id=row.dataset_id,
        dataset_version=row.dataset_version,
        horizon=row.horizon,
        granularity=row.granularity,  # type: ignore[arg-type]
        status=row.status,  # type: ignore[arg-type]
        output_uri=row.output_uri,
        error_message=row.error_message,
        created_at=row.created_at,
    )


def _promotion_body(decisions: list[PromotionDecisionRow]) -> PromotionBody | None:
    """Return the latest decision that stored quality-gate checks."""

    gated = [row for row in decisions if row.checks]
    if not gated:
        return None
    row = gated[-1]
    thresholds = {
        str(key): float(value)
        for key, value in row.thresholds.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    return PromotionBody(
        reference_id=row.reference_id,
        thresholds=thresholds,
        checks=GateChecksBody.model_validate(row.checks),
        reason=row.reason,
        p90_coverage=row.p90_coverage,
        regressed_categories=[
            SegmentRegressionBody.model_validate(item) for item in row.regressed_categories
        ],
    )


def _require_dataset(repository: MetadataRepository, dataset_id: str) -> DatasetRow:
    dataset = repository.get_dataset(dataset_id)
    if dataset is None:
        raise ApiError(404, "not_found", "Dataset was not found.")
    return dataset


def _apply_registry_entry(model: ModelVersionRow, entry: RegistryEntry | None) -> None:
    if entry is None:
        return
    model.registry_arn = entry.arn
    model.version = entry.version
    model.registry_status = entry.status.value


def _require_model(repository: MetadataRepository, model_id: str) -> ModelVersionRow:
    model = repository.get_model(model_id)
    if model is None:
        raise ApiError(404, "not_found", "Model was not found.")
    return model


def _now() -> datetime:
    return datetime.now(UTC)
