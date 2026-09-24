"""Forecast lifecycle operations. SQL stays in the repository."""

from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

from forecastops_api.errors import ApiError
from forecastops_api.persistence import (
    DatasetRow,
    ForecastPointRow,
    ForecastRunRow,
    ModelVersionRow,
    PromotionDecisionRow,
    TrainingRunRow,
)
from forecastops_api.repositories import Repository
from forecastops_api.runner import LocalJobRunner
from forecastops_api.schemas import (
    ApproveRequest,
    DataQualityItem,
    DatasetCreate,
    DatasetResponse,
    DatasetValidate,
    ForecastCreate,
    ForecastPointResponse,
    ForecastResponse,
    ModelResponse,
    RejectRequest,
    TrainingCreate,
    TrainingResponse,
)
from forecastops_api.settings import Settings
from forecastops_contracts import SCHEMA_VERSION
from forecastops_ml.data import load_dataset, validate_dataset
from forecastops_ml.data.quality import ValidationConfig

_FORECASTABLE = {"APPROVED", "PRODUCTION"}
_CLOUD_FAMILIES = {"deepar"}


class ForecastService:
    """Register data, train local models, and serve stored forecasts."""

    def __init__(self, repository: Repository, settings: Settings, artifact_dir: Path) -> None:
        self._repository = repository
        self._settings = settings
        self._runner = LocalJobRunner(repository, artifact_dir)

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

    def list_datasets(self) -> list[DatasetRow]:
        """Return every registered dataset."""

        return self._repository.list_datasets()

    def get_dataset(self, dataset_id: str) -> DatasetRow:
        """Return one dataset or raise a not-found error."""

        return _require_dataset(self._repository, dataset_id)

    def validate_dataset(self, dataset_id: str, body: DatasetValidate) -> DatasetRow:
        """Run the quality checks and store the report on the dataset."""

        dataset = _require_dataset(self._repository, dataset_id)
        frame, dimensions = load_dataset(Path(dataset.uri))
        as_of = body.as_of or dataset.date_max or date.today()
        report = validate_dataset(frame, dimensions, ValidationConfig(as_of=as_of))
        dataset.status = report.status
        dataset.quality_report = report.to_dict()
        self._repository.save()
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
        self._runner.train(run, frame)
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
        model.status = "APPROVED"
        model.approved_at = _now()
        model.approved_by = body.actor_id
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
        model.status = "REJECTED"
        model.rejected_by = body.actor_id
        model.rejection_reason = "human"
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

    def start_forecast(self, body: ForecastCreate) -> ForecastRunRow:
        """Queue a forecast and execute it in this process."""

        self._require_horizon(body.horizon, body.granularity)
        model = _require_model(self._repository, body.model_id)
        if model.status not in _FORECASTABLE:
            raise ApiError(
                409,
                "model_not_approved",
                "Forecasts require a model in APPROVED or PRODUCTION status.",
            )
        dataset_id = body.dataset_id or model.dataset_id
        dataset = _require_dataset(self._repository, dataset_id)
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
            created_at=_now(),
        )
        self._repository.add(forecast)
        frame, _dimensions = load_dataset(Path(dataset.uri))
        self._runner.forecast(forecast, frame, model.model_family)
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

    def forecast_series(self, forecast_id: str) -> list[ForecastPointRow]:
        """Return stored points for one forecast."""

        self.get_forecast(forecast_id)
        return self._repository.list_points(forecast_id)

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
        dataset_version=row.dataset_version,
        model_family=row.model_family,
        configuration=row.configuration,
        status=row.status,  # type: ignore[arg-type]
        started_at=row.started_at,
        finished_at=row.finished_at,
        artifact_uri=row.artifact_uri,
        metrics=row.metrics,
        git_sha=row.git_sha,
        pipeline_execution_arn=row.pipeline_execution_arn,
        created_at=row.created_at,
    )


def model_response(row: ModelVersionRow) -> ModelResponse:
    """Map a model row to the API schema."""

    return ModelResponse(
        id=row.id,
        model_family=row.model_family,
        version=row.version,
        training_run_id=row.training_run_id,
        dataset_version=row.dataset_version,
        registry_arn=row.registry_arn,
        status=row.status,  # type: ignore[arg-type]
        metrics=row.metrics,
        approved_at=row.approved_at,
        created_at=row.created_at,
    )


def forecast_response(row: ForecastRunRow, model_version: str) -> ForecastResponse:
    """Map a forecast row to the API schema."""

    return ForecastResponse(
        id=row.id,
        model_version=model_version,
        dataset_version=row.dataset_version,
        horizon=row.horizon,
        granularity=row.granularity,  # type: ignore[arg-type]
        status=row.status,  # type: ignore[arg-type]
        output_uri=row.output_uri,
        created_at=row.created_at,
    )


def point_response(row: ForecastPointRow) -> ForecastPointResponse:
    """Map a forecast point to the API schema."""

    return ForecastPointResponse(
        series_id=row.series_id,
        date=row.date,
        p10=row.p10,
        p50=row.p50,
        p90=row.p90,
        actual=row.actual,
    )


def _require_dataset(repository: Repository, dataset_id: str) -> DatasetRow:
    dataset = repository.get_dataset(dataset_id)
    if dataset is None:
        raise ApiError(404, "not_found", "Dataset was not found.")
    return dataset


def _require_model(repository: Repository, model_id: str) -> ModelVersionRow:
    model = repository.get_model(model_id)
    if model is None:
        raise ApiError(404, "not_found", "Model was not found.")
    return model


def _now() -> datetime:
    return datetime.now(UTC)
