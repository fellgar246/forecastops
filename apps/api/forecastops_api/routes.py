"""HTTP routes for the local forecast lifecycle."""

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session, sessionmaker

from forecastops_api.errors import ApiError, ErrorBody
from forecastops_api.repositories import Repository
from forecastops_api.schemas import (
    AcceptedJob,
    ApproveRequest,
    DataQualityList,
    DatasetCreate,
    DatasetList,
    DatasetResponse,
    DatasetValidate,
    ForecastCreate,
    ForecastList,
    ForecastResponse,
    ForecastSeries,
    ModelList,
    ModelPerformance,
    ModelResponse,
    RejectRequest,
    TrainingCreate,
    TrainingList,
    TrainingResponse,
)
from forecastops_api.services import (
    ForecastService,
    dataset_response,
    forecast_response,
    model_response,
    point_response,
    training_response,
)
from forecastops_api.settings import Settings

router = APIRouter()

EXPLANATIONS_DISABLED = "Explanations are not enabled."


def get_session(request: Request) -> Iterator[Session]:
    """Open a session for one request and commit when the handler succeeds."""

    factory: sessionmaker[Session] = request.app.state.session_factory
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]


def get_service(request: Request, session: SessionDep) -> ForecastService:
    """Build the forecast service for this request."""

    settings: Settings = request.app.state.settings
    artifact_dir: Path = request.app.state.artifact_dir
    return ForecastService(Repository(session), settings, artifact_dir)


ServiceDep = Annotated[ForecastService, Depends(get_service)]


@router.post("/datasets", status_code=202, response_model=AcceptedJob)
def create_dataset(body: DatasetCreate, service: ServiceDep) -> AcceptedJob:
    """Register a dataset directory."""

    dataset = service.register_dataset(body)
    return AcceptedJob(job_id=dataset.id)


@router.get("/datasets", response_model=DatasetList)
def list_datasets(service: ServiceDep) -> DatasetList:
    """List registered datasets."""

    return DatasetList(items=[dataset_response(row) for row in service.list_datasets()])


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: str, service: ServiceDep) -> DatasetResponse:
    """Return one dataset."""

    return dataset_response(service.get_dataset(dataset_id))


@router.post("/datasets/{dataset_id}/validate", response_model=DatasetResponse)
def validate_dataset(
    dataset_id: str,
    service: ServiceDep,
    body: DatasetValidate | None = None,
) -> DatasetResponse:
    """Validate a dataset and return it with the quality report."""

    return dataset_response(service.validate_dataset(dataset_id, body or DatasetValidate()))


@router.post("/training-runs", status_code=202, response_model=AcceptedJob)
def create_training_run(
    body: TrainingCreate,
    service: ServiceDep,
) -> AcceptedJob:
    """Start a training run. Poll ``GET /training-runs/{id}`` for status."""

    run = service.start_training(body)
    return AcceptedJob(job_id=run.id)


@router.get("/training-runs", response_model=TrainingList)
def list_training_runs(service: ServiceDep) -> TrainingList:
    """List training runs."""

    return TrainingList(items=[training_response(row) for row in service.list_training_runs()])


@router.get("/training-runs/{run_id}", response_model=TrainingResponse)
def get_training_run(run_id: str, service: ServiceDep) -> TrainingResponse:
    """Return one training run, including its status."""

    return training_response(service.get_training_run(run_id))


@router.get("/models", response_model=ModelList)
def list_models(service: ServiceDep) -> ModelList:
    """List model versions."""

    return ModelList(items=[model_response(row) for row in service.list_models()])


@router.get("/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: str, service: ServiceDep) -> ModelResponse:
    """Return one model version."""

    return model_response(service.get_model(model_id))


@router.post("/models/{model_id}/approve", response_model=ModelResponse)
def approve_model(
    model_id: str,
    body: ApproveRequest,
    service: ServiceDep,
) -> ModelResponse:
    """Approve a model that is waiting for a person."""

    return model_response(service.approve(model_id, body))


@router.post("/models/{model_id}/reject", response_model=ModelResponse)
def reject_model(
    model_id: str,
    body: RejectRequest,
    service: ServiceDep,
) -> ModelResponse:
    """Reject a model that is waiting for a person."""

    return model_response(service.reject(model_id, body))


@router.post("/forecasts", status_code=202, response_model=AcceptedJob)
def create_forecast(
    body: ForecastCreate,
    service: ServiceDep,
) -> AcceptedJob:
    """Start a forecast. Poll ``GET /forecasts/{id}`` for status."""

    forecast = service.start_forecast(body)
    return AcceptedJob(job_id=forecast.id)


@router.get("/forecasts", response_model=ForecastList)
def list_forecasts(service: ServiceDep) -> ForecastList:
    """List forecast runs."""

    return ForecastList(
        items=[
            forecast_response(row, service.get_model(row.model_version_id).version)
            for row in service.list_forecasts()
        ]
    )


@router.get("/forecasts/{forecast_id}", response_model=ForecastResponse)
def get_forecast(forecast_id: str, service: ServiceDep) -> ForecastResponse:
    """Return one forecast run."""

    forecast = service.get_forecast(forecast_id)
    model = service.get_model(forecast.model_version_id)
    return forecast_response(forecast, model.version)


@router.get("/forecasts/{forecast_id}/series", response_model=ForecastSeries)
def get_forecast_series(
    forecast_id: str,
    service: ServiceDep,
) -> ForecastSeries:
    """Return forecast points, including ``p50``."""

    points = [point_response(row) for row in service.forecast_series(forecast_id)]
    return ForecastSeries(items=points)


@router.post(
    "/forecasts/{forecast_id}/explanation",
    status_code=409,
    response_model=ErrorBody,
)
def create_explanation(forecast_id: str) -> ErrorBody:
    """Explanations are not available yet."""

    _ = forecast_id
    raise ApiError(409, "explanations_disabled", EXPLANATIONS_DISABLED)


@router.get(
    "/forecasts/{forecast_id}/explanation",
    status_code=409,
    response_model=ErrorBody,
)
def get_explanation(forecast_id: str) -> ErrorBody:
    """Explanations are not available yet."""

    _ = forecast_id
    raise ApiError(409, "explanations_disabled", EXPLANATIONS_DISABLED)


@router.get("/metrics/model-performance", response_model=ModelPerformance)
def model_performance(service: ServiceDep) -> ModelPerformance:
    """Return stored model metrics."""

    return ModelPerformance(items=[model_response(row) for row in service.model_performance()])


@router.get("/metrics/data-quality", response_model=DataQualityList)
def data_quality(service: ServiceDep) -> DataQualityList:
    """Return stored dataset quality reports."""

    return DataQualityList(items=service.data_quality())


@router.post("/admin/retrain", status_code=202, response_model=AcceptedJob)
def retrain(service: ServiceDep) -> AcceptedJob:
    """Train the production family again without approving the new model."""

    run = service.retrain()
    return AcceptedJob(job_id=run.id)
