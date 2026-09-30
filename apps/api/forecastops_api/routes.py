"""HTTP routes for the local forecast lifecycle."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, sessionmaker

from forecastops_api.artifacts import ArtifactStore
from forecastops_api.explanations import ExplanationService
from forecastops_api.repositories import Repository
from forecastops_api.schemas import (
    AcceptedJob,
    ApproveRequest,
    DataQualityList,
    DatasetCatalog,
    DatasetCreate,
    DatasetList,
    DatasetResponse,
    DatasetUploadResponse,
    DatasetValidate,
    ForecastCreate,
    ForecastList,
    ForecastResponse,
    ForecastSeries,
    ModelList,
    ModelPerformance,
    ModelResponse,
    PresignedUploadRequest,
    PresignedUploadResponse,
    RejectRequest,
    TrainingCreate,
    TrainingList,
    TrainingResponse,
)
from forecastops_api.services import (
    ForecastService,
    dataset_response,
    training_response,
)
from forecastops_api.settings import Settings
from forecastops_ml.registry import ModelRegistry

router = APIRouter()


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
    artifacts: ArtifactStore = request.app.state.artifact_store
    registry: ModelRegistry | None = request.app.state.model_registry
    return ForecastService(
        Repository(session),
        settings,
        artifacts,
        registry,
        batch=getattr(request.app.state, "batch_inference", None),
        predictor=getattr(request.app.state, "forecast_predictor", None),
    )


ServiceDep = Annotated[ForecastService, Depends(get_service)]


def get_explanations(request: Request, session: SessionDep) -> ExplanationService:
    """Build the explanation service for this request."""

    client = getattr(request.app.state, "explanation_client", None)
    return ExplanationService(Repository(session), request.app.state.settings, client)


ExplanationDep = Annotated[ExplanationService, Depends(get_explanations)]


@router.post("/datasets", status_code=202, response_model=AcceptedJob)
def create_dataset(body: DatasetCreate, service: ServiceDep) -> AcceptedJob:
    """Register a dataset directory."""

    dataset = service.register_dataset(body)
    return AcceptedJob(job_id=dataset.id)


@router.post("/datasets/uploads", status_code=201, response_model=DatasetUploadResponse)
def upload_dataset(
    service: ServiceDep,
    file: Annotated[UploadFile, File()],
) -> DatasetUploadResponse:
    """Store a dataset file. Local mode accepts the bytes directly."""

    filename = file.filename or ""
    body = file.file.read()
    payload = body if isinstance(body, bytes) else bytes(body)
    return service.store_uploaded_dataset(filename, payload)


@router.post("/datasets/upload-url", response_model=PresignedUploadResponse)
def create_presigned_dataset_upload(
    body: PresignedUploadRequest,
    service: ServiceDep,
) -> PresignedUploadResponse:
    """Issue a pre-signed URL when remote artifact storage is enabled."""

    return service.issue_presigned_upload(body.name)


@router.get("/datasets", response_model=DatasetList)
def list_datasets(service: ServiceDep) -> DatasetList:
    """List registered datasets."""

    return DatasetList(items=[dataset_response(row) for row in service.list_datasets()])


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: str, service: ServiceDep) -> DatasetResponse:
    """Return one dataset."""

    return dataset_response(service.get_dataset(dataset_id))


@router.get("/datasets/{dataset_id}/catalog", response_model=DatasetCatalog)
def dataset_catalog(dataset_id: str, service: ServiceDep) -> DatasetCatalog:
    """Return stores, categories, and SKUs for forecast filters."""

    return service.catalog(dataset_id)


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

    return ModelList(items=[service.present_model(row) for row in service.list_models()])


@router.get("/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: str, service: ServiceDep) -> ModelResponse:
    """Return one model version."""

    return service.present_model(service.get_model(model_id))


@router.post("/models/{model_id}/approve", response_model=ModelResponse)
def approve_model(
    model_id: str,
    body: ApproveRequest,
    service: ServiceDep,
) -> ModelResponse:
    """Approve a model that is waiting for a person."""

    return service.present_model(service.approve(model_id, body))


@router.post("/models/{model_id}/reject", response_model=ModelResponse)
def reject_model(
    model_id: str,
    body: RejectRequest,
    service: ServiceDep,
) -> ModelResponse:
    """Reject a model that is waiting for a person."""

    return service.present_model(service.reject(model_id, body))


@router.post("/forecasts", status_code=202, response_model=AcceptedJob)
def create_forecast(
    body: ForecastCreate,
    service: ServiceDep,
    request: Request,
) -> AcceptedJob:
    """Queue a forecast. Poll ``GET /forecasts/{id}`` for status."""

    execute = bool(getattr(request.app.state, "execute_forecasts_inline", True))
    forecast = service.start_forecast(body, execute=execute)
    return AcceptedJob(job_id=forecast.id)


@router.get("/forecasts", response_model=ForecastList)
def list_forecasts(service: ServiceDep) -> ForecastList:
    """List forecast runs."""

    return ForecastList(items=[service.present_forecast(row) for row in service.list_forecasts()])


@router.get("/forecasts/{forecast_id}", response_model=ForecastResponse)
def get_forecast(forecast_id: str, service: ServiceDep) -> ForecastResponse:
    """Return one forecast run."""

    return service.present_forecast(service.get_forecast(forecast_id))


@router.get("/forecasts/{forecast_id}/series", response_model=ForecastSeries)
def get_forecast_series(
    forecast_id: str,
    service: ServiceDep,
    category: Annotated[str | None, Query()] = None,
    sku: Annotated[str | None, Query()] = None,
    store: Annotated[str | None, Query()] = None,
    horizon: Annotated[int | None, Query()] = None,
) -> ForecastSeries:
    """Return forecast points for the requested category, SKU, store, and horizon."""

    return service.forecast_series(
        forecast_id,
        category=_blank_to_none(category),
        sku=_blank_to_none(sku),
        store=_blank_to_none(store),
        horizon=horizon,
    )


@router.post("/forecasts/{forecast_id}/explanation")
def create_explanation(
    forecast_id: str,
    service: ExplanationDep,
    category: Annotated[str | None, Query()] = None,
    sku: Annotated[str | None, Query()] = None,
    store: Annotated[str | None, Query()] = None,
) -> JSONResponse:
    """Write an explanation for a finished forecast, or return the cached one."""

    result = service.request(
        forecast_id,
        category=_blank_to_none(category),
        sku=_blank_to_none(sku),
        store=_blank_to_none(store),
    )
    return JSONResponse(status_code=result.status_code, content=result.content)


@router.get("/forecasts/{forecast_id}/explanation")
def get_explanation(
    forecast_id: str,
    service: ExplanationDep,
    category: Annotated[str | None, Query()] = None,
    sku: Annotated[str | None, Query()] = None,
    store: Annotated[str | None, Query()] = None,
) -> JSONResponse:
    """Return the stored explanation for this forecast."""

    result = service.read(
        forecast_id,
        category=_blank_to_none(category),
        sku=_blank_to_none(sku),
        store=_blank_to_none(store),
    )
    return JSONResponse(status_code=result.status_code, content=result.content)


@router.get("/metrics/model-performance", response_model=ModelPerformance)
def model_performance(service: ServiceDep) -> ModelPerformance:
    """Return stored model metrics."""

    return ModelPerformance(
        items=[service.present_model(row) for row in service.model_performance()]
    )


@router.get("/metrics/data-quality", response_model=DataQualityList)
def data_quality(service: ServiceDep) -> DataQualityList:
    """Return stored dataset quality reports."""

    return DataQualityList(items=service.data_quality())


def _blank_to_none(value: str | None) -> str | None:
    if value is None or value.strip() == "":
        return None
    return value


@router.post("/admin/retrain", status_code=202, response_model=AcceptedJob)
def retrain(service: ServiceDep) -> AcceptedJob:
    """Train the production family again without approving the new model."""

    run = service.retrain()
    return AcceptedJob(job_id=run.id)
