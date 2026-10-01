"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from forecastops_api.artifacts import select_artifact_store
from forecastops_api.auth import authentication_required, build_token_verifier
from forecastops_api.errors import install_error_handlers
from forecastops_api.forecast_jobs import select_batch_inference
from forecastops_api.health import router as health_router
from forecastops_api.logging import configure_logging
from forecastops_api.observability import CorrelationIdMiddleware, configure_metrics
from forecastops_api.observability import router as metrics_router
from forecastops_api.registry import select_model_registry
from forecastops_api.routes import router as forecast_router
from forecastops_api.settings import ExecutionMode, Settings, get_settings


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and refuse to start when settings are invalid."""

    configure_logging()
    settings = get_settings()
    configure_metrics(settings)
    log = structlog.get_logger()
    log.info(
        "api.started",
        execution_mode=settings.execution_mode.value,
        aws_enabled=settings.aws_enabled,
        auth_enabled=settings.auth_enabled,
        bedrock_enabled=settings.bedrock_enabled,
        sagemaker_enabled=settings.sagemaker_enabled,
    )
    yield
    log.info("api.stopped")


def create_app(
    settings: Settings | None = None,
    *,
    metadata_table: object | None = None,
) -> FastAPI:
    """Build the API application."""

    resolved = settings if settings is not None else get_settings()
    application = FastAPI(title="ForecastOps API", version="0.1.0", lifespan=lifespan)
    application.state.settings = resolved
    application.state.artifact_dir = resolved.artifact_dir
    application.state.artifact_store = select_artifact_store(resolved)
    application.state.model_registry = select_model_registry(resolved)
    application.state.batch_inference = select_batch_inference(resolved)
    application.state.token_verifier = (
        build_token_verifier(resolved) if authentication_required(resolved) else None
    )
    application.state.execute_forecasts_inline = True
    application.state.forecast_predictor = None
    _configure_metadata(application, resolved, metadata_table)
    configure_logging()
    configure_metrics(resolved)
    application.add_middleware(CorrelationIdMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[resolved.web_origin],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Accept", "X-Correlation-Id"],
        expose_headers=["X-Correlation-Id"],
    )
    install_error_handlers(application)
    application.include_router(health_router)
    application.include_router(metrics_router)
    application.include_router(forecast_router)
    return application


def _configure_metadata(
    application: FastAPI,
    settings: Settings,
    metadata_table: object | None,
) -> None:
    """Select PostgreSQL or DynamoDB. Local mode does not open a cloud client."""

    if settings.execution_mode is ExecutionMode.AWS:
        if metadata_table is None:
            from forecastops_api.dynamodb import open_metadata_table

            metadata_table = open_metadata_table(settings)
        application.state.metadata_backend = "dynamodb"
        application.state.metadata_table = metadata_table
        application.state.engine = None
        application.state.session_factory = None
        return
    if metadata_table is not None:
        raise ValueError("A metadata table is only used when EXECUTION_MODE is aws.")
    engine = create_engine(settings.database_url)
    application.state.metadata_backend = "sql"
    application.state.metadata_table = None
    application.state.engine = engine
    application.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)


app = create_app()
