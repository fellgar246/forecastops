"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from forecastops_api.artifacts import select_artifact_store
from forecastops_api.errors import install_error_handlers
from forecastops_api.forecast_jobs import select_batch_inference
from forecastops_api.health import router as health_router
from forecastops_api.logging import configure_logging
from forecastops_api.registry import select_model_registry
from forecastops_api.routes import router as forecast_router
from forecastops_api.settings import Settings, get_settings


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and refuse to start when settings are invalid."""

    configure_logging()
    settings = get_settings()
    log = structlog.get_logger()
    log.info(
        "api.started",
        execution_mode=settings.execution_mode.value,
        aws_enabled=settings.aws_enabled,
        bedrock_enabled=settings.bedrock_enabled,
        sagemaker_enabled=settings.sagemaker_enabled,
    )
    yield
    log.info("api.stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API application."""

    resolved = settings if settings is not None else get_settings()
    application = FastAPI(title="ForecastOps API", version="0.1.0", lifespan=lifespan)
    application.state.settings = resolved
    application.state.artifact_dir = resolved.artifact_dir
    application.state.artifact_store = select_artifact_store(resolved)
    application.state.model_registry = select_model_registry(resolved)
    application.state.batch_inference = select_batch_inference(resolved)
    application.state.execute_forecasts_inline = True
    application.state.forecast_predictor = None
    engine = create_engine(resolved.database_url)
    application.state.engine = engine
    application.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[resolved.web_origin],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )
    install_error_handlers(application)
    application.include_router(health_router)
    application.include_router(forecast_router)
    return application


app = create_app()
