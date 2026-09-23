"""Liveness response for the local and cloud profiles."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from forecastops_api.settings import ExecutionMode, get_settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Process status and the cloud flags that affect this process."""

    status: Literal["healthy"]
    execution_mode: ExecutionMode
    aws_enabled: bool
    bedrock_enabled: bool
    sagemaker_enabled: bool


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the API process is up and which cloud flags are set."""

    settings = get_settings()
    return HealthResponse(
        status="healthy",
        execution_mode=settings.execution_mode,
        aws_enabled=settings.aws_enabled,
        bedrock_enabled=settings.bedrock_enabled,
        sagemaker_enabled=settings.sagemaker_enabled,
    )
