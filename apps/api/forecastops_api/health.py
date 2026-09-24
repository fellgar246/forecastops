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


class AwsHealthResponse(BaseModel):
    """Each cloud integration and whether this process may call it."""

    aws_enabled: bool
    aws_ml_enabled: bool
    bedrock_enabled: bool
    sagemaker_enabled: bool
    online_inference: bool


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


@router.get("/health/aws", response_model=AwsHealthResponse)
def health_aws() -> AwsHealthResponse:
    """Report whether each cloud integration is enabled."""

    settings = get_settings()
    return AwsHealthResponse(
        aws_enabled=settings.aws_enabled,
        aws_ml_enabled=settings.aws_ml_enabled,
        bedrock_enabled=settings.bedrock_enabled,
        sagemaker_enabled=settings.sagemaker_enabled,
        online_inference=settings.online_inference,
    )
