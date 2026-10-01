"""Process configuration loaded from the environment."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExecutionMode(StrEnum):
    """Where the platform runs."""

    LOCAL = "local"
    AWS = "aws"


class Settings(BaseSettings):
    """Required runtime flags and cost ceilings.

    Contract fields have no defaults. A missing value fails startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    execution_mode: ExecutionMode
    aws_enabled: bool
    # Local mode may leave this false and stamp every row with tenant "local".
    # Cloud mode requires it, and then Cognito validates each bearer token.
    auth_enabled: bool
    bedrock_enabled: bool
    sagemaker_enabled: bool
    online_inference: bool
    ai_enabled: bool
    training_enabled: bool
    aws_ml_enabled: bool

    max_bedrock_calls_per_day: int = Field(gt=0)
    max_bedrock_input_tokens: int = Field(gt=0)
    max_bedrock_output_tokens: int = Field(gt=0)
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    max_training_jobs_per_day: int = Field(ge=0)
    max_training_runtime_minutes: int = Field(gt=0)
    allow_gpu_training: bool
    max_batch_inference_jobs_per_day: int = Field(ge=0)
    max_dataset_rows_demo: int = Field(gt=0)
    max_forecast_horizon_days: int = Field(gt=0)

    random_seed: int
    # PSI above psi_retrain_threshold recommends retraining. PSI above
    # psi_warning_threshold and at or below that value is a warning.
    psi_warning_threshold: float = Field(default=0.1, gt=0)
    psi_retrain_threshold: float = Field(default=0.2, gt=0)
    wape_degradation_limit: float = Field(default=0.2, gt=0)
    max_dataset_age_days: int = Field(default=30, ge=0)
    frequency_warning_delta: float = Field(default=0.05, ge=0)
    recent_window_days: int = Field(default=28, gt=0)
    # Cognito user pool that issues access tokens. Required when auth is enabled.
    cognito_user_pool_id: str = ""
    cognito_app_client_id: str = ""
    database_url: str = "postgresql+psycopg://forecastops:forecastops@localhost:5432/forecastops"
    # DynamoDB table for domain documents. Required when EXECUTION_MODE is aws.
    # Local mode leaves this empty and does not open a cloud client.
    metadata_table_name: str = ""
    web_origin: str = "http://localhost:3000"
    artifact_dir: Path = Path("var/artifacts")
    artifacts_bucket: str = ""
    aws_region: str = "us-east-1"

    @model_validator(mode="after")
    def drift_thresholds_are_ordered(self) -> Self:
        """Keep the warning band below the retrain threshold."""

        if self.psi_warning_threshold >= self.psi_retrain_threshold:
            raise ValueError("psi_warning_threshold must be below psi_retrain_threshold.")
        if self.execution_mode is ExecutionMode.AWS and not self.auth_enabled:
            raise ValueError("AUTH_ENABLED must be true when EXECUTION_MODE is aws.")
        if self.auth_enabled and (
            not self.cognito_user_pool_id.strip() or not self.cognito_app_client_id.strip()
        ):
            raise ValueError(
                "COGNITO_USER_POOL_ID and COGNITO_APP_CLIENT_ID are required when "
                "AUTH_ENABLED is true."
            )
        return self


def load_settings() -> Settings:
    """Load settings or stop the process with an English explanation."""

    try:
        # Field values come from the environment. The type checker cannot see them.
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        lines = ["Startup failed because the environment configuration is invalid."]
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "settings"
            lines.append(f"- {location}: {error['msg']}")
        raise SystemExit("\n".join(lines)) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process settings, loaded once."""

    return load_settings()
