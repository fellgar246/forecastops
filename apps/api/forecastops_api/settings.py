"""Process configuration loaded from the environment."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, ValidationError
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
    database_url: str = "postgresql+psycopg://forecastops:forecastops@localhost:5432/forecastops"
    web_origin: str = "http://localhost:3000"
    artifact_dir: Path = Path("var/artifacts")
    artifacts_bucket: str = ""
    aws_region: str = "us-east-1"


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
