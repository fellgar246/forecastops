"""Cost ceilings checked before a job starts.

Each check raises :class:`forecastops_api.errors.ApiError` when the request
would cross a ceiling or a kill switch. Callers do not start the job after that.
"""

from collections.abc import Mapping

from forecastops_api.errors import ApiError
from forecastops_api.settings import ExecutionMode, Settings
from forecastops_ml.inference import BatchCapacityError, admit_forecast

_RESET = "00:00 UTC"
_CLOUD_FAMILY = "deepar"


def require_training_enabled(settings: Settings) -> None:
    """Refuse a training run when the training switch is off."""

    if not settings.training_enabled:
        raise ApiError(409, "training_disabled", "Training is disabled.")


def require_cpu_training(settings: Settings, configuration: Mapping[str, object]) -> None:
    """Refuse GPU training. The ceiling keeps that switch off."""

    if settings.allow_gpu_training or _requests_gpu(configuration):
        raise ApiError(409, "gpu_training_disabled", "GPU training is not allowed.")


def require_aws_ml(settings: Settings) -> None:
    """Refuse a cloud training or batch job when that switch is off."""

    if not settings.aws_ml_enabled:
        raise ApiError(
            409,
            "aws_ml_disabled",
            "Cloud training and batch jobs are disabled.",
        )


def require_bedrock_provider(settings: Settings) -> None:
    """Refuse a remote explanation call when the provider switch is off.

    The local mock stays available, so a forecast can still be explained
    without opening a remote client.
    """

    if not settings.bedrock_enabled:
        raise ApiError(
            409,
            "bedrock_disabled",
            "Language-model explanations are disabled.",
        )


def require_training_capacity(settings: Settings, jobs_started_today: int) -> None:
    """Refuse another training run after the daily ceiling."""

    limit = settings.max_training_jobs_per_day
    if jobs_started_today >= limit:
        raise ApiError(
            429,
            "training_limit",
            (f"The daily training limit of {limit} jobs has been reached. It resets at {_RESET}."),
            {"max_training_jobs_per_day": limit},
        )


def require_dataset_rows(settings: Settings, row_count: int) -> None:
    """Refuse a dataset larger than the demo ceiling."""

    limit = settings.max_dataset_rows_demo
    if row_count > limit:
        raise ApiError(
            422,
            "dataset_row_limit",
            f"Dataset cannot exceed {limit} rows.",
            {"row_count": row_count, "max_dataset_rows_demo": limit},
        )


def require_forecast_horizon(settings: Settings, horizon_days: int) -> None:
    """Refuse a horizon longer than the ceiling."""

    limit = settings.max_forecast_horizon_days
    if horizon_days > limit:
        raise ApiError(
            422,
            "horizon_too_long",
            f"Forecast horizon cannot exceed {limit} days.",
            {"horizon_days": horizon_days, "max_forecast_horizon_days": limit},
        )


def require_batch_capacity(settings: Settings, jobs_started_today: int) -> None:
    """Refuse another forecast after the daily batch ceiling."""

    limit = settings.max_batch_inference_jobs_per_day
    try:
        admit_forecast(jobs_started_today=jobs_started_today, daily_limit=limit, replay=False)
    except BatchCapacityError as exc:
        raise ApiError(
            429,
            "batch_inference_limit",
            str(exc),
            {"max_batch_inference_jobs_per_day": limit},
        ) from exc


def require_explanation_calls(settings: Settings, calls_today: int) -> None:
    """Refuse another explanation after the daily call ceiling."""

    limit = settings.max_bedrock_calls_per_day
    if calls_today >= limit:
        raise ApiError(
            429,
            "explanation_quota",
            (
                f"The daily explanation limit of {limit} calls has been reached. "
                f"It resets at {_RESET}."
            ),
        )


def require_explanation_input_tokens(settings: Settings, tokens: int) -> None:
    """Refuse an explanation whose input is over the token ceiling."""

    _require_tokens(settings.max_bedrock_input_tokens, tokens, kind="input")


def require_explanation_output_tokens(settings: Settings, tokens: int) -> None:
    """Refuse an explanation whose output is over the token ceiling."""

    _require_tokens(settings.max_bedrock_output_tokens, tokens, kind="output")


def cloud_batch_requested(
    *,
    model_family: str,
    batch_configured: bool,
    execution_mode: ExecutionMode,
    sagemaker_enabled: bool,
) -> bool:
    """Return whether this forecast would start a cloud batch job."""

    if model_family != _CLOUD_FAMILY:
        return False
    if batch_configured:
        return True
    return execution_mode is ExecutionMode.AWS and sagemaker_enabled


def _require_tokens(limit: int, tokens: int, *, kind: str) -> None:
    if tokens > limit:
        raise ApiError(
            429,
            "explanation_quota",
            f"The explanation exceeds the {kind} token ceiling of {limit}. It resets at {_RESET}.",
        )


def _requests_gpu(configuration: Mapping[str, object]) -> bool:
    for key in ("gpu", "use_gpu", "allow_gpu"):
        if configuration.get(key) is True:
            return True
    instance = configuration.get("instance_type")
    if not isinstance(instance, str):
        return False
    lowered = instance.strip().lower()
    return "gpu" in lowered or lowered.startswith(("ml.g", "ml.p"))
