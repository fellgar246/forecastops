"""Cost posture for the current environment.

Estimated spend stays empty until a billing API is configured. The monthly
budget is the project ceiling, not an invoice.
"""

from datetime import UTC, datetime
from pathlib import Path

from forecastops_api.repositories import MetadataRepository
from forecastops_api.schemas import CostResponse
from forecastops_api.settings import Settings

BILLING_NOT_CONFIGURED = "Billing is not configured."


class CostService:
    """Assemble usage, switches, and the last cleanup time."""

    def __init__(self, repository: MetadataRepository, settings: Settings) -> None:
        self._repository = repository
        self._settings = settings

    def snapshot(self, *, now: datetime | None = None) -> CostResponse:
        """Return the figures shown on the cost page."""

        moment = now or datetime.now(UTC)
        usage = self._repository.explanation_usage_on(moment.date())
        training_runs = sum(
            1 for row in self._repository.list_training_runs() if _in_month(row.created_at, moment)
        )
        return CostResponse(
            monthly_budget_usd=self._settings.monthly_budget_usd,
            training_runs_this_month=training_runs,
            explanation_calls_today=usage.calls,
            approximate_explanation_tokens=usage.input_tokens + usage.output_tokens,
            estimated_spend_usd=None,
            spend_note=BILLING_NOT_CONFIGURED,
            aws_enabled=self._settings.aws_enabled,
            aws_ml_enabled=self._settings.aws_ml_enabled,
            bedrock_enabled=self._settings.bedrock_enabled,
            sagemaker_enabled=self._settings.sagemaker_enabled,
            training_enabled=self._settings.training_enabled,
            online_inference=self._settings.online_inference,
            last_cleanup_at=read_last_cleanup(self._settings.cost_state_file),
        )


def read_last_cleanup(path: Path) -> datetime | None:
    """Return the cleanup timestamp, or none when it is missing or unreadable."""

    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if text == "":
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _in_month(value: datetime, moment: datetime) -> bool:
    current = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    current = current.astimezone(UTC)
    stamp = moment.astimezone(UTC)
    return current.year == stamp.year and current.month == stamp.month
