"""Structured forecast package and the explanation written from it."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ForecastWindow(BaseModel):
    """Quantile totals for the scope being explained."""

    model_config = ConfigDict(extra="forbid")

    horizon_weeks: float
    p10: float | None = None
    p50: float
    p90: float | None = None


class HistoryWindow(BaseModel):
    """Recent demand used as context. Missing history stays null."""

    model_config = ConfigDict(extra="forbid")

    previous_period_units: float | None = None
    year_over_year_pct: float | None = None


class Signal(BaseModel):
    """One named signal from the forecast package.

    ``kind`` is ``driver`` for a tree association and ``context`` for a
    global-model field. Context is not a cause.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    direction: Literal["positive", "negative"] | None = None
    value: float | None = None
    kind: Literal["driver", "context"] = "context"


class ExplanationPackage(BaseModel):
    """The only object an explanation adapter may read."""

    model_config = ConfigDict(extra="forbid")

    scope: dict[str, str]
    forecast: ForecastWindow
    history: HistoryWindow
    signals: list[Signal]
    model_metrics: dict[str, float] = Field(default_factory=dict)
    positive_drivers: list[str] = Field(default_factory=list)
    negative_drivers: list[str] = Field(default_factory=list)

    def prompt_payload(self) -> dict[str, object]:
        """Return the package JSON, omitting empty driver lists."""

        payload: dict[str, object] = dict(self.model_dump(mode="json"))
        if not payload["positive_drivers"]:
            del payload["positive_drivers"]
        if not payload["negative_drivers"]:
            del payload["negative_drivers"]
        return payload


class ExplanationDraft(BaseModel):
    """Untrusted explanation text before the deterministic checks."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    drivers: list[str]
    risks: list[str]
    uncertainty_note: str
    recommended_checks: list[str]


class ExplanationUsage(BaseModel):
    """Token counts and latency for one explanation call."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
