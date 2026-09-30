"""Explanation adapters. The mock stays on the machine. Bedrock is optional."""

import json
import time
from collections.abc import Mapping
from typing import Protocol, cast, runtime_checkable

from forecastops_ml.explanations.models import (
    ExplanationDraft,
    ExplanationPackage,
    ExplanationUsage,
)
from forecastops_ml.explanations.prompt import render_prompt
from forecastops_ml.explanations.validation import format_number, parse_draft, reject_draft


class ExplanationClient(Protocol):
    """Write one draft from a forecast package."""

    model_id: str

    def explain(self, package: ExplanationPackage) -> ExplanationDraft:
        """Return a draft. The caller validates it."""

    @property
    def usage(self) -> ExplanationUsage:
        """Token counts and latency for the latest call."""


@runtime_checkable
class SupportsRead(Protocol):
    """Response body that yields bytes or text."""

    def read(self) -> bytes | str:
        """Return the response body."""


class InvokeModel(Protocol):
    """Subset of the Bedrock runtime used to invoke an explanation model."""

    def invoke_model(self, **kwargs: object) -> Mapping[str, object]:
        """Invoke one model and return the provider response."""


class MockExplanationClient:
    """Schema-valid explanation built only from the package fields.

    This adapter does not open a network client.
    """

    model_id = "mock-explainer"

    def __init__(self) -> None:
        self._usage = ExplanationUsage(input_tokens=0, output_tokens=0, latency_ms=0)

    def explain(self, package: ExplanationPackage) -> ExplanationDraft:
        started = time.perf_counter()
        prompt = render_prompt(package)
        draft = _draft_from_package(package)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self._usage = ExplanationUsage(
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(draft.model_dump_json()),
            latency_ms=elapsed_ms,
        )
        return draft

    @property
    def usage(self) -> ExplanationUsage:
        return self._usage


class BedrockExplanationClient:
    """Call one Bedrock model and parse its JSON draft.

    The model id comes from settings. Construction is the remote-client
    boundary: pass ``runtime`` in tests so this class does not open a client.
    """

    def __init__(
        self,
        model_id: str,
        *,
        max_output_tokens: int,
        region: str,
        runtime: InvokeModel | None = None,
    ) -> None:
        if model_id == "":
            raise ValueError("Bedrock model id is required.")
        self.model_id = model_id
        self._max_output_tokens = max_output_tokens
        self._usage = ExplanationUsage(input_tokens=0, output_tokens=0, latency_ms=0)
        if runtime is None:
            import boto3

            client = boto3.client("bedrock-runtime", region_name=region)
            self._runtime: InvokeModel = cast(InvokeModel, client)
        else:
            self._runtime = runtime

    def explain(self, package: ExplanationPackage) -> ExplanationDraft:
        started = time.perf_counter()
        prompt = render_prompt(package)
        response = self._runtime.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self._max_output_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
            ).encode(),
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        payload = _read_payload(response)
        usage = payload.get("usage")
        input_tokens = _usage_count(usage, "input_tokens", estimate_tokens(prompt))
        output_tokens = _usage_count(usage, "output_tokens", 0)
        self._usage = ExplanationUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
        )
        return parse_draft(_json_object(_message_text(payload)))

    @property
    def usage(self) -> ExplanationUsage:
        return self._usage


def select_explanation_client(
    *,
    bedrock_enabled: bool,
    model_id: str,
    max_output_tokens: int,
    region: str,
    runtime: InvokeModel | None = None,
) -> ExplanationClient:
    """Return the Bedrock client only when that integration is enabled."""

    if bedrock_enabled:
        return BedrockExplanationClient(
            model_id,
            max_output_tokens=max_output_tokens,
            region=region,
            runtime=runtime,
        )
    return MockExplanationClient()


def estimate_tokens(text: str) -> int:
    """Estimate tokens from text length. The same estimate gates the request."""

    return max(1, (len(text) + 3) // 4)


def _draft_from_package(package: ExplanationPackage) -> ExplanationDraft:
    sentences = [f"The median forecast is {format_number(package.forecast.p50)}."]
    if package.forecast.p10 is not None and package.forecast.p90 is not None:
        low = format_number(package.forecast.p10)
        high = format_number(package.forecast.p90)
        sentences.append(f"The lower estimate is {low} and the upper estimate is {high}.")
    if package.history.previous_period_units is not None:
        previous = format_number(package.history.previous_period_units)
        sentences.append(f"The previous period total is {previous}.")
    if package.history.year_over_year_pct is not None:
        change = format_number(package.history.year_over_year_pct)
        sentences.append(f"The year over year change is {change} percent.")
    if package.forecast.p10 is not None and package.forecast.p90 is not None:
        risks = ["The lower and upper estimates show the remaining range."]
    else:
        risks = ["This forecast does not include a lower and upper estimate."]
    return ExplanationDraft(
        summary=" ".join(sentences),
        drivers=[signal.name for signal in package.signals],
        risks=risks,
        uncertainty_note="The forecast remains uncertain.",
        recommended_checks=["Compare the median forecast with the previous period before acting."],
    )


def _read_payload(response: Mapping[str, object]) -> dict[str, object]:
    body = response.get("body")
    if isinstance(body, SupportsRead):
        raw: object = body.read()
    else:
        raw = body
    if isinstance(raw, bytes):
        text = raw.decode()
    elif isinstance(raw, str):
        text = raw
    else:
        reject_draft("schema", "The explanation failed the schema check.")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        reject_draft("schema", "The explanation failed the schema check.", exc)
    if not isinstance(parsed, dict):
        reject_draft("schema", "The explanation failed the schema check.")
    return cast(dict[str, object], parsed)


def _message_text(payload: Mapping[str, object]) -> str:
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        reject_draft("schema", "The explanation failed the schema check.")
    first = content[0]
    if not isinstance(first, Mapping):
        reject_draft("schema", "The explanation failed the schema check.")
    text = first.get("text")
    if not isinstance(text, str):
        reject_draft("schema", "The explanation failed the schema check.")
    return text


def _json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.startswith("```")]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        reject_draft("schema", "The explanation failed the schema check.")
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        reject_draft("schema", "The explanation failed the schema check.", exc)
    if not isinstance(parsed, dict):
        reject_draft("schema", "The explanation failed the schema check.")
    return cast(dict[str, object], parsed)


def _usage_count(usage: object, key: str, default: int) -> int:
    if not isinstance(usage, dict):
        return default
    value = usage.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return max(0, int(value))
