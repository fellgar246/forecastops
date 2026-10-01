"""Optional explanation judge. It is off unless a caller turns it on.

Scores are readability, groundedness, and actionability, each from 1 to 5.
They never decide whether the evaluation suite passes. The judge calls
Bedrock only when that integration is enabled and a caller asks for scores.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

from forecastops_ml.explanations.models import ExplanationDraft, ExplanationPackage

_JUDGE_INSTRUCTION = (
    "Score the explanation on readability, groundedness, and actionability. "
    "Each score is an integer from 1 to 5. "
    "Return JSON with keys readability, groundedness, and actionability. "
    "Do not change the forecast."
)


class InvokeModel(Protocol):
    """Subset of the Bedrock runtime used to score one explanation."""

    def invoke_model(self, **kwargs: object) -> Mapping[str, object]:
        """Invoke one model and return the provider response."""


@runtime_checkable
class SupportsRead(Protocol):
    """Response body that yields bytes or text."""

    def read(self) -> bytes | str:
        """Return the response body."""


@dataclass(frozen=True)
class JudgeScores:
    """One scenario's optional quality scores. Each value is 1 to 5."""

    readability: int
    groundedness: int
    actionability: int

    def __post_init__(self) -> None:
        _score("readability", self.readability)
        _score("groundedness", self.groundedness)
        _score("actionability", self.actionability)


class BedrockExplanationJudge:
    """Ask one Bedrock model for the three optional scores.

    Pass ``runtime`` in tests so this class does not open a client.
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
        if runtime is None:
            import boto3

            client = boto3.client("bedrock-runtime", region_name=region)
            self._runtime: InvokeModel = cast(InvokeModel, client)
        else:
            self._runtime = runtime

    def score(self, package: ExplanationPackage, draft: ExplanationDraft) -> JudgeScores:
        """Return the three scores for one explanation."""

        prompt = _prompt(package, draft)
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
        return _parse_scores(_message_text(_read_payload(response)))


def _prompt(package: ExplanationPackage, draft: ExplanationDraft) -> str:
    payload = json.dumps(
        {
            "draft": draft.model_dump(mode="json"),
            "package": package.prompt_payload(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{_JUDGE_INSTRUCTION}\n{payload}\n"


def _parse_scores(text: str) -> JudgeScores:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.startswith("```")]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The judge response did not include scores.")
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("The judge response did not include scores.") from exc
    if not isinstance(payload, dict):
        raise ValueError("The judge response did not include scores.")
    data = cast(dict[str, object], payload)
    return JudgeScores(
        readability=_score("readability", data.get("readability")),
        groundedness=_score("groundedness", data.get("groundedness")),
        actionability=_score("actionability", data.get("actionability")),
    )


def _score(label: str, value: object) -> int:
    if type(value) is not int or not 1 <= value <= 5:
        raise ValueError(f"{label} must be an integer from 1 to 5.")
    return value


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
        raise ValueError("The judge response did not include scores.")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("The judge response did not include scores.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("The judge response did not include scores.")
    return cast(dict[str, object], parsed)


def _message_text(payload: Mapping[str, object]) -> str:
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("The judge response did not include scores.")
    first = content[0]
    if not isinstance(first, Mapping):
        raise ValueError("The judge response did not include scores.")
    text = first.get("text")
    if not isinstance(text, str):
        raise ValueError("The judge response did not include scores.")
    return text
