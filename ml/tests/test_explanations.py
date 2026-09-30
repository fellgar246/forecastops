"""Golden checks for grounded forecast explanations."""

import io
import json
from collections.abc import Mapping

import pytest

from forecastops_ml.explanations import (
    PROMPT_VERSION,
    BedrockExplanationClient,
    ExplanationValidationError,
    MockExplanationClient,
    build_explanation_package,
    explanation_validation_failures,
    render_prompt,
    select_explanation_client,
    validate_explanation,
)
from forecastops_ml.explanations.models import ExplanationDraft
from forecastops_ml.explanations.validation import parse_draft

SAMPLE = {
    "scope": {"category": "Electronics", "store": "CDMX-01"},
    "model_family": "seasonal_naive",
    "horizon_weeks": 8,
    "p10": 1530,
    "p50": 1840,
    "p90": 2130,
    "previous_period_units": 1590,
    "year_over_year_pct": 11.4,
    "wape": 0.132,
    "bias": 0.004,
}


def test_numbers_survive_with_the_exact_p50() -> None:
    package = _package()
    draft = MockExplanationClient().explain(package)
    validate_explanation(package, draft)
    assert "1840" in draft.summary
    assert "1900" not in draft.summary
    assert "2130" in draft.summary
    assert "1530" in draft.summary


def test_prompt_forbids_changing_numbers_promotions_causes_and_certainty() -> None:
    prompt = render_prompt(_package())
    assert PROMPT_VERSION in prompt
    assert "Do not change numerical predictions." in prompt
    assert "Do not invent promotions." in prompt
    assert "Do not invent causal claims." in prompt
    assert "Do not hide uncertainty." in prompt
    assert "Do not claim certainty." in prompt


def test_mock_uses_only_package_fields_and_makes_no_remote_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[object] = []

    class Spy:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            constructed.append(True)

    monkeypatch.setattr(
        "forecastops_ml.explanations.clients.BedrockExplanationClient",
        Spy,
    )
    client = select_explanation_client(
        bedrock_enabled=False,
        model_id="ignored",
        max_output_tokens=100,
        region="us-east-1",
    )
    draft = client.explain(_package())
    validate_explanation(_package(), draft)
    assert constructed == []
    assert isinstance(client, MockExplanationClient)
    assert "promotion" not in draft.summary.lower()
    assert "caused" not in draft.summary.lower()
    assert "certainty" not in draft.uncertainty_note.lower()


def test_changed_p50_fails_validation() -> None:
    before = explanation_validation_failures()
    draft = _valid_draft()
    draft = draft.model_copy(update={"summary": "The median forecast is 1900."})
    with pytest.raises(ExplanationValidationError, match="forecast_numbers") as caught:
        validate_explanation(_package(), draft)
    assert caught.value.check == "forecast_numbers"
    assert explanation_validation_failures() == before + 1


def test_an_extra_forecast_total_fails_even_when_p50_is_present() -> None:
    draft = _valid_draft().model_copy(
        update={"summary": "The median forecast is 1840 and another total is 2500."}
    )
    with pytest.raises(ExplanationValidationError, match="forecast_numbers"):
        validate_explanation(_package(), draft)


def test_unknown_signal_fails_validation() -> None:
    draft = _valid_draft().model_copy(update={"drivers": ["invented_promotion"]})
    with pytest.raises(ExplanationValidationError, match="known_signals") as caught:
        validate_explanation(_package(), draft)
    assert caught.value.check == "known_signals"


def test_empty_uncertainty_note_fails_validation() -> None:
    draft = _valid_draft().model_copy(update={"uncertainty_note": ""})
    with pytest.raises(ExplanationValidationError, match="uncertainty_note"):
        validate_explanation(_package(), draft)


def test_blank_uncertainty_note_fails_validation() -> None:
    draft = _valid_draft().model_copy(update={"uncertainty_note": "   "})
    with pytest.raises(ExplanationValidationError, match="uncertainty_note"):
        validate_explanation(_package(), draft)


def test_tree_signals_keep_positive_and_negative_drivers() -> None:
    package = build_explanation_package(
        scope={"series": "store-1|sku-1"},
        model_family="gradient_boosting",
        horizon_weeks=4,
        p50=120,
        positive_drivers=("holiday", "promotion"),
        negative_drivers=("price",),
        driver_values={"holiday": 3.0, "promotion": 1.0, "price": -2.0},
    )
    assert package.positive_drivers == ["holiday", "promotion"]
    assert package.negative_drivers == ["price"]
    assert [signal.name for signal in package.signals] == ["holiday", "promotion", "price"]
    assert [signal.direction for signal in package.signals] == ["positive", "positive", "negative"]
    assert {signal.kind for signal in package.signals} == {"driver"}
    draft = MockExplanationClient().explain(package)
    validate_explanation(package, draft)


def test_deepar_signals_are_context_not_causes() -> None:
    package = build_explanation_package(
        scope={"category": "Electronics"},
        model_family="deepar",
        horizon_weeks=8,
        p10=10,
        p50=12,
        p90=15,
    )
    names = [signal.name for signal in package.signals]
    assert names == [
        "historical_context",
        "recent_trend",
        "known_future_covariates",
        "seasonal_pattern",
        "uncertainty",
        "baseline_comparison",
    ]
    assert {signal.kind for signal in package.signals} == {"context"}
    assert all("cause" not in name for name in names)
    assert package.positive_drivers == []
    draft = MockExplanationClient().explain(package)
    validate_explanation(package, draft)
    assert "caused" not in draft.summary.lower()


def test_schema_rejects_a_draft_that_is_not_an_object() -> None:
    before = explanation_validation_failures()
    with pytest.raises(ExplanationValidationError, match="schema"):
        parse_draft(["not", "an", "object"])
    assert explanation_validation_failures() == before + 1


def test_package_totals_that_match_are_allowed() -> None:
    draft = ExplanationDraft(
        summary=(
            "The median forecast is 1840. "
            "The lower estimate is 1530 and the upper estimate is 2130."
        ),
        drivers=["historical_context", "recent_trend"],
        risks=["The previous period total is 1590."],
        uncertainty_note="The forecast remains uncertain.",
        recommended_checks=["The year over year change is 11.4 percent."],
    )
    validate_explanation(_package(), draft)


def test_bedrock_client_parses_a_draft_without_opening_a_session() -> None:
    package = _package()
    draft = {
        "summary": "The median forecast is 1840.",
        "drivers": ["historical_context"],
        "risks": ["The lower estimate is 1530."],
        "uncertainty_note": "The forecast remains uncertain.",
        "recommended_checks": ["Compare the median forecast with history."],
    }
    runtime = _Runtime(json.dumps(draft), input_tokens=12, output_tokens=8)
    client = BedrockExplanationClient(
        "anthropic.example",
        max_output_tokens=700,
        region="us-east-1",
        runtime=runtime,
    )
    parsed = client.explain(package)
    validate_explanation(package, parsed)
    assert runtime.calls == 1
    assert client.usage.input_tokens == 12
    assert client.usage.output_tokens == 8
    assert client.usage.latency_ms >= 0
    body = json.loads(runtime.payloads[0]["body"].decode())
    assert body["max_tokens"] == 700
    assert "Do not invent promotions." in body["messages"][0]["content"]


def _package():
    return build_explanation_package(**SAMPLE)


def _valid_draft() -> ExplanationDraft:
    return MockExplanationClient().explain(_package())


class _Runtime:
    def __init__(self, text: str, *, input_tokens: int, output_tokens: int) -> None:
        self.calls = 0
        self.payloads: list[dict[str, object]] = []
        self._text = text
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def invoke_model(self, **kwargs: object) -> Mapping[str, object]:
        self.calls += 1
        body = kwargs["body"]
        assert isinstance(body, bytes)
        self.payloads.append({"body": body, "modelId": kwargs["modelId"]})
        payload = {
            "content": [{"text": self._text}],
            "usage": {
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
            },
        }
        return {"body": io.BytesIO(json.dumps(payload).encode())}
