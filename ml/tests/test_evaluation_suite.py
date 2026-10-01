"""Frozen benchmark bands and deterministic explanation checks."""

import io
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import boto3
import pytest

from forecastops_ml.evaluation.benchmark import (
    MeasuredModel,
    ModelScore,
    compare_measurements,
    load_benchmark_pin,
)
from forecastops_ml.evaluation.explanation_suite import (
    ExplanationSummary,
    explanation_scenarios,
)
from forecastops_ml.evaluation.judge import BedrockExplanationJudge
from forecastops_ml.evaluation.suite import (
    EvaluationSuiteReport,
    JudgeSummary,
    main,
    run_evaluation_suite,
    write_report,
)
from forecastops_ml.explanations import (
    ExplanationDraft,
    MockExplanationClient,
    explanation_validation_failures,
    grade_explanation,
)

PIN = (
    Path(__file__).resolve().parents[1]
    / "forecastops_ml"
    / "evaluation"
    / "fixtures"
    / "benchmark-v1.json"
)


@pytest.fixture(scope="module")
def suite_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[EvaluationSuiteReport, Path]:
    original = boto3.client

    def forbid(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("boto3.client was called")

    boto3.client = forbid
    try:
        destination = tmp_path_factory.mktemp("evaluation") / "evaluation-report.json"
        report = run_evaluation_suite(destination)
    finally:
        boto3.client = original
    return report, destination


def test_pinned_identity_matches_the_checked_in_benchmark() -> None:
    pin = load_benchmark_pin()

    assert pin.benchmark_version == "benchmark-v1"
    assert pin.seed == 20260921
    assert pin.profile == "test"
    assert pin.daily_row_count == 436
    assert pin.weekly_row_count == 70
    assert pin.folds == 5
    assert pin.horizon == 1
    assert pin.tolerance == 1e-6
    assert [item.family for item in pin.models] == [
        "naive",
        "seasonal_naive",
        "gradient_boosting",
    ]
    assert [item.grain for item in pin.models] == ["day", "day", "week"]


def test_benchmark_metrics_stay_inside_the_pinned_bands(
    suite_run: tuple[EvaluationSuiteReport, Path],
) -> None:
    report, destination = suite_run
    pin = load_benchmark_pin()
    document = json.loads(destination.read_text(encoding="utf-8"))

    assert report.gate_passed()
    assert document["benchmark_version"] == "benchmark-v1"
    assert "judge" not in document
    assert document["explanations"] == {
        "number_failures": 0,
        "scenarios": 50,
        "signal_failures": 0,
        "uncertainty_failures": 0,
    }
    assert [row["family"] for row in document["models"]] == [
        "naive",
        "seasonal_naive",
        "gradient_boosting",
    ]
    for row, pinned in zip(document["models"], pin.models, strict=True):
        assert row["passed"] is True
        assert row["wape"] != 0
        assert row["wape"] == pytest.approx(pinned.wape, abs=pin.tolerance)
        assert row["wape"] == pytest.approx(
            next(item.wape for item in report.models if item.family == row["family"])
        )


def test_tightened_wape_fails_the_regression(
    suite_run: tuple[EvaluationSuiteReport, Path],
) -> None:
    report, _destination = suite_run
    pin = load_benchmark_pin()
    measured = tuple(MeasuredModel(item.family, item.wape) for item in report.models)
    tightened = replace(
        pin,
        models=tuple(
            replace(item, wape=item.wape - 0.05) if item.family == "naive" else item
            for item in pin.models
        ),
    )

    scores = compare_measurements(measured, tightened)
    naive = next(item for item in scores if item.family == "naive")

    assert naive.passed is False
    assert naive.wape == pytest.approx(measured[0].wape)
    failed = replace(report, models=scores)
    assert failed.gate_passed() is False


def test_pin_rejects_a_model_list_that_drops_a_family(tmp_path: Path) -> None:
    raw = json.loads(PIN.read_text(encoding="utf-8"))
    raw["models"] = list(reversed(raw["models"]))
    path = tmp_path / "benchmark-v1.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="naive, seasonal_naive, gradient_boosting"):
        load_benchmark_pin(path)


def test_explanation_scenarios_cover_the_required_situations() -> None:
    scenarios = explanation_scenarios()
    counts = Counter(item.kind for item in scenarios)

    assert len(scenarios) >= 50
    assert counts["normal"] >= 1
    assert counts["promotion"] >= 1
    assert counts["stockout"] >= 1
    assert counts["wide_interval"] >= 1
    assert counts["flat"] >= 1
    for scenario in scenarios:
        names = {signal.name for signal in scenario.package.signals}
        forecast = scenario.package.forecast
        wide = (
            forecast.p10 is not None
            and forecast.p90 is not None
            and (forecast.p90 - forecast.p10) >= 2 * abs(forecast.p50)
        )
        flat = forecast.p10 == forecast.p50 == forecast.p90
        if scenario.kind == "normal":
            assert not wide
            assert not flat
            assert "promotion" not in names
            assert "stockout" not in names
        elif scenario.kind == "promotion":
            assert "promotion" in names
        elif scenario.kind == "stockout":
            stockouts = [signal for signal in scenario.package.signals if signal.name == "stockout"]
            assert len(stockouts) == 1
            assert stockouts[0].kind == "context"
        elif scenario.kind == "wide_interval":
            assert wide
        else:
            assert scenario.kind == "flat"
            assert flat


def test_mock_drafts_pass_the_deterministic_checks_without_a_remote_call() -> None:
    before = explanation_validation_failures()
    scenarios = explanation_scenarios()
    client = MockExplanationClient()
    for scenario in scenarios:
        checks = grade_explanation(scenario.package, client.explain(scenario.package))
        assert checks.number_preserved
        assert checks.signals_known
        assert checks.uncertainty_acknowledged
    assert explanation_validation_failures() == before


def test_each_deterministic_check_is_recorded_on_its_own() -> None:
    before = explanation_validation_failures()
    scenario = explanation_scenarios()[0]
    draft = ExplanationDraft(
        summary="The median forecast is 999999.",
        drivers=["invented_promotion"],
        risks=[],
        uncertainty_note="   ",
        recommended_checks=[],
    )

    checks = grade_explanation(scenario.package, draft)

    assert checks.number_preserved is False
    assert checks.signals_known is False
    assert checks.uncertainty_acknowledged is False
    assert explanation_validation_failures() == before


def test_explanation_failures_fail_the_gate_when_the_judge_score_is_high() -> None:
    report = EvaluationSuiteReport(
        benchmark_version="benchmark-v1",
        models=(
            ModelScore("naive", 0.3, True),
            ModelScore("seasonal_naive", 0.2, True),
            ModelScore("gradient_boosting", 0.2, True),
        ),
        explanations=ExplanationSummary(
            scenarios=50,
            number_failures=1,
            signal_failures=0,
            uncertainty_failures=0,
        ),
        judge=JudgeSummary(
            status="scored",
            readability=5,
            groundedness=5,
            actionability=5,
        ),
    )

    assert report.gate_passed() is False


def test_judge_is_skipped_when_bedrock_is_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_measurements(monkeypatch)

    def forbid(self: object, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("Bedrock judge was constructed")

    monkeypatch.setattr(BedrockExplanationJudge, "__init__", forbid)
    report = run_evaluation_suite(
        tmp_path / "evaluation-report.json",
        judge_enabled=True,
        bedrock_enabled=False,
    )

    assert report.gate_passed()
    assert report.judge is not None
    assert report.judge.status == "skipped"
    assert report.judge.reason == "BEDROCK_ENABLED is false."


def test_low_judge_scores_do_not_fail_the_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_measurements(monkeypatch)
    report = run_evaluation_suite(
        tmp_path / "evaluation-report.json",
        judge_enabled=True,
        bedrock_enabled=True,
        judge_runtime=_Scores(1, 1, 1),
    )

    assert report.gate_passed()
    assert report.judge is not None
    assert report.judge.status == "scored"
    assert report.judge.readability == 1
    assert report.judge.groundedness == 1
    assert report.judge.actionability == 1


def test_a_judge_error_stays_out_of_the_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_measurements(monkeypatch)
    report = run_evaluation_suite(
        tmp_path / "evaluation-report.json",
        judge_enabled=True,
        bedrock_enabled=True,
        judge_runtime=_Scores(9, 1, 1),
    )

    assert report.gate_passed()
    assert report.judge is not None
    assert report.judge.status == "error"


def test_cli_defaults_the_judge_and_bedrock_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXPLANATION_JUDGE_ENABLED", raising=False)
    monkeypatch.delenv("BEDROCK_ENABLED", raising=False)
    captured: dict[str, object] = {}

    def fake(output: Path, **kwargs: object) -> EvaluationSuiteReport:
        captured.update(kwargs)
        report = _passing_report()
        write_report(report, output)
        return report

    monkeypatch.setattr("forecastops_ml.evaluation.suite.run_evaluation_suite", fake)
    destination = tmp_path / "evaluation-report.json"

    assert main(["--output", str(destination)]) == 0
    assert captured["judge_enabled"] is False
    assert captured["bedrock_enabled"] is False
    assert destination.is_file()


def test_cli_returns_failure_when_the_gate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(output: Path, **_kwargs: object) -> EvaluationSuiteReport:
        report = replace(
            _passing_report(),
            explanations=ExplanationSummary(50, 0, 2, 0),
        )
        write_report(report, output)
        return report

    monkeypatch.setattr("forecastops_ml.evaluation.suite.run_evaluation_suite", fake)

    assert main(["--output", str(tmp_path / "evaluation-report.json")]) == 1


def _stub_measurements(monkeypatch: pytest.MonkeyPatch) -> None:
    def measure(pin: object = None) -> tuple[MeasuredModel, ...]:
        selected = load_benchmark_pin() if pin is None else pin
        return tuple(MeasuredModel(item.family, item.wape) for item in selected.models)

    monkeypatch.setattr("forecastops_ml.evaluation.suite.measure_benchmark", measure)


def _passing_report() -> EvaluationSuiteReport:
    return EvaluationSuiteReport(
        benchmark_version="benchmark-v1",
        models=(
            ModelScore("naive", 0.3, True),
            ModelScore("seasonal_naive", 0.28, True),
            ModelScore("gradient_boosting", 0.27, True),
        ),
        explanations=ExplanationSummary(50, 0, 0, 0),
    )


class _Scores:
    def __init__(self, readability: int, groundedness: int, actionability: int) -> None:
        self._body = json.dumps(
            {
                "content": [
                    {
                        "text": json.dumps(
                            {
                                "actionability": actionability,
                                "groundedness": groundedness,
                                "readability": readability,
                            }
                        )
                    }
                ]
            }
        ).encode()

    def invoke_model(self, **_kwargs: object) -> dict[str, object]:
        return {"body": io.BytesIO(self._body)}
