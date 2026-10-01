"""Run benchmark-v1 and the explanation checks, then write the JSON report.

The optional judge is off unless ``EXPLANATION_JUDGE_ENABLED`` is set. It
calls Bedrock only when ``BEDROCK_ENABLED`` is also set. Judge scores are
reported beside the gate and cannot fail the run by themselves.
"""

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from forecastops_ml.evaluation.benchmark import (
    BENCHMARK_VERSION,
    ModelScore,
    compare_measurements,
    load_benchmark_pin,
    measure_benchmark,
)
from forecastops_ml.evaluation.explanation_suite import (
    ExplanationScenario,
    ExplanationSummary,
    explanation_scenarios,
    score_explanations,
)
from forecastops_ml.evaluation.judge import BedrockExplanationJudge, InvokeModel, JudgeScores
from forecastops_ml.explanations import MockExplanationClient

DEFAULT_JUDGE_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
DEFAULT_REGION = "us-east-1"
_TRUE = {"1", "true", "yes"}


@dataclass(frozen=True)
class JudgeSummary:
    """Optional judge outcome. ``skipped`` and ``error`` are not gate failures."""

    status: str
    readability: float | None = None
    groundedness: float | None = None
    actionability: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this judge outcome."""

        payload: dict[str, object] = {"status": self.status}
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.readability is not None:
            payload["actionability"] = self.actionability
            payload["groundedness"] = self.groundedness
            payload["readability"] = self.readability
        return payload


@dataclass(frozen=True)
class EvaluationSuiteReport:
    """Machine-readable outcome of one evaluation run."""

    benchmark_version: str
    models: tuple[ModelScore, ...]
    explanations: ExplanationSummary
    judge: JudgeSummary | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON document for this report."""

        document: dict[str, object] = {
            "benchmark_version": self.benchmark_version,
            "explanations": self.explanations.to_dict(),
            "models": [_model_dict(item) for item in self.models],
        }
        if self.judge is not None:
            document["judge"] = self.judge.to_dict()
        return document

    def gate_passed(self) -> bool:
        """Return whether the benchmark bands and deterministic checks passed.

        Judge scores are ignored here.
        """

        if self.explanations.scenarios < 50:
            return False
        if self.explanations.number_failures != 0:
            return False
        if self.explanations.signal_failures != 0:
            return False
        if self.explanations.uncertainty_failures != 0:
            return False
        return all(item.passed for item in self.models)


def run_evaluation_suite(
    output: Path,
    *,
    judge_enabled: bool = False,
    bedrock_enabled: bool = False,
    judge_runtime: InvokeModel | None = None,
    model_id: str = DEFAULT_JUDGE_MODEL_ID,
    region: str = DEFAULT_REGION,
) -> EvaluationSuiteReport:
    """Score the pinned benchmark and explanation scenarios, then write ``output``.

    Explanation text comes from the local mock. Bedrock is used only for the
    optional judge, and only when both flags are true.
    """

    pin = load_benchmark_pin()
    scores = compare_measurements(measure_benchmark(pin), pin)
    scenarios = explanation_scenarios()
    client = MockExplanationClient()
    explanations = score_explanations(scenarios, client)
    judge = _judge_summary(
        scenarios,
        client,
        judge_enabled=judge_enabled,
        bedrock_enabled=bedrock_enabled,
        judge_runtime=judge_runtime,
        model_id=model_id,
        region=region,
    )
    report = EvaluationSuiteReport(
        benchmark_version=BENCHMARK_VERSION,
        models=scores,
        explanations=explanations,
        judge=judge,
    )
    write_report(report, output)
    return report


def write_report(report: EvaluationSuiteReport, output: Path) -> Path:
    """Write the suite report as JSON and return that path."""

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    output.write_text(payload, encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """Write the report and return 0 when the gate passes."""

    parser = argparse.ArgumentParser(
        description="Score the frozen benchmark and explanation checks."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_evaluation_suite(
        args.output,
        judge_enabled=_env_flag("EXPLANATION_JUDGE_ENABLED"),
        bedrock_enabled=_env_flag("BEDROCK_ENABLED"),
        model_id=os.environ.get("BEDROCK_MODEL_ID", DEFAULT_JUDGE_MODEL_ID),
        region=os.environ.get("AWS_REGION", DEFAULT_REGION),
    )
    status = "passed" if report.gate_passed() else "failed"
    print(f"Evaluation suite {status}. Report: {args.output}")
    return 0 if report.gate_passed() else 1


def _judge_summary(
    scenarios: tuple[ExplanationScenario, ...],
    client: MockExplanationClient,
    *,
    judge_enabled: bool,
    bedrock_enabled: bool,
    judge_runtime: InvokeModel | None,
    model_id: str,
    region: str,
) -> JudgeSummary | None:
    if not judge_enabled:
        return None
    if not bedrock_enabled:
        return JudgeSummary(status="skipped", reason="BEDROCK_ENABLED is false.")
    judge = BedrockExplanationJudge(
        model_id,
        max_output_tokens=200,
        region=region,
        runtime=judge_runtime,
    )
    readability: list[int] = []
    groundedness: list[int] = []
    actionability: list[int] = []
    try:
        for scenario in scenarios:
            scores: JudgeScores = judge.score(scenario.package, client.explain(scenario.package))
            readability.append(scores.readability)
            groundedness.append(scores.groundedness)
            actionability.append(scores.actionability)
    except ValueError as exc:
        return JudgeSummary(status="error", reason=str(exc))
    count = len(readability)
    return JudgeSummary(
        status="scored",
        readability=sum(readability) / count,
        groundedness=sum(groundedness) / count,
        actionability=sum(actionability) / count,
    )


def _model_dict(score: ModelScore) -> dict[str, object]:
    return {"family": score.family, "passed": score.passed, "wape": score.wape}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


if __name__ == "__main__":
    raise SystemExit(main())
