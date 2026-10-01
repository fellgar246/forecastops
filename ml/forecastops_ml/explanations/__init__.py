"""Grounded business explanations of a forecast package.

A language model may describe the package. It does not change the forecast.
"""

from forecastops_ml.explanations.clients import (
    BedrockExplanationClient,
    ExplanationClient,
    MockExplanationClient,
    select_explanation_client,
)
from forecastops_ml.explanations.models import (
    ExplanationDraft,
    ExplanationPackage,
    ExplanationUsage,
    Signal,
)
from forecastops_ml.explanations.package import build_explanation_package
from forecastops_ml.explanations.prompt import PROMPT_VERSION, render_prompt
from forecastops_ml.explanations.validation import (
    ExplanationChecks,
    ExplanationValidationError,
    explanation_validation_failures,
    grade_explanation,
    validate_explanation,
)

__all__ = [
    "PROMPT_VERSION",
    "BedrockExplanationClient",
    "ExplanationChecks",
    "ExplanationClient",
    "ExplanationDraft",
    "ExplanationPackage",
    "ExplanationUsage",
    "ExplanationValidationError",
    "MockExplanationClient",
    "Signal",
    "build_explanation_package",
    "explanation_validation_failures",
    "grade_explanation",
    "render_prompt",
    "select_explanation_client",
    "validate_explanation",
]
