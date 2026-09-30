"""Versioned instructions for an explanation adapter."""

import json

from forecastops_ml.explanations.models import ExplanationPackage

PROMPT_VERSION = "v1"

_RULES = (
    "Do not change numerical predictions.",
    "Do not invent promotions.",
    "Do not invent causal claims.",
    "Do not hide uncertainty.",
    "Do not claim certainty.",
)


def render_prompt(package: ExplanationPackage) -> str:
    """Render the versioned prompt for ``package``.

    The instruction forbids changing numerical predictions, inventing
    promotions, inventing causal claims, hiding uncertainty, and claiming
    certainty.
    """

    payload = json.dumps(package.prompt_payload(), sort_keys=True, separators=(",", ":"))
    rules = "\n".join(_RULES)
    return (
        "You write a business explanation of one retail demand forecast.\n"
        "Use only the JSON package below. Describe signals and context, not causes.\n"
        f"{rules}\n"
        "Return JSON with keys summary, drivers, risks, uncertainty_note, "
        "and recommended_checks.\n"
        "Each drivers entry must be a signal name copied from the package.\n"
        "uncertainty_note must be non-empty.\n"
        f"prompt_version: {PROMPT_VERSION}\n"
        f"package:\n{payload}\n"
    )
