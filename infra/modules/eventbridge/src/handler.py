"""Receive one scheduled action.

This function does not start training. The monthly action stays pending until
a person confirms the retrain request.
"""

from typing import Any

DAILY_FORECAST = "daily_forecast"
WEEKLY_EVALUATION = "weekly_evaluation"
MONTHLY_RETRAIN = "monthly_retrain"
ACTIONS = {DAILY_FORECAST, WEEKLY_EVALUATION, MONTHLY_RETRAIN}


def handler(event: dict[str, Any], _context: object) -> dict[str, Any]:
    """Name the action a clock requested. Monthly retraining is not started."""

    action = event.get("action")
    if action not in ACTIONS:
        return {"action": action, "status": "REJECTED", "training_started": False}
    if action == MONTHLY_RETRAIN:
        return {"action": action, "status": "PENDING", "training_started": False}
    return {"action": action, "status": "ACCEPTED", "training_started": False}
