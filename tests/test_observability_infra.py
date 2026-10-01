"""Operational alarms stay defined and unsubscribed until demo asks for email."""

import re
from pathlib import Path

from forecastops_api.observability import ALARMS, METRIC_NAMES

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "infra" / "modules" / "observability" / "main.tf"
GROUPS = ("API", "Training", "Forecast", "Explanation")


def test_dashboard_groups_and_alarm_list_match_the_catalog() -> None:
    text = MODULE.read_text(encoding="utf-8")
    for alarm in ALARMS:
        assert alarm.name in text
        for name in alarm.metric_names:
            assert name in METRIC_NAMES
            assert name in text
    for title in GROUPS:
        assert title in text
    assert 'namespace = "ForecastOps"' in text
    assert "actions_enabled     = local.notifications_enabled" in text
    assert 'var.environment == "demo" && var.enable_notifications' in text


def test_alarm_notifications_default_off_and_log_retention_is_unchanged() -> None:
    for name in ("dev", "demo"):
        variables = (ROOT / "infra" / "environments" / name / "variables.tf").read_text(
            encoding="utf-8"
        )
        main = (ROOT / "infra" / "environments" / name / "main.tf").read_text(encoding="utf-8")
        assert re.search(
            r'variable "enable_alarm_notifications"[\s\S]*?default\s*=\s*false',
            variables,
        )
        assert 'var.environment == "demo" ? 14 : 7' in main
        assert "module.observability" in main or 'source = "../../modules/observability"' in main
