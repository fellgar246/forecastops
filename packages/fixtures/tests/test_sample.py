"""Checked-in dataset fixture tests."""

import csv
from pathlib import Path

from forecastops_contracts import REQUIRED_COLUMNS, SCHEMA_VERSION

FIXTURE = Path(__file__).resolve().parents[1] / SCHEMA_VERSION / "sample.csv"


def test_sample_fixture_matches_the_schema() -> None:
    with FIXTURE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert list(rows[0]) == list(REQUIRED_COLUMNS)
    assert rows[0]["store_id"] == "store-1"
    assert int(rows[0]["units_sold"]) == 10
