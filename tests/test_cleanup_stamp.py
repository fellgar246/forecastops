"""The cleanup command records the time the cost page can show."""

import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cleanup_stamp_is_a_utc_timestamp(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "aws" / "common.sh"
    completed = subprocess.run(
        ["bash", "-c", f'source "{script}" && record_cleanup_stamp "{tmp_path}"'],
        check=True,
        capture_output=True,
        text=True,
    )
    stamp = (tmp_path / "var" / "cost" / "last_cleanup_at").read_text(encoding="utf-8").strip()
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert "Recorded last cleanup at" in completed.stdout
    assert stamp in completed.stdout
