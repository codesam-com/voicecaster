# src/voicecaster/intake/logging_report.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from .json_utils import write_json_atomic
from .time_utils import utc_now_iso


def build_initial_report(episode_id: str, episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "workflow": "intake",
        "started_at": utc_now_iso(),
        "finished_at": None,
        "result": "running",
        "notes": [],
        "source": {
            "url_original": episode.get("url"),
            "url_normalized": None,
            "source_type": None,
        },
    }


def finalize_report(
    report: dict[str, Any],
    *,
    result: str,
    note: str | None = None,
) -> dict[str, Any]:
    report["finished_at"] = utc_now_iso()
    report["result"] = result
    if note:
        report.setdefault("notes", []).append(note)
    return report


def save_report(report_path: Path, report: dict[str, Any]) -> None:
    write_json_atomic(report_path, report, indent=2)
