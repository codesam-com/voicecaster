# src/voicecaster/intake/handle_result.py
from __future__ import annotations

from typing import Any


MAX_RETRIES = 10


def apply_success_result(episode: dict[str, Any]) -> dict[str, Any]:
    before = int(episode.get("retries", 0) or 0)
    episode["retries"] = 0
    episode["status"] = "transcript"
    return {
        "result": "success",
        "status_before": "intake",
        "status_after": "transcript",
        "retries_before": before,
        "retries_after": 0,
    }


def apply_failure_result(episode: dict[str, Any], error_type: str) -> dict[str, Any]:
    before = int(episode.get("retries", 0) or 0)

    if error_type == "network":
        return {
            "result": "failure",
            "error_type": "network",
            "status_before": episode.get("status"),
            "status_after": episode.get("status"),
            "retries_before": before,
            "retries_after": before,
        }

    after = before + 1
    episode["retries"] = after

    if after > MAX_RETRIES:
        episode["status"] = "ruined"
    else:
        episode["status"] = "intake"

    return {
        "result": "failure",
        "error_type": error_type,
        "status_before": "intake",
        "status_after": episode.get("status"),
        "retries_before": before,
        "retries_after": after,
    }
