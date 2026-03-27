from __future__ import annotations

import json
from datetime import datetime, timedelta, UTC
from pathlib import Path


DEFAULT_MULTIPLIER = 1.0


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _default_workflow_payload() -> dict:
    return {
        "last_run_finished_at": None,
        "last_run_duration_seconds": 0,
        "multiplier": DEFAULT_MULTIPLIER,
    }


def load_runtime_control(path: Path) -> dict:
    if not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    return json.loads(raw)


def _ensure_workflow_entry(payload: dict, workflow_name: str) -> dict:
    if workflow_name not in payload or not isinstance(payload[workflow_name], dict):
        payload[workflow_name] = _default_workflow_payload()
    else:
        payload[workflow_name].setdefault("last_run_finished_at", None)
        payload[workflow_name].setdefault("last_run_duration_seconds", 0)
        payload[workflow_name].setdefault("multiplier", DEFAULT_MULTIPLIER)
    return payload


def should_run_now(path: Path, workflow_name: str, now: datetime | None = None) -> tuple[bool, dict]:
    now = now or datetime.now(UTC)

    payload = load_runtime_control(path)
    payload = _ensure_workflow_entry(payload, workflow_name)

    entry = payload[workflow_name]

    multiplier = float(entry.get("multiplier", DEFAULT_MULTIPLIER))
    last_finished = _parse_dt(entry.get("last_run_finished_at"))
    last_duration = float(entry.get("last_run_duration_seconds", 0))

    debug_info = {
        "workflow_name": workflow_name,
        "last_run_finished_at": entry.get("last_run_finished_at"),
        "last_run_duration_seconds": last_duration,
        "multiplier": multiplier,
        "decision": None,
        "next_allowed_run_at": None,
    }

    if last_finished is None or last_duration <= 0:
        debug_info["decision"] = "run"
        return True, debug_info

    next_allowed = last_finished + timedelta(seconds=last_duration * multiplier)
    debug_info["next_allowed_run_at"] = next_allowed.isoformat()

    if now < next_allowed:
        debug_info["decision"] = "skip"
        return False, debug_info

    debug_info["decision"] = "run"
    return True, debug_info


def update_runtime_control(
    path: Path,
    workflow_name: str,
    duration_seconds: float,
    finished_at: datetime | None = None,
) -> None:
    finished_at = finished_at or datetime.now(UTC)

    payload = load_runtime_control(path)
    payload = _ensure_workflow_entry(payload, workflow_name)

    payload[workflow_name]["last_run_finished_at"] = finished_at.isoformat()
    payload[workflow_name]["last_run_duration_seconds"] = round(float(duration_seconds), 3)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
