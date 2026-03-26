from __future__ import annotations

import json
from datetime import datetime, timedelta, UTC
from pathlib import Path


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_runtime_control(path: Path) -> dict:
    if not path.exists():
        return {
            "last_run_finished_at": None,
            "last_run_duration_seconds": 0,
            "multiplier": 1.0,
        }

    return json.loads(path.read_text(encoding="utf-8"))


def should_run_now(path: Path, now: datetime | None = None) -> tuple[bool, dict]:
    now = now or datetime.now(UTC)
    payload = load_runtime_control(path)

    multiplier = float(payload.get("multiplier", 1.0))
    last_finished = _parse_dt(payload.get("last_run_finished_at"))
    last_duration = float(payload.get("last_run_duration_seconds", 0))

    if last_finished is None or last_duration <= 0:
        payload["decision"] = "run"
        return True, payload

    next_allowed = last_finished + timedelta(seconds=last_duration * multiplier)

    payload["next_allowed_run_at"] = next_allowed.isoformat()

    if now < next_allowed:
        payload["decision"] = "skip"
        return False, payload

    payload["decision"] = "run"
    return True, payload


def update_runtime_control(path: Path, duration_seconds: float, finished_at: datetime | None = None) -> None:
    finished_at = finished_at or datetime.now(UTC)
    current = load_runtime_control(path)

    current["last_run_finished_at"] = finished_at.isoformat()
    current["last_run_duration_seconds"] = round(float(duration_seconds), 3)
    current.setdefault("multiplier", 1.0)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(current, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
