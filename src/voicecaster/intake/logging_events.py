# src/voicecaster/intake/logging_events.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .time_utils import utc_now_iso


def append_event(event_log_path: Path, event_type: str, **fields: Any) -> None:
    event_log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": utc_now_iso(),
        "event": event_type,
        **fields,
    }
    with event_log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False))
        handle.write("\n")
