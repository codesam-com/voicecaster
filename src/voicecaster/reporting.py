from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
