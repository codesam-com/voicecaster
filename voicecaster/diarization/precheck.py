from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import INPUTS_JSON_PATH


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _has_pending_diarization_episode(inputs_payload: list[dict[str, Any]]) -> bool:
    for item in inputs_payload:
        if not isinstance(item, dict):
            continue
        if item.get("status") == "diarization":
            return True
    return False


def _emit_github_output(name: str, value: str) -> None:
    github_output = Path.cwd() / "github_output_fallback.txt"

    import os
    output_path = os.getenv("GITHUB_OUTPUT")
    if output_path:
        path = Path(output_path)
    else:
        path = github_output

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{name}={value}\n")


def main() -> int:
    try:
        payload = _load_json(INPUTS_JSON_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"[precheck] failed to load inputs: {exc}")
        _emit_github_output("should_run", "false")
        return 0

    if not isinstance(payload, list):
        print("[precheck] inputs/inputs.json must contain a JSON list")
        _emit_github_output("should_run", "false")
        return 0

    should_run = _has_pending_diarization_episode(payload)

    print(f"[precheck] should_run={str(should_run).lower()}")
    _emit_github_output("should_run", "true" if should_run else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
