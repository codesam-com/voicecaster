from __future__ import annotations

import json
import sys
from pathlib import Path

INPUTS_PATH = Path("inputs/inputs.json")


def load_inputs() -> list[dict]:
    if not INPUTS_PATH.exists():
        return []

    try:
        data = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[precheck] Failed to parse inputs file: {exc}", file=sys.stderr)
        return []

    if not isinstance(data, list):
        print("[precheck] inputs.json is not a list.", file=sys.stderr)
        return []

    return data


def find_next_alignment_episode(inputs_data: list[dict]) -> dict | None:
    for episode in inputs_data:
        if not isinstance(episode, dict):
            continue
        if episode.get("status") == "alignment":
            return episode
    return None


def main() -> int:
    inputs_data = load_inputs()
    episode = find_next_alignment_episode(inputs_data)

    should_run = "true" if episode else "false"
    episode_id = str(episode.get("id", "")) if episode else ""

    print(f"should_run={should_run}")
    print(f"episode_id={episode_id}")

    if episode:
        print(f"[precheck] Found episode with status=alignment: {episode_id}")
    else:
        print("[precheck] No episode with status=alignment found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
