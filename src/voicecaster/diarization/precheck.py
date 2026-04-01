# src/voicecaster/diarization/precheck.py

from __future__ import annotations

import json
import sys
from pathlib import Path


INPUTS_PATH = Path("inputs/inputs.json")


def find_next_diarization_episode() -> dict | None:
    if not INPUTS_PATH.exists():
        return None

    data = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))

    for episode in data:
        if episode.get("status") == "diarization":
            return episode

    return None


def main() -> None:
    episode = find_next_diarization_episode()

    should_run = "true" if episode else "false"

    # GitHub Actions output
    print(f"should_run={should_run}")

    # También útil para logs
    if episode:
        print(f"[precheck] Found episode: {episode.get('id')}")
    else:
        print("[precheck] No episode in status=diarization")


if __name__ == "__main__":
    main()
