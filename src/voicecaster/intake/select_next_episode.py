# src/voicecaster/intake/select_next_episode.py
from __future__ import annotations

from typing import Any


def select_next_intake_episode(episodes: list[dict[str, Any]]) -> tuple[int | None, dict[str, Any] | None]:
    for index, episode in enumerate(episodes):
        if episode.get("status") == "intake":
            return index, episode
    return None, None
