# src/voicecaster/intake/detect_duplicates.py
from __future__ import annotations

from typing import Any


def detect_and_mark_duplicate_intake_ids(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    changes: list[dict[str, Any]] = []

    for index, episode in enumerate(episodes):
        if episode.get("status") != "intake":
            continue

        episode_id = str(episode.get("id", "")).strip()
        if not episode_id:
            continue

        if episode_id in seen_ids:
            old_status = episode.get("status")
            episode["status"] = "duplicated"
            changes.append(
                {
                    "index": index,
                    "id": episode_id,
                    "from_status": old_status,
                    "to_status": "duplicated",
                }
            )
        else:
            seen_ids.add(episode_id)

    return changes
