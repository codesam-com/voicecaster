from __future__ import annotations

from typing import List

from .config import INPUT_EPISODES_PATH, PROCESSED_EPISODES_PATH
from .models import EpisodeEntry
from .yaml_io import read_yaml, write_yaml


ACTIVE_STATUSES = {"pending", "processing", "pending_review", "failed", "incompatible"}
PROCESSED_STATUSES = {"done"}


def load_pending_queue() -> List[EpisodeEntry]:
    payload = read_yaml(INPUT_EPISODES_PATH) or []
    items = [EpisodeEntry.model_validate(item) for item in payload]
    return [item for item in items if item.status in ACTIVE_STATUSES]


def save_queue(items: List[EpisodeEntry]) -> None:
    filtered = [item for item in items if item.status in ACTIVE_STATUSES]
    write_yaml(
        INPUT_EPISODES_PATH,
        [item.model_dump(mode="json") for item in filtered],
    )


def load_processed_queue() -> List[EpisodeEntry]:
    payload = read_yaml(PROCESSED_EPISODES_PATH) or []
    items = [EpisodeEntry.model_validate(item) for item in payload]
    return [item for item in items if item.status in PROCESSED_STATUSES]


def save_processed_queue(items: List[EpisodeEntry]) -> None:
    filtered = [item for item in items if item.status in PROCESSED_STATUSES]
    write_yaml(
        PROCESSED_EPISODES_PATH,
        [item.model_dump(mode="json") for item in filtered],
    )


def reserve_next_pending_episode() -> EpisodeEntry | None:
    items = load_pending_queue()
    for idx, item in enumerate(items):
        if item.status == "pending":
            item.status = "processing"
            items[idx] = item
            save_queue(items)
            return item
    return None


def update_episode_status(
    episode_id: str,
    new_status: str,
    increment_retries: bool = False,
) -> None:
    items = load_pending_queue()
    for idx, item in enumerate(items):
        if item.id == episode_id:
            item.status = new_status
            if increment_retries:
                item.retries += 1
            items[idx] = item
            break
    save_queue(items)


def move_episode_to_processed(episode_id: str) -> None:
    items = load_pending_queue()
    processed = load_processed_queue()

    remaining = []
    moved = None

    for item in items:
        if item.id == episode_id:
            moved = item
        else:
            remaining.append(item)

    if moved is None:
        return

    moved.status = "done"

    processed = [p for p in processed if p.id != episode_id]
    processed.append(moved)

    save_queue(remaining)
    save_processed_queue(processed)


def has_processing_episode() -> bool:
    return any(item.status == "processing" for item in load_pending_queue())
