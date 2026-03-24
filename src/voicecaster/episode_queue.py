from __future__ import annotations

from typing import List

from .config import INPUT_EPISODES_PATH, PROCESSED_EPISODES_PATH
from .models import EpisodeEntry
from .yaml_io import read_yaml, write_yaml


def load_pending_queue() -> List[EpisodeEntry]:
    payload = read_yaml(INPUT_EPISODES_PATH) or []
    return [EpisodeEntry.model_validate(item) for item in payload]


def save_queue(items: List[EpisodeEntry]) -> None:
    write_yaml(
        INPUT_EPISODES_PATH,
        [item.model_dump(mode="json", exclude_none=False) for item in items],
    )


def load_processed_queue() -> List[EpisodeEntry]:
    payload = read_yaml(PROCESSED_EPISODES_PATH) or []
    return [EpisodeEntry.model_validate(item) for item in payload]


def save_processed_queue(items: List[EpisodeEntry]) -> None:
    write_yaml(
        PROCESSED_EPISODES_PATH,
        [item.model_dump(mode="json", exclude_none=False) for item in items],
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


def update_episode_status(episode_id: str, new_status: str, increment_retries: bool = False) -> None:
    items = load_pending_queue()
    for idx, item in enumerate(items):
        if item.id == episode_id:
            item.status = new_status
            if increment_retries:
                item.retries += 1
            items[idx] = item
            break
    save_queue(items)


def update_episode_operational_audio_url(
    episode_id: str,
    new_url: str,
) -> None:
    items = load_pending_queue()
    for idx, item in enumerate(items):
        if item.id != episode_id:
            continue

        if item.source_url_original is None:
            item.source_url_original = item.url

        item.url = new_url
        item.source_drive_url = new_url
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

    processed.append(moved)
    save_queue(remaining)
    save_processed_queue(processed)
