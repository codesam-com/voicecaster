from __future__ import annotations

from pathlib import Path

from .config import REVIEWS_DIR, SPEAKERS_DIR, WORK_DIR
from .episode_queue import load_pending_queue, move_episode_to_processed, update_episode_status
from .reporting import write_json, utc_now_iso
from .yaml_io import read_yaml, write_yaml


def _find_pending_review_episode():
    for item in load_pending_queue():
        if item.status == "pending_review":
            return item
    return None


def run_postaudit() -> int:
    episode = _find_pending_review_episode()
    if episode is None:
        print("No hay episodios en pending_review.")
        return 0

    audit_path = REVIEWS_DIR / episode.id / "audit.yaml"
    audit_data = read_yaml(audit_path) or {}

    if not audit_data.get("approved_as_source_of_truth", False):
        print(f"{episode.id}: sin aprobación final. POSTAUDITORÍA no ejecutada.")
        return 0

    final_mapping = audit_data.get("speaker_mapping_final", {})
    for speaker_id, real_name in final_mapping.items():
        speaker_file = SPEAKERS_DIR / f"{real_name}.yaml"
        current = read_yaml(speaker_file) or {
            "name": real_name,
            "identifiers": [],
            "embeddings": [],
            "voice_models": [],
            "speech_style": {},
            "episode_history": [],
            "versions": [],
        }
        current["episode_history"].append(
            {
                "episode_id": episode.id,
                "speaker_id": speaker_id,
                "validated_at": utc_now_iso(),
            }
        )
        write_yaml(speaker_file, current)

    write_json(
        WORK_DIR / episode.id / "report_postaudit.json",
        {
            "episode_id": episode.id,
            "phase": "postaudit",
            "finished_at": utc_now_iso(),
            "result": "done",
            "validated_mapping": final_mapping,
        },
    )

    update_episode_status(episode.id, "done")
    move_episode_to_processed(episode.id)
    print(f"POSTAUDITORÍA completada para {episode.id} -> done")
    return 0
