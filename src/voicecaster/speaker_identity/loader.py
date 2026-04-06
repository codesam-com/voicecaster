from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import INPUTS_PATH, TARGET_STATUS, WORK_DIR


def load_inputs() -> list[dict[str, Any]]:
    if not INPUTS_PATH.exists():
        raise RuntimeError(f"Inputs file not found: {INPUTS_PATH}")

    data = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("inputs/inputs.json must contain a JSON array.")
    return data


def save_inputs(data: list[dict[str, Any]]) -> None:
    INPUTS_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def select_episode() -> dict[str, Any]:
    data = load_inputs()
    for ep in data:
        if ep.get("status") == TARGET_STATUS:
            return ep
    raise RuntimeError(f"No episode with status={TARGET_STATUS} found.")


def update_episode_status(episode_id: str, new_status: str) -> None:
    data = load_inputs()
    for ep in data:
        if ep.get("id") == episode_id:
            ep["status"] = new_status
            ep["retries"] = 0
            break
    save_inputs(data)


def increment_retries(episode_id: str) -> int:
    data = load_inputs()
    current_retries = 0
    for ep in data:
        if ep.get("id") == episode_id:
            current_retries = int(ep.get("retries", 0)) + 1
            ep["retries"] = current_retries
            break
    save_inputs(data)
    return current_retries


def mark_ruined(episode_id: str) -> None:
    data = load_inputs()
    for ep in data:
        if ep.get("id") == episode_id:
            ep["status"] = "ruined"
            break
    save_inputs(data)


def _load_json(path: Path, expected_type: type) -> Any:
    if not path.exists():
        raise RuntimeError(f"Missing required file: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, expected_type):
        raise RuntimeError(f"Invalid JSON type in {path}: expected {expected_type.__name__}")
    return data


def load_episode_inputs_record(episode_id: str) -> dict[str, Any]:
    data = load_inputs()
    for ep in data:
        if ep.get("id") == episode_id:
            return ep
    raise RuntimeError(f"Episode not found in inputs.json: {episode_id}")


def ensure_required_paths(work_episode_dir: Path) -> None:
    required_paths = [
        work_episode_dir / "04_alignment" / "aligned_utterances.json",
        work_episode_dir / "04_alignment" / "aligned_turns.json",
        work_episode_dir / "04_alignment" / "aligned_words.json",
        work_episode_dir / "04_alignment" / "speakers_index.json",
        work_episode_dir / "04_alignment" / "alignment_metadata.json",
        work_episode_dir / "03_diarization" / "speaker_segments.json",
        work_episode_dir / "03_diarization" / "speaker_metrics.json",
        work_episode_dir / "03_diarization" / "transcript_with_speakers.json",
    ]

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required artifacts: {missing}")


def load_aligned_utterances(work_episode_dir: Path) -> list[dict[str, Any]]:
    return _load_json(
        work_episode_dir / "04_alignment" / "aligned_utterances.json",
        list,
    )


def load_aligned_turns(work_episode_dir: Path) -> list[dict[str, Any]]:
    return _load_json(
        work_episode_dir / "04_alignment" / "aligned_turns.json",
        list,
    )


def load_aligned_words(work_episode_dir: Path) -> list[dict[str, Any]]:
    return _load_json(
        work_episode_dir / "04_alignment" / "aligned_words.json",
        list,
    )


def load_speakers_index(work_episode_dir: Path) -> dict[str, Any]:
    return _load_json(
        work_episode_dir / "04_alignment" / "speakers_index.json",
        dict,
    )


def load_alignment_metadata(work_episode_dir: Path) -> dict[str, Any]:
    return _load_json(
        work_episode_dir / "04_alignment" / "alignment_metadata.json",
        dict,
    )


def load_speaker_segments(work_episode_dir: Path) -> list[dict[str, Any]]:
    return _load_json(
        work_episode_dir / "03_diarization" / "speaker_segments.json",
        list,
    )


def load_speaker_metrics(work_episode_dir: Path) -> dict[str, Any]:
    return _load_json(
        work_episode_dir / "03_diarization" / "speaker_metrics.json",
        dict,
    )


def load_transcript_with_speakers(work_episode_dir: Path) -> list[dict[str, Any]]:
    return _load_json(
        work_episode_dir / "03_diarization" / "transcript_with_speakers.json",
        list,
    )


def load_transcript_preview_if_exists(work_episode_dir: Path) -> dict[str, Any] | None:
    path = work_episode_dir / "02_transcription" / "transcript_preview.json"
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return data


def ensure_identity_dir(work_episode_dir: Path) -> Path:
    identity_dir = work_episode_dir / "05_speaker_identity"
    identity_dir.mkdir(parents=True, exist_ok=True)
    return identity_dir


def get_work_episode_dir(episode_id: str) -> Path:
    return WORK_DIR / episode_id
