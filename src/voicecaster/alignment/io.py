from __future__ import annotations

import json
from pathlib import Path
from typing import Any

INPUTS_PATH = Path("inputs/inputs.json")
WORK_DIR = Path("work")


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
        if ep.get("status") == "alignment":
            return ep
    raise RuntimeError("No episode with status=alignment found.")


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


def ensure_required_paths(work_episode_dir: Path) -> None:
    required_paths = [
        work_episode_dir / "02_transcription" / "transcript_segments.json",
        work_episode_dir / "02_transcription" / "full_transcript.srt",
        work_episode_dir / "02_transcription" / "full_transcript.txt",
        work_episode_dir / "02_transcription" / "transcription_metadata.json",
        work_episode_dir / "03_diarization" / "speaker_segments.json",
        work_episode_dir / "03_diarization" / "transcript_with_speakers.json",
        work_episode_dir / "03_diarization" / "speaker_metrics.json",
        work_episode_dir / "03_diarization" / "diarization_metadata.json",
    ]

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required upstream artifacts: {missing}")


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise RuntimeError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_transcript_segments(work_episode_dir: Path) -> list[dict[str, Any]]:
    path = work_episode_dir / "02_transcription" / "transcript_segments.json"
    data = _load_json(path)
    if not isinstance(data, list):
        raise RuntimeError(f"Invalid JSON list in {path}")
    return data


def load_transcript_with_speakers(work_episode_dir: Path) -> list[dict[str, Any]]:
    path = work_episode_dir / "03_diarization" / "transcript_with_speakers.json"
    data = _load_json(path)
    if not isinstance(data, list):
        raise RuntimeError(f"Invalid JSON list in {path}")
    return data


def load_speaker_segments(work_episode_dir: Path) -> list[dict[str, Any]]:
    path = work_episode_dir / "03_diarization" / "speaker_segments.json"
    data = _load_json(path)
    if not isinstance(data, list):
        raise RuntimeError(f"Invalid JSON list in {path}")
    return data


def load_speaker_metrics(work_episode_dir: Path) -> dict[str, Any]:
    path = work_episode_dir / "03_diarization" / "speaker_metrics.json"
    data = _load_json(path)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid JSON object in {path}")
    return data


def load_diarization_metadata(work_episode_dir: Path) -> dict[str, Any]:
    path = work_episode_dir / "03_diarization" / "diarization_metadata.json"
    data = _load_json(path)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid JSON object in {path}")
    return data
