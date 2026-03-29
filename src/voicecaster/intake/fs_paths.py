# src/voicecaster/intake/fs_paths.py
from __future__ import annotations

from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_inputs_json_path() -> Path:
    return get_repo_root() / "inputs" / "inputs.json"


def get_work_root() -> Path:
    return get_repo_root() / "work"


def get_episode_workdir(episode_id: str) -> Path:
    return get_work_root() / episode_id


def get_episode_status_path(episode_id: str) -> Path:
    return get_episode_workdir(episode_id) / "status.json"


def get_episode_request_path(episode_id: str) -> Path:
    return get_episode_workdir(episode_id) / "00_intake" / "request.json"


def get_episode_normalized_source_path(episode_id: str) -> Path:
    return get_episode_workdir(episode_id) / "00_intake" / "normalized_source.json"


def get_episode_source_metadata_path(episode_id: str) -> Path:
    return get_episode_workdir(episode_id) / "00_intake" / "source_metadata.json"


def get_episode_intake_result_path(episode_id: str) -> Path:
    return get_episode_workdir(episode_id) / "00_intake" / "intake_result.json"


def get_episode_cleanup_path(episode_id: str) -> Path:
    return get_episode_workdir(episode_id) / "00_intake" / "cleanup.json"


def get_episode_events_log_path(episode_id: str) -> Path:
    return get_episode_workdir(episode_id) / "01_logs" / "events.jsonl"


def get_episode_report_path(episode_id: str) -> Path:
    return get_episode_workdir(episode_id) / "01_logs" / "report.json"


def get_episode_error_path(episode_id: str) -> Path:
    return get_episode_workdir(episode_id) / "01_logs" / "error.json"


def get_episode_temp_dir(episode_id: str) -> Path:
    return get_episode_workdir(episode_id) / "99_temp"


def get_episode_temp_audio_path(episode_id: str) -> Path:
    return get_episode_temp_dir(episode_id) / "source_audio.bin"
