from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_STATUS_PAYLOAD: dict[str, Any] = {
    "episode_id": None,
    "phase": "preaudit",
    "status": "idle",
    "current_workflow": None,
    "current_step": None,
    "preaudit_stage": "not_started",
    "started_at": None,
    "updated_at": None,
    "finished_at": None,
    "result": None,
    "stage_outputs": {
        "intake_done": False,
        "transcription_done": False,
        "diarization_done": False,
        "alignment_done": False,
        "review_prepare_done": False,
    },
    "stage_attempts": {
        "intake": 0,
        "transcription": 0,
        "diarization": 0,
        "alignment": 0,
        "review_prepare": 0,
    },
    "artifacts": {
        "source_audio": None,
        "source_metadata": None,
        "vad_segments": None,
        "whisper_raw_segments": None,
        "whisperx_aligned_segments": None,
        "transcript_segments": None,
        "speaker_segments": None,
        "speaker_timeline": None,
        "aligned_transcript_segments": None,
        "doubtful_segments": None,
        "redecoded_segments": None,
        "full_transcript_speakers": None,
        "speakers_auto_dir": None,
        "speakers_reviewed_dir": None,
        "episode_json": None,
        "report_json": None,
    },
    "last_error": None,
}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def default_status_payload(episode_id: str) -> dict[str, Any]:
    payload = deepcopy(DEFAULT_STATUS_PAYLOAD)
    payload["episode_id"] = episode_id
    payload["updated_at"] = utc_now_iso()
    return payload


def load_status_json(work_dir: Path, episode_id: str) -> dict[str, Any]:
    status_path = work_dir / "status.json"
    if not status_path.exists():
        return default_status_payload(episode_id)

    raw = status_path.read_text(encoding="utf-8").strip()
    if not raw:
        return default_status_payload(episode_id)

    loaded = json.loads(raw)

    merged = default_status_payload(episode_id)
    merged.update(
        {
            k: v
            for k, v in loaded.items()
            if k not in {"stage_outputs", "stage_attempts", "artifacts"}
        }
    )
    merged["stage_outputs"].update(loaded.get("stage_outputs", {}))
    merged["stage_attempts"].update(loaded.get("stage_attempts", {}))
    merged["artifacts"].update(loaded.get("artifacts", {}))
    merged["last_error"] = loaded.get("last_error")
    merged["updated_at"] = utc_now_iso()
    return merged


def save_status_json(work_dir: Path, payload: dict[str, Any]) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = utc_now_iso()
    (work_dir / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def mark_workflow_started(
    payload: dict[str, Any],
    workflow_name: str,
    stage_name: str,
    step_name: str,
) -> dict[str, Any]:
    payload["status"] = "running"
    payload["current_workflow"] = workflow_name
    payload["current_step"] = step_name
    payload["preaudit_stage"] = stage_name
    if payload.get("started_at") is None:
        payload["started_at"] = utc_now_iso()
    payload["stage_attempts"][stage_name] = int(
        payload["stage_attempts"].get(stage_name, 0)
    ) + 1
    return payload


def mark_workflow_completed(
    payload: dict[str, Any],
    workflow_name: str,
    stage_name: str,
) -> dict[str, Any]:
    payload["status"] = "completed"
    payload["current_workflow"] = workflow_name
    payload["current_step"] = "done"
    payload["preaudit_stage"] = stage_name
    payload["finished_at"] = utc_now_iso()

    flag_name = f"{stage_name}_done"
    if flag_name in payload["stage_outputs"]:
        payload["stage_outputs"][flag_name] = True

    return payload


def mark_workflow_failed(
    payload: dict[str, Any],
    workflow_name: str,
    stage_name: str,
    step_name: str,
    message: str,
) -> dict[str, Any]:
    payload["status"] = "failed"
    payload["current_workflow"] = workflow_name
    payload["current_step"] = step_name
    payload["preaudit_stage"] = stage_name
    payload["result"] = "failed"
    payload["finished_at"] = utc_now_iso()
    payload["last_error"] = {
        "workflow": workflow_name,
        "step": step_name,
        "at": utc_now_iso(),
        "message": message,
    }
    return payload
