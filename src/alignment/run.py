from __future__ import annotations

import json
import traceback
from pathlib import Path

from voicecaster.config import RUNTIME_CONTROL_PATH
from voicecaster.reporting import utc_now_iso, write_json
from voicecaster.runtime_control import should_run_now, update_runtime_control
from voicecaster.status_manager import (
    load_status_json,
    mark_workflow_completed,
    mark_workflow_failed,
    mark_workflow_started,
    save_status_json,
)
from voicecaster.work_layout import ensure_work_layout

from .assign_segments import assign_segments
from .assign_words import assign_words
from .export_srt import export_srt
from .loader import load_speaker_segments, load_transcript_preview
from .metrics import compute_metrics
from .normalizer import sort_by_time, validate_monotonic
from .split_merge import build_utterances, merge_adjacent_same_speaker_utterances

WORKFLOW_NAME = "04_alignment"
WORKFLOW_STAGE = "alignment"
ALGORITHM_VERSION = "04_alignment_v1"


def _repo_root() -> Path:
    """
    Asume:
    src/voicecaster/alignment/run.py
    -> repo root = 4 niveles arriba desde este archivo
    """
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_event(events_path: Path, payload: dict) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_inputs(inputs_path: Path) -> list[dict]:
    data = _read_json(inputs_path)
    if not isinstance(data, list):
        raise ValueError("inputs/inputs.json debe contener una lista")
    return data


def _save_inputs(inputs_path: Path, data: list[dict]) -> None:
    write_json(inputs_path, data)


def _select_alignment_episode(inputs_path: Path) -> dict:
    episodes = _load_inputs(inputs_path)

    for episode in episodes:
        if episode.get("status") == WORKFLOW_STAGE:
            return episode

    raise RuntimeError("No hay episodios con status='alignment'")


def _update_inputs_status(inputs_path: Path, episode_id: str, new_status: str) -> None:
    episodes = _load_inputs(inputs_path)
    updated = False

    for episode in episodes:
        if episode.get("id") == episode_id:
            episode["status"] = new_status
            episode["retries"] = 0
            updated = True
            break

    if not updated:
        raise ValueError(f"No se encontró episodio {episode_id} en inputs/inputs.json")

    _save_inputs(inputs_path, episodes)


def _write_aligned_words(path: Path, episode_id: str, aligned_words: list) -> None:
    write_json(
        path,
        {
            "episode_id": episode_id,
            "algorithm_version": ALGORITHM_VERSION,
            "words": [w.to_dict() for w in aligned_words],
        },
    )


def _write_aligned_utterances(path: Path, episode_id: str, utterances: list) -> None:
    write_json(
        path,
        {
            "episode_id": episode_id,
            "algorithm_version": ALGORITHM_VERSION,
            "utterances": [u.to_dict() for u in utterances],
        },
    )


def _write_alignment_preview(path: Path, episode_id: str, utterances: list, metadata: dict) -> None:
    write_json(
        path,
        {
            "episode_id": episode_id,
            "algorithm_version": ALGORITHM_VERSION,
            "preview_utterances": [u.to_dict() for u in utterances[:50]],
            "summary": {
                "aligned_utterances": metadata["output_summary"]["aligned_utterances"],
                "unknown_utterances": metadata["output_summary"]["unknown_utterances"],
                "warnings": metadata["warnings"],
            },
        },
    )


def _write_alignment_result(
    path: Path,
    status_before: str,
    status_after: str,
    warnings: list[str],
    files_generated: list[str],
) -> None:
    write_json(
        path,
        {
            "result": "success",
            "status_before": status_before,
            "status_after": status_after,
            "message": "Alignment completed successfully.",
            "workflow": WORKFLOW_NAME,
            "algorithm_version": ALGORITHM_VERSION,
            "warnings": warnings,
            "files_generated": files_generated,
            "finished_at": utc_now_iso(),
        },
    )


def _update_status_json_success(
    status_path: Path,
    episode_id: str,
    metadata: dict,
) -> None:
    status = load_status_json(status_path)

    status["episode_id"] = episode_id
    status["status"] = "completed"
    status["current_stage"] = WORKFLOW_STAGE

    status.setdefault("alignment", {})
    status["alignment"]["result"] = "success"
    status["alignment"]["algorithm_version"] = ALGORITHM_VERSION
    status["alignment"]["finished_at"] = utc_now_iso()
    status["alignment"]["word_assignment_ratio"] = metadata["quality_metrics"]["word_assignment_ratio"]
    status["alignment"]["utterance_assignment_ratio"] = metadata["quality_metrics"]["utterance_assignment_ratio"]
    status["alignment"]["warnings"] = metadata["warnings"]

    save_status_json(status_path, status)


def _update_status_json_failure(
    status_path: Path,
    episode_id: str,
    error_message: str,
) -> None:
    status = load_status_json(status_path)

    status["episode_id"] = episode_id
    status["status"] = WORKFLOW_STAGE
    status["current_stage"] = WORKFLOW_STAGE

    status.setdefault("alignment", {})
    status["alignment"]["result"] = "failed"
    status["alignment"]["finished_at"] = utc_now_iso()
    status["alignment"]["error"] = error_message

    save_status_json(status_path, status)


def run() -> None:
    repo_root = _repo_root()
    inputs_path = repo_root / "inputs" / "inputs.json"

    if not should_run_now(RUNTIME_CONTROL_PATH, WORKFLOW_NAME):
        return

    episode = _select_alignment_episode(inputs_path)
    episode_id = episode["id"]
    status_before = episode["status"]

    work_dir = repo_root / "work" / episode_id
    ensure_work_layout(work_dir)

    alignment_dir = work_dir / "04_alignment"
    logs_dir = work_dir / "01_logs"

    transcript_path = work_dir / "02_transcription" / "transcript_preview.json"
    speaker_path = work_dir / "03_diarization" / "speaker_segments.json"
    status_path = work_dir / "status.json"
    report_path = logs_dir / "report.json"
    events_path = logs_dir / "events.jsonl"
    error_path = logs_dir / "error.json"

    alignment_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    mark_workflow_started(
        status_path=status_path,
        workflow_name=WORKFLOW_NAME,
        stage_name=WORKFLOW_STAGE,
        episode_id=episode_id,
    )

    _append_event(
        events_path,
        {
            "ts": utc_now_iso(),
            "event": "alignment_started",
            "workflow": WORKFLOW_NAME,
            "episode_id": episode_id,
        },
    )

    try:
        transcript_segments = load_transcript_preview(transcript_path)
        speaker_segments = load_speaker_segments(speaker_path)

        transcript_segments = sort_by_time(transcript_segments)
        speaker_segments = sort_by_time(speaker_segments)

        validate_monotonic(transcript_segments, "transcript_segments")
        validate_monotonic(speaker_segments, "speaker_segments")

        aligned_words = assign_words(
            transcript_segments=transcript_segments,
            speaker_segments=speaker_segments,
            tolerance=0.35,
        )

        segment_assignments = assign_segments(
            transcript_segments=transcript_segments,
            aligned_words=aligned_words,
            speaker_segments=speaker_segments,
        )

        utterances = build_utterances(aligned_words)
        utterances = merge_adjacent_same_speaker_utterances(
            utterances=utterances,
            max_gap=0.60,
        )

        metadata = compute_metrics(
            transcript_segments=transcript_segments,
            speaker_segments=speaker_segments,
            aligned_words=aligned_words,
            utterances=utterances,
            algorithm_version=ALGORITHM_VERSION,
        )
        metadata_dict = metadata.to_dict()

        _write_aligned_words(
            alignment_dir / "aligned_words.json",
            episode_id,
            aligned_words,
        )
        _write_aligned_utterances(
            alignment_dir / "aligned_utterances.json",
            episode_id,
            utterances,
        )
        write_json(
            alignment_dir / "segment_assignments.json",
            {
                "episode_id": episode_id,
                "algorithm_version": ALGORITHM_VERSION,
                "segments": segment_assignments,
            },
        )
        export_srt(
            alignment_dir / "subtitles_speakers.srt",
            utterances,
        )
        write_json(
            alignment_dir / "alignment_metadata.json",
            metadata_dict,
        )
        _write_alignment_preview(
            alignment_dir / "alignment_preview.json",
            episode_id,
            utterances,
            metadata_dict,
        )
        _write_alignment_result(
            alignment_dir / "alignment_result.json",
            status_before=status_before,
            status_after="completed",
            warnings=metadata_dict["warnings"],
            files_generated=[
                "aligned_words.json",
                "aligned_utterances.json",
                "segment_assignments.json",
                "subtitles_speakers.srt",
                "alignment_metadata.json",
                "alignment_preview.json",
                "alignment_result.json",
            ],
        )

        _update_status_json_success(
            status_path=status_path,
            episode_id=episode_id,
            metadata=metadata_dict,
        )

        _update_inputs_status(
            inputs_path=inputs_path,
            episode_id=episode_id,
            new_status="completed",
        )

        mark_workflow_completed(
            status_path=status_path,
            workflow_name=WORKFLOW_NAME,
            episode_id=episode_id,
            result="alignment_completed",
        )

        write_json(
            report_path,
            {
                "workflow": WORKFLOW_NAME,
                "episode_id": episode_id,
                "status_before": status_before,
                "status_after": "completed",
                "result": "success",
                "algorithm_version": ALGORITHM_VERSION,
                "metrics": metadata_dict["quality_metrics"],
                "warnings": metadata_dict["warnings"],
                "finished_at": utc_now_iso(),
            },
        )

        _append_event(
            events_path,
            {
                "ts": utc_now_iso(),
                "event": "alignment_completed",
                "workflow": WORKFLOW_NAME,
                "episode_id": episode_id,
                "word_assignment_ratio": metadata_dict["quality_metrics"]["word_assignment_ratio"],
                "utterance_assignment_ratio": metadata_dict["quality_metrics"]["utterance_assignment_ratio"],
            },
        )

        update_runtime_control(RUNTIME_CONTROL_PATH, WORKFLOW_NAME)

    except Exception as exc:
        error_payload = {
            "workflow": WORKFLOW_NAME,
            "episode_id": episode_id,
            "result": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at": utc_now_iso(),
        }
        write_json(error_path, error_payload)

        _update_status_json_failure(
            status_path=status_path,
            episode_id=episode_id,
            error_message=str(exc),
        )

        mark_workflow_failed(
            status_path=status_path,
            workflow_name=WORKFLOW_NAME,
            episode_id=episode_id,
            error_message=str(exc),
        )

        _append_event(
            events_path,
            {
                "ts": utc_now_iso(),
                "event": "alignment_failed",
                "workflow": WORKFLOW_NAME,
                "episode_id": episode_id,
                "error": str(exc),
            },
        )

        raise


if __name__ == "__main__":
    run()
