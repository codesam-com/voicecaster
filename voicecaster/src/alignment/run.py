from __future__ import annotations

import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .assign_segments import assign_speakers_to_segments
from .assign_words import assign_speakers_to_words
from .export_srt import write_speaker_srt
from .loader import (
    build_alignment_paths,
    load_speaker_segments_json,
    load_transcript_preview_json,
    validate_raw_speaker_segments,
    validate_raw_transcript_preview,
    validate_required_files,
)
from .metrics import build_alignment_preview, compute_alignment_metadata
from .normalizer import (
    normalize_speaker_segments,
    normalize_transcript_segments,
    validate_monotonic_timeline,
)
from .schemas import AlignmentConfig
from .split_merge import build_aligned_utterances

# Ajusta estos imports a tu proyecto real
from voicecaster.src.config import INPUTS_JSON_PATH, WORK_DIR
from voicecaster.src.reporting import append_event, utc_now_iso, write_json
from voicecaster.src.status_manager import (
    load_status_json,
    mark_workflow_completed,
    mark_workflow_failed,
    mark_workflow_started,
    save_status_json,
)
from voicecaster.src.work_layout import ensure_work_layout


WORKFLOW_NAME = "04_alignment"
STAGE_NAME = "alignment"
ALGORITHM_VERSION = "04_alignment_v1"


def run_alignment() -> None:
    episode = select_next_alignment_episode(INPUTS_JSON_PATH)
    if episode is None:
        print("No pending episode found for alignment")
        return

    process_episode_alignment(episode)


def process_episode_alignment(episode: dict[str, Any]) -> None:
    episode_id = str(episode["id"])
    episode_dir = Path(WORK_DIR) / episode_id

    ensure_work_layout(episode_id)

    started_at = utc_now_iso()
    status_before = episode.get("status", "")

    try:
        mark_workflow_started(
            episode_id=episode_id,
            workflow_name=WORKFLOW_NAME,
            stage_name=STAGE_NAME,
        )

        append_event(
            episode_id=episode_id,
            event={
                "ts": utc_now_iso(),
                "level": "info",
                "workflow": WORKFLOW_NAME,
                "stage": STAGE_NAME,
                "message": "Alignment started",
            },
        )

        paths = build_alignment_paths(episode_dir)
        validate_required_files(paths)

        raw_transcript = load_transcript_preview_json(paths.transcript_preview_path)
        raw_speakers = load_speaker_segments_json(paths.speaker_segments_path)

        validate_raw_transcript_preview(raw_transcript)
        validate_raw_speaker_segments(raw_speakers)

        transcript_segments = normalize_transcript_segments(raw_transcript)
        speaker_segments = normalize_speaker_segments(raw_speakers)
        validate_monotonic_timeline(transcript_segments, speaker_segments)

        config = AlignmentConfig()

        aligned_words = assign_speakers_to_words(
            transcript_segments=transcript_segments,
            speaker_segments=speaker_segments,
            config=config,
        )

        segment_assignments = assign_speakers_to_segments(
            transcript_segments=transcript_segments,
            aligned_words=aligned_words,
            speaker_segments=speaker_segments,
        )

        utterances = build_aligned_utterances(
            transcript_segments=transcript_segments,
            aligned_words=aligned_words,
            segment_assignments=segment_assignments,
            config=config,
        )

        metadata = compute_alignment_metadata(
            transcript_segments=transcript_segments,
            speaker_segments=speaker_segments,
            aligned_words=aligned_words,
            utterances=utterances,
        )

        preview = build_alignment_preview(
            utterances=utterances,
            metadata=metadata,
            preview_limit=config.preview_limit,
        )

        write_alignment_outputs(
            paths=paths,
            aligned_words=aligned_words,
            utterances=utterances,
            metadata=metadata,
            preview=preview,
            status_before=status_before,
        )

        update_status_json_after_success(
            episode_id=episode_id,
            algorithm_version=ALGORITHM_VERSION,
            metadata=metadata,
        )

        update_inputs_status(
            inputs_json_path=INPUTS_JSON_PATH,
            episode_id=episode_id,
            new_status="completed",
            retries_reset=True,
        )

        mark_workflow_completed(
            episode_id=episode_id,
            workflow_name=WORKFLOW_NAME,
            stage_name=STAGE_NAME,
            result="alignment_completed",
        )

        append_event(
            episode_id=episode_id,
            event={
                "ts": utc_now_iso(),
                "level": "info",
                "workflow": WORKFLOW_NAME,
                "stage": STAGE_NAME,
                "message": "Alignment completed successfully",
                "metrics": {
                    "word_assignment_ratio": metadata["quality_metrics"]["word_assignment_ratio"],
                    "utterance_assignment_ratio": metadata["quality_metrics"]["utterance_assignment_ratio"],
                    "aligned_utterances": metadata["output_summary"]["aligned_utterances"],
                },
                "warnings": metadata.get("warnings", []),
            },
        )

    except Exception as exc:
        error_trace = traceback.format_exc()

        append_event(
            episode_id=episode_id,
            event={
                "ts": utc_now_iso(),
                "level": "error",
                "workflow": WORKFLOW_NAME,
                "stage": STAGE_NAME,
                "message": str(exc),
            },
        )

        write_alignment_error(
            episode_dir=episode_dir,
            error_payload={
                "workflow": WORKFLOW_NAME,
                "stage": STAGE_NAME,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": error_trace,
                "failed_at": utc_now_iso(),
            },
        )

        incremented_retries = update_inputs_status_on_failure(
            inputs_json_path=INPUTS_JSON_PATH,
            episode_id=episode_id,
            max_retries=10,
        )

        mark_workflow_failed(
            episode_id=episode_id,
            workflow_name=WORKFLOW_NAME,
            stage_name=STAGE_NAME,
            error_message=str(exc),
        )

        update_status_json_after_failure(
            episode_id=episode_id,
            error_message=str(exc),
            retries=incremented_retries,
        )

        raise


def select_next_alignment_episode(inputs_json_path: Path) -> dict[str, Any] | None:
    payload = load_json(inputs_json_path)
    if not isinstance(payload, list):
        raise ValueError("inputs.json must contain a list of episodes")

    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("status") == "alignment":
            return item

    return None


def write_alignment_outputs(
    *,
    paths,
    aligned_words,
    utterances,
    metadata: dict[str, Any],
    preview: dict[str, Any],
    status_before: str,
) -> None:
    paths.alignment_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        paths.aligned_words_path,
        {
            "algorithm_version": ALGORITHM_VERSION,
            "words": [word.to_dict() for word in aligned_words],
        },
    )

    write_json(
        paths.aligned_utterances_path,
        {
            "algorithm_version": ALGORITHM_VERSION,
            "utterances": [utterance.to_dict() for utterance in utterances],
        },
    )

    write_json(paths.alignment_metadata_path, metadata)
    write_json(paths.alignment_preview_path, preview)

    write_json(
        paths.alignment_result_path,
        {
            "result": "success",
            "status_before": status_before,
            "status_after": "completed",
            "workflow": WORKFLOW_NAME,
            "stage": STAGE_NAME,
            "message": "Alignment completed successfully.",
            "algorithm_version": ALGORITHM_VERSION,
            "files_generated": [
                "aligned_words.json",
                "aligned_utterances.json",
                "alignment_metadata.json",
                "alignment_preview.json",
                "alignment_result.json",
                "subtitles_speakers.srt",
            ],
            "warnings": metadata.get("warnings", []),
            "finished_at": utc_now_iso(),
        },
    )

    write_speaker_srt(paths.subtitles_speakers_path, utterances)


def write_alignment_error(*, episode_dir: Path, error_payload: dict[str, Any]) -> None:
    logs_dir = episode_dir / "01_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_json(logs_dir / "error.json", error_payload)


def update_status_json_after_success(
    *,
    episode_id: str,
    algorithm_version: str,
    metadata: dict[str, Any],
) -> None:
    status = load_status_json(episode_id)

    status["status"] = "completed"
    status["current_stage"] = STAGE_NAME
    status["alignment"] = {
        "result": "success",
        "algorithm_version": algorithm_version,
        "finished_at": utc_now_iso(),
        "word_assignment_ratio": metadata["quality_metrics"]["word_assignment_ratio"],
        "utterance_assignment_ratio": metadata["quality_metrics"]["utterance_assignment_ratio"],
        "unknown_word_ratio": metadata["quality_metrics"]["unknown_word_ratio"],
        "unknown_utterance_ratio": metadata["quality_metrics"]["unknown_utterance_ratio"],
        "warnings": metadata.get("warnings", []),
    }

    save_status_json(episode_id, status)


def update_status_json_after_failure(
    *,
    episode_id: str,
    error_message: str,
    retries: int,
) -> None:
    status = load_status_json(episode_id)

    status["status"] = "alignment"
    status["current_stage"] = STAGE_NAME
    status["alignment"] = {
        "result": "failed",
        "finished_at": utc_now_iso(),
        "error": error_message,
        "retries": retries,
    }

    save_status_json(episode_id, status)


def update_inputs_status(
    *,
    inputs_json_path: Path,
    episode_id: str,
    new_status: str,
    retries_reset: bool,
) -> None:
    payload = load_json(inputs_json_path)
    if not isinstance(payload, list):
        raise ValueError("inputs.json must contain a list of episodes")

    updated = False
    for item in payload:
        if isinstance(item, dict) and str(item.get("id")) == episode_id:
            item["status"] = new_status
            if retries_reset:
                item["retries"] = 0
            updated = True
            break

    if not updated:
        raise ValueError(f"Episode not found in inputs.json: {episode_id}")

    write_json(inputs_json_path, payload)


def update_inputs_status_on_failure(
    *,
    inputs_json_path: Path,
    episode_id: str,
    max_retries: int,
) -> int:
    payload = load_json(inputs_json_path)
    if not isinstance(payload, list):
        raise ValueError("inputs.json must contain a list of episodes")

    updated = False
    new_retries = 0

    for item in payload:
        if isinstance(item, dict) and str(item.get("id")) == episode_id:
            retries = int(item.get("retries", 0)) + 1
            item["retries"] = retries
            item["status"] = "ruined" if retries > max_retries else "alignment"
            new_retries = retries
            updated = True
            break

    if not updated:
        raise ValueError(f"Episode not found in inputs.json: {episode_id}")

    write_json(inputs_json_path, payload)
    return new_retries


def load_json(path: Path) -> Any:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
