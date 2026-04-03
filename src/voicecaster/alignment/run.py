from __future__ import annotations

import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .assign_segments import assign_speakers_to_segments
from .assign_words import assign_speakers_to_words
from .export_srt import write_speaker_srt
from .loader import (
    AlignmentInputError,
    build_alignment_paths,
    ensure_stage_dir,
    load_speaker_segments,
    load_transcript_segments,
    validate_required_inputs,
)
from .metrics import compute_alignment_metrics
from .normalizer import (
    AlignmentNormalizationError,
    normalize_speaker_segments,
    normalize_transcript_segments,
)
from .schemas import AlignmentMetrics, dataclass_to_dict
from .split_merge import (
    merge_adjacent_same_speaker_utterances,
    split_segments_into_utterances,
)

ALIGNMENT_ALGORITHM_VERSION = "04_alignment_v1"

NEAREST_SPEAKER_GAP_TOLERANCE = 0.35
MIN_WORDS_PER_SPLIT_CHUNK = 2
MIN_CHUNK_DURATION = 0.35
NOISE_BRIDGE_MAX_DURATION = 0.30
MERGE_SAME_SPEAKER_GAP_MAX = 0.60

HIGH_UNKNOWN_WORD_RATIO_WARNING = 0.01
HIGH_UNKNOWN_UTTERANCE_RATIO_WARNING = 0.01
HIGH_MULTI_SPEAKER_SEGMENT_RATIO_WARNING = 0.15

MAX_RETRIES = 10

WORKFLOW_NAME = "04_alignment"
TARGET_STATUS = "alignment"
SUCCESS_STATUS = "completed"
RUINED_STATUS = "ruined"


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_alignment() -> int:
    repo_root = resolve_repo_root()
    inputs_path = repo_root / "inputs" / "inputs.json"
    work_root = repo_root / "work"

    inputs_data = load_inputs_json(inputs_path)
    episode = select_next_alignment_episode(inputs_data)

    if episode is None:
        print("[04_alignment] No episode found with status='alignment'.")
        return 0

    episode_id = episode["id"]
    episode_root = work_root / episode_id
    logs_dir = episode_root / "01_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    status_path = episode_root / "status.json"
    report_path = logs_dir / "report.json"
    events_path = logs_dir / "events.jsonl"

    started_at = utc_now_iso()

    append_event(
        events_path,
        {
            "ts": started_at,
            "level": "info",
            "workflow": WORKFLOW_NAME,
            "episode_id": episode_id,
            "event": "workflow_started",
            "status_before": episode.get("status"),
        },
    )

    status_data = load_status_json(status_path, episode_id=episode_id)
    status_data = mark_workflow_started(status_data, started_at=started_at)
    save_status_json(status_path, status_data)

    try:
        result = process_episode(
            episode_id=episode_id,
            work_root=work_root,
        )

        finished_at = utc_now_iso()

        status_data = load_status_json(status_path, episode_id=episode_id)
        status_data = mark_workflow_completed(
            status_data=status_data,
            finished_at=finished_at,
            result=result,
        )
        save_status_json(status_path, status_data)

        update_inputs_episode(
            inputs_data=inputs_data,
            episode_id=episode_id,
            new_status=SUCCESS_STATUS,
            retries=0,
        )
        save_inputs_json(inputs_path, inputs_data)

        report_payload = build_success_report(
            episode=episode,
            result=result,
            started_at=started_at,
            finished_at=finished_at,
        )
        write_report(report_path, report_payload)

        append_event(
            events_path,
            {
                "ts": finished_at,
                "level": "info",
                "workflow": WORKFLOW_NAME,
                "episode_id": episode_id,
                "event": "workflow_completed",
                "status_after": SUCCESS_STATUS,
                "warnings": result["metrics"]["warnings"],
            },
        )

        print(f"[04_alignment] Success for episode_id={episode_id}")
        return 0

    except Exception as exc:
        finished_at = utc_now_iso()
        traceback_text = traceback.format_exc()

        status_data = load_status_json(status_path, episode_id=episode_id)
        status_data = mark_workflow_failed(
            status_data=status_data,
            finished_at=finished_at,
            error=exc,
            traceback_text=traceback_text,
        )
        save_status_json(status_path, status_data)

        current_retries = int(episode.get("retries", 0) or 0) + 1
        next_status = RUINED_STATUS if current_retries > MAX_RETRIES else TARGET_STATUS

        update_inputs_episode(
            inputs_data=inputs_data,
            episode_id=episode_id,
            new_status=next_status,
            retries=current_retries,
        )
        save_inputs_json(inputs_path, inputs_data)

        report_payload = build_failure_report(
            episode=episode,
            error=exc,
            traceback_text=traceback_text,
            started_at=started_at,
            finished_at=finished_at,
            retries=current_retries,
            next_status=next_status,
        )
        write_report(report_path, report_payload)

        append_event(
            events_path,
            {
                "ts": finished_at,
                "level": "error",
                "workflow": WORKFLOW_NAME,
                "episode_id": episode_id,
                "event": "workflow_failed",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "next_status": next_status,
                "retries": current_retries,
            },
        )

        print(f"[04_alignment] Failed for episode_id={episode_id}: {exc}")
        return 1


def process_episode(episode_id: str, work_root: Path) -> dict[str, Any]:
    paths = build_alignment_paths(work_root=work_root, episode_id=episode_id)
    ensure_stage_dir(paths)

    transcript_raw = load_transcript_segments(paths.transcript_segments_json)
    speakers_raw = load_speaker_segments(paths.speaker_segments_json)
    validate_required_inputs(transcript_raw, speakers_raw)

    transcript_segments = normalize_transcript_segments(transcript_raw)
    speaker_segments = normalize_speaker_segments(speakers_raw)

    aligned_words = assign_speakers_to_words(
        transcript_segments=transcript_segments,
        speaker_segments=speaker_segments,
        nearest_gap_tolerance=NEAREST_SPEAKER_GAP_TOLERANCE,
    )

    segment_assignments = assign_speakers_to_segments(
        transcript_segments=transcript_segments,
        aligned_words=aligned_words,
        speaker_segments=speaker_segments,
    )

    utterances = split_segments_into_utterances(
        transcript_segments=transcript_segments,
        aligned_words=aligned_words,
        segment_assignments=segment_assignments,
        min_words_per_chunk=MIN_WORDS_PER_SPLIT_CHUNK,
        min_chunk_duration=MIN_CHUNK_DURATION,
        noise_bridge_max_duration=NOISE_BRIDGE_MAX_DURATION,
    )

    utterances = merge_adjacent_same_speaker_utterances(
        utterances=utterances,
        max_gap=MERGE_SAME_SPEAKER_GAP_MAX,
    )

    metrics = compute_alignment_metrics(
        transcript_segments=transcript_segments,
        speaker_segments=speaker_segments,
        aligned_words=aligned_words,
        utterances=utterances,
        algorithm_version=ALIGNMENT_ALGORITHM_VERSION,
        high_unknown_word_ratio_warning=HIGH_UNKNOWN_WORD_RATIO_WARNING,
        high_unknown_utterance_ratio_warning=HIGH_UNKNOWN_UTTERANCE_RATIO_WARNING,
        high_multi_speaker_segment_ratio_warning=HIGH_MULTI_SPEAKER_SEGMENT_RATIO_WARNING,
    )

    _write_json(
        paths.aligned_words_json,
        {
            "episode_id": episode_id,
            "algorithm_version": ALIGNMENT_ALGORITHM_VERSION,
            "words": dataclass_to_dict(aligned_words),
        },
    )

    _write_json(
        paths.aligned_utterances_json,
        {
            "episode_id": episode_id,
            "algorithm_version": ALIGNMENT_ALGORITHM_VERSION,
            "utterances": dataclass_to_dict(utterances),
        },
    )

    write_speaker_srt(paths.subtitles_speakers_srt, utterances)

    _write_json(paths.alignment_metadata_json, dataclass_to_dict(metrics))

    _write_json(
        paths.alignment_preview_json,
        _build_alignment_preview(episode_id, utterances, metrics),
    )

    _write_json(
        paths.alignment_result_json,
        {
            "result": "success",
            "episode_id": episode_id,
            "algorithm_version": ALIGNMENT_ALGORITHM_VERSION,
            "files_generated": [
                str(paths.aligned_words_json.name),
                str(paths.aligned_utterances_json.name),
                str(paths.subtitles_speakers_srt.name),
                str(paths.alignment_metadata_json.name),
                str(paths.alignment_result_json.name),
                str(paths.alignment_preview_json.name),
            ],
            "warnings": metrics.warnings,
        },
    )

    return {
        "episode_id": episode_id,
        "algorithm_version": ALIGNMENT_ALGORITHM_VERSION,
        "metrics": dataclass_to_dict(metrics),
    }


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_inputs_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"inputs.json not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise RuntimeError("inputs/inputs.json must contain a JSON list")

    return data


def save_inputs_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def select_next_alignment_episode(inputs_data: list[dict[str, Any]]) -> dict[str, Any] | None:
    for episode in inputs_data:
        if not isinstance(episode, dict):
            continue
        if episode.get("status") == TARGET_STATUS:
            episode.setdefault("retries", 0)
            return episode
    return None


def update_inputs_episode(
    inputs_data: list[dict[str, Any]],
    episode_id: str,
    new_status: str,
    retries: int,
) -> None:
    for episode in inputs_data:
        if isinstance(episode, dict) and episode.get("id") == episode_id:
            episode["status"] = new_status
            episode["retries"] = retries
            return

    raise RuntimeError(f"Episode not found in inputs.json: {episode_id}")


def load_status_json(path: Path, episode_id: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "episode_id": episode_id,
            "status": TARGET_STATUS,
            "current_stage": "alignment",
            "alignment": {},
        }

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise RuntimeError(f"status.json must contain a JSON object: {path}")

    data.setdefault("episode_id", episode_id)
    data.setdefault("alignment", {})
    return data


def save_status_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def mark_workflow_started(status_data: dict[str, Any], started_at: str) -> dict[str, Any]:
    status_data["status"] = TARGET_STATUS
    status_data["current_stage"] = "alignment"
    status_data["alignment"] = {
        **status_data.get("alignment", {}),
        "started_at": started_at,
        "finished_at": None,
        "result": "running",
        "algorithm_version": ALIGNMENT_ALGORITHM_VERSION,
        "warnings": [],
    }
    return status_data


def mark_workflow_completed(
    status_data: dict[str, Any],
    finished_at: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    metrics = result["metrics"]

    status_data["status"] = SUCCESS_STATUS
    status_data["current_stage"] = "alignment"
    status_data["alignment"] = {
        **status_data.get("alignment", {}),
        "finished_at": finished_at,
        "result": "success",
        "algorithm_version": result["algorithm_version"],
        "word_assignment_ratio": metrics["quality_metrics"]["word_assignment_ratio"],
        "utterance_assignment_ratio": metrics["quality_metrics"]["utterance_assignment_ratio"],
        "unknown_word_ratio": metrics["quality_metrics"]["unknown_word_ratio"],
        "unknown_utterance_ratio": metrics["quality_metrics"]["unknown_utterance_ratio"],
        "warnings": metrics["warnings"],
    }
    return status_data


def mark_workflow_failed(
    status_data: dict[str, Any],
    finished_at: str,
    error: Exception,
    traceback_text: str,
) -> dict[str, Any]:
    status_data["current_stage"] = "alignment"
    status_data["alignment"] = {
        **status_data.get("alignment", {}),
        "finished_at": finished_at,
        "result": "failed",
        "algorithm_version": ALIGNMENT_ALGORITHM_VERSION,
        "error_type": error.__class__.__name__,
        "error": str(error),
    }
    status_data["last_error"] = {
        "stage": "alignment",
        "finished_at": finished_at,
        "error_type": error.__class__.__name__,
        "error": str(error),
        "traceback": traceback_text,
    }
    return status_data


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def build_success_report(
    episode: dict[str, Any],
    result: dict[str, Any],
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    metrics = result["metrics"]
    return {
        "workflow": WORKFLOW_NAME,
        "episode_id": episode.get("id"),
        "podcast_title": episode.get("podcast_title"),
        "episode_title": episode.get("episode_title"),
        "status_before": TARGET_STATUS,
        "status_after": SUCCESS_STATUS,
        "result": "success",
        "started_at": started_at,
        "finished_at": finished_at,
        "algorithm_version": result["algorithm_version"],
        "metrics": {
            "word_assignment_ratio": metrics["quality_metrics"]["word_assignment_ratio"],
            "utterance_assignment_ratio": metrics["quality_metrics"]["utterance_assignment_ratio"],
            "unknown_word_ratio": metrics["quality_metrics"]["unknown_word_ratio"],
            "unknown_utterance_ratio": metrics["quality_metrics"]["unknown_utterance_ratio"],
            "aligned_utterances": metrics["output_summary"]["aligned_utterances"],
            "aligned_words": metrics["output_summary"]["aligned_words"],
        },
        "warnings": metrics["warnings"],
    }


def build_failure_report(
    episode: dict[str, Any],
    error: Exception,
    traceback_text: str,
    started_at: str,
    finished_at: str,
    retries: int,
    next_status: str,
) -> dict[str, Any]:
    return {
        "workflow": WORKFLOW_NAME,
        "episode_id": episode.get("id"),
        "podcast_title": episode.get("podcast_title"),
        "episode_title": episode.get("episode_title"),
        "status_before": TARGET_STATUS,
        "status_after": next_status,
        "result": "failed",
        "started_at": started_at,
        "finished_at": finished_at,
        "algorithm_version": ALIGNMENT_ALGORITHM_VERSION,
        "retries": retries,
        "error_type": error.__class__.__name__,
        "error": str(error),
        "traceback": traceback_text,
    }


def _build_alignment_preview(
    episode_id: str,
    utterances: list[Any],
    metrics: AlignmentMetrics,
    preview_size: int = 50,
) -> dict[str, Any]:
    speaker_counts: dict[str, int] = {}
    for utt in utterances:
        speaker_counts[utt.speaker] = speaker_counts.get(utt.speaker, 0) + 1

    return {
        "episode_id": episode_id,
        "algorithm_version": metrics.algorithm_version,
        "summary": {
            "aligned_utterances": len(utterances),
            "warnings": metrics.warnings,
            "speakers": speaker_counts,
        },
        "utterances_preview": dataclass_to_dict(utterances[:preview_size]),
    }


def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        raise ValueError("Output path cannot be None")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(run_alignment())
