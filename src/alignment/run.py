from __future__ import annotations

import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

from .assign_segments import assign_segments
from .assign_words import assign_words
from .export_srt import export_srt
from .loader import load_transcript_preview, load_speaker_segments
from .metrics import compute_metrics
from .normalizer import sort_by_time, validate_monotonic
from .split_merge import build_utterances, merge_adjacent_same_speaker_utterances

ALGORITHM_VERSION = "04_alignment_v1"


# =========================
# GENERIC HELPERS
# =========================

def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# =========================
# INPUT SELECTION
# =========================

def select_alignment_episode(inputs_path: Path) -> dict:
    data = read_json(inputs_path)
    if not isinstance(data, list):
        raise ValueError("inputs.json debe contener una lista")

    for episode in data:
        if episode.get("status") == "alignment":
            return episode

    raise RuntimeError("No hay episodios con status='alignment'")


def update_episode_status_in_inputs(inputs_path: Path, episode_id: str, new_status: str) -> None:
    data = read_json(inputs_path)
    updated = False

    for episode in data:
        if episode.get("id") == episode_id:
            episode["status"] = new_status
            episode["retries"] = 0
            updated = True
            break

    if not updated:
        raise ValueError(f"No se encontró episodio {episode_id} en inputs.json")

    write_json(inputs_path, data)


# =========================
# STATUS / REPORT HELPERS
# =========================

def load_status_json(path: Path) -> dict:
    if path.exists():
        return read_json(path)
    return {}


def save_status_json(path: Path, data: dict) -> None:
    write_json(path, data)


def mark_alignment_started(status_path: Path, episode_id: str) -> None:
    status = load_status_json(status_path)
    status["episode_id"] = episode_id
    status["status"] = "alignment"
    status["current_stage"] = "alignment"
    status.setdefault("alignment", {})
    status["alignment"]["started_at"] = utc_now_iso()
    status["alignment"]["result"] = "running"
    save_status_json(status_path, status)


def mark_alignment_completed(status_path: Path, metadata: dict) -> None:
    status = load_status_json(status_path)
    status["status"] = "completed"
    status["current_stage"] = "alignment"
    status.setdefault("alignment", {})
    status["alignment"]["finished_at"] = utc_now_iso()
    status["alignment"]["result"] = "success"
    status["alignment"]["algorithm_version"] = ALGORITHM_VERSION
    status["alignment"]["word_assignment_ratio"] = metadata["quality_metrics"]["word_assignment_ratio"]
    status["alignment"]["utterance_assignment_ratio"] = metadata["quality_metrics"]["utterance_assignment_ratio"]
    status["alignment"]["warnings"] = metadata["warnings"]
    save_status_json(status_path, status)


def mark_alignment_failed(status_path: Path, error_message: str) -> None:
    status = load_status_json(status_path)
    status["status"] = "alignment"
    status["current_stage"] = "alignment"
    status.setdefault("alignment", {})
    status["alignment"]["finished_at"] = utc_now_iso()
    status["alignment"]["result"] = "failed"
    status["alignment"]["error"] = error_message
    save_status_json(status_path, status)


# =========================
# OUTPUT WRITERS
# =========================

def write_aligned_words(path: Path, episode_id: str, aligned_words) -> None:
    payload = {
        "episode_id": episode_id,
        "algorithm_version": ALGORITHM_VERSION,
        "words": [w.to_dict() for w in aligned_words],
    }
    write_json(path, payload)


def write_aligned_utterances(path: Path, episode_id: str, utterances) -> None:
    payload = {
        "episode_id": episode_id,
        "algorithm_version": ALGORITHM_VERSION,
        "utterances": [u.to_dict() for u in utterances],
    }
    write_json(path, payload)


def write_alignment_preview(path: Path, episode_id: str, utterances, metadata: dict) -> None:
    payload = {
        "episode_id": episode_id,
        "algorithm_version": ALGORITHM_VERSION,
        "preview_utterances": [u.to_dict() for u in utterances[:50]],
        "summary": {
            "aligned_utterances": metadata["output_summary"]["aligned_utterances"],
            "unknown_utterances": metadata["output_summary"]["unknown_utterances"],
            "warnings": metadata["warnings"],
        },
    }
    write_json(path, payload)


def write_alignment_result(
    path: Path,
    status_before: str,
    status_after: str,
    warnings: list[str],
    files_generated: list[str],
) -> None:
    payload = {
        "result": "success",
        "status_before": status_before,
        "status_after": status_after,
        "message": "Alignment completed successfully.",
        "algorithm_version": ALGORITHM_VERSION,
        "warnings": warnings,
        "files_generated": files_generated,
        "finished_at": utc_now_iso(),
    }
    write_json(path, payload)


def write_report_json(
    path: Path,
    episode_id: str,
    status_before: str,
    status_after: str,
    metadata: dict,
) -> None:
    payload = {
        "workflow": "04_alignment",
        "episode_id": episode_id,
        "status_before": status_before,
        "status_after": status_after,
        "result": "success",
        "algorithm_version": ALGORITHM_VERSION,
        "metrics": metadata["quality_metrics"],
        "warnings": metadata["warnings"],
        "finished_at": utc_now_iso(),
    }
    write_json(path, payload)


# =========================
# MAIN RUNNER
# =========================

def run(repo_root: Path) -> None:
    inputs_path = repo_root / "inputs" / "inputs.json"

    episode = select_alignment_episode(inputs_path)
    episode_id = episode["id"]
    status_before = episode["status"]

    work_dir = repo_root / "work" / episode_id
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

    append_jsonl(
        events_path,
        {
            "ts": utc_now_iso(),
            "event": "alignment_started",
            "episode_id": episode_id,
            "workflow": "04_alignment",
        },
    )

    mark_alignment_started(status_path, episode_id)

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

        _segment_assignments = assign_segments(
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

        write_aligned_words(
            alignment_dir / "aligned_words.json",
            episode_id,
            aligned_words,
        )
        write_aligned_utterances(
            alignment_dir / "aligned_utterances.json",
            episode_id,
            utterances,
        )
        export_srt(
            alignment_dir / "subtitles_speakers.srt",
            utterances,
        )
        write_json(
            alignment_dir / "alignment_metadata.json",
            metadata_dict,
        )
        write_alignment_preview(
            alignment_dir / "alignment_preview.json",
            episode_id,
            utterances,
            metadata_dict,
        )
        write_alignment_result(
            alignment_dir / "alignment_result.json",
            status_before=status_before,
            status_after="completed",
            warnings=metadata_dict["warnings"],
            files_generated=[
                "aligned_words.json",
                "aligned_utterances.json",
                "subtitles_speakers.srt",
                "alignment_metadata.json",
                "alignment_preview.json",
                "alignment_result.json",
            ],
        )

        mark_alignment_completed(status_path, metadata_dict)
        update_episode_status_in_inputs(inputs_path, episode_id, "completed")

        write_report_json(
            report_path,
            episode_id=episode_id,
            status_before=status_before,
            status_after="completed",
            metadata=metadata_dict,
        )

        append_jsonl(
            events_path,
            {
                "ts": utc_now_iso(),
                "event": "alignment_completed",
                "episode_id": episode_id,
                "workflow": "04_alignment",
                "word_assignment_ratio": metadata_dict["quality_metrics"]["word_assignment_ratio"],
                "utterance_assignment_ratio": metadata_dict["quality_metrics"]["utterance_assignment_ratio"],
            },
        )

    except Exception as exc:
        error_payload = {
            "workflow": "04_alignment",
            "episode_id": episode_id,
            "result": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at": utc_now_iso(),
        }
        write_json(error_path, error_payload)
        mark_alignment_failed(status_path, str(exc))

        append_jsonl(
            events_path,
            {
                "ts": utc_now_iso(),
                "event": "alignment_failed",
                "episode_id": episode_id,
                "workflow": "04_alignment",
                "error": str(exc),
            },
        )

        raise
