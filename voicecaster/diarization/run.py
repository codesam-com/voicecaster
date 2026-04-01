from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .audio import AudioPreparationError, prepare_temp_audio_path
from .config import (
    ACTION_NAME,
    ACTION_VERSION,
    INPUTS_JSON_PATH,
    LOW_CONFIDENCE_THRESHOLD,
    MERGE_GAP_SECONDS,
    MIN_SEGMENT_SECONDS,
    OUTPUT_DIR_NAME,
    TEMP_DIR_NAME,
    USE_GPU_IF_AVAILABLE,
    WORK_DIR,
    get_hf_token,
)
from .engine_pyannote import (
    PyannoteDiarizationError,
    PyannotePipelineLoadError,
    run_pyannote_diarization,
)
from .normalize_segments import normalize_speaker_segments
from .reconcile_with_transcript import assign_speakers_to_transcript
from .write_outputs import (
    write_diarization_metadata_json,
    write_diarization_raw_json,
    write_diarization_result_json,
    write_per_speaker_outputs,
    write_speaker_metrics_json,
    write_speaker_segments_json,
    write_subtitles_diarized_srt,
    write_transcript_with_speakers_json,
)


class RecoverableError(RuntimeError):
    pass


class FatalError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FatalError(f"Missing required file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FatalError(f"Invalid JSON in file {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def select_episode_for_diarization(inputs_payload: list[dict[str, Any]]) -> dict[str, Any]:
    for item in inputs_payload:
        if not isinstance(item, dict):
            continue
        if item.get("status") == "diarization":
            return item
    raise FatalError("No episode with status='diarization' found in inputs/inputs.json")


def update_episode_in_inputs(
    inputs_payload: list[dict[str, Any]],
    episode_id: str,
    *,
    new_status: str | None = None,
    retries: int | None = None,
) -> list[dict[str, Any]]:
    updated = False

    for item in inputs_payload:
        if not isinstance(item, dict):
            continue
        if item.get("id") != episode_id:
            continue

        if new_status is not None:
            item["status"] = new_status
        if retries is not None:
            item["retries"] = retries

        updated = True
        break

    if not updated:
        raise FatalError(f"Episode id not found in inputs payload: {episode_id}")

    return inputs_payload


def get_episode_paths(episode_id: str) -> dict[str, Path]:
    episode_dir = WORK_DIR / episode_id
    transcription_dir = episode_dir / "02_transcription"
    diarization_dir = episode_dir / OUTPUT_DIR_NAME
    temp_dir = episode_dir / TEMP_DIR_NAME
    status_path = episode_dir / "status.json"

    return {
        "episode_dir": episode_dir,
        "transcription_dir": transcription_dir,
        "diarization_dir": diarization_dir,
        "temp_dir": temp_dir,
        "status_path": status_path,
        "transcript_preview_path": transcription_dir / "transcript_preview.json",
        "full_transcript_path": transcription_dir / "full_transcript.txt",
        "subtitles_path": transcription_dir / "subtitles.srt",
        "transcription_metadata_path": transcription_dir / "transcription_metadata.json",
    }


def precheck_required_files(paths: dict[str, Path]) -> None:
    required = [
        paths["status_path"],
        paths["transcript_preview_path"],
        paths["full_transcript_path"],
        paths["subtitles_path"],
        paths["transcription_metadata_path"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FatalError(f"Missing required transcription artifacts: {missing}")


def load_status_json(status_path: Path) -> dict[str, Any]:
    payload = load_json(status_path)
    if not isinstance(payload, dict):
        raise FatalError(f"status.json must be a JSON object: {status_path}")
    return payload


def save_status_json(status_path: Path, payload: dict[str, Any]) -> None:
    write_json(status_path, payload)


def ensure_episode_ready_for_diarization(
    episode: dict[str, Any],
    status_payload: dict[str, Any],
) -> None:
    if episode.get("status") != "diarization":
        raise FatalError(
            f"Episode in inputs is not in diarization status: {episode.get('status')}"
        )
    if status_payload.get("status") != "diarization":
        raise FatalError(
            f"status.json is not in diarization status: {status_payload.get('status')}"
        )


def mark_status_started(status_payload: dict[str, Any]) -> dict[str, Any]:
    status_payload["current_action"] = ACTION_NAME
    status_payload["last_started_at"] = utc_now_iso()
    return status_payload


def mark_status_success(status_payload: dict[str, Any], episode_id: str) -> dict[str, Any]:
    status_payload["episode_id"] = episode_id
    status_payload["status"] = "alignment"
    status_payload["current_action"] = ACTION_NAME
    status_payload["last_result"] = "diarization_completed"
    status_payload["last_finished_at"] = utc_now_iso()
    status_payload["retries"] = 0

    artifacts = status_payload.setdefault("artifacts", {})
    artifacts["diarization"] = {
        "speaker_segments_json": f"work/{episode_id}/03_diarization/speaker_segments.json",
        "transcript_with_speakers_json": f"work/{episode_id}/03_diarization/transcript_with_speakers.json",
        "subtitles_diarized_srt": f"work/{episode_id}/03_diarization/subtitles_diarized.srt",
        "speaker_metrics_json": f"work/{episode_id}/03_diarization/speaker_metrics.json",
    }
    return status_payload


def mark_status_failure(
    status_payload: dict[str, Any],
    error_message: str,
    retries: int,
    ruined: bool,
) -> dict[str, Any]:
    status_payload["current_action"] = ACTION_NAME
    status_payload["last_result"] = "diarization_failed"
    status_payload["last_error"] = error_message
    status_payload["last_finished_at"] = utc_now_iso()
    status_payload["retries"] = retries
    status_payload["status"] = "ruined" if ruined else "diarization"
    return status_payload


def compute_speaker_metrics(utterances: list[Any]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = {}
    for utt in utterances:
        speaker = utt.speaker or "unknown_speaker"
        grouped.setdefault(speaker, []).append(utt)

    total_speech_seconds = sum(utt.duration for utt in utterances)
    speakers_payload: list[dict[str, Any]] = []

    for speaker, items in sorted(grouped.items(), key=lambda x: x[0]):
        speech_seconds = sum(utt.duration for utt in items)
        num_turns = len(items)
        avg_turn_seconds = speech_seconds / num_turns if num_turns else 0.0
        longest_turn_seconds = max((utt.duration for utt in items), default=0.0)

        speakers_payload.append(
            {
                "speaker": speaker,
                "speech_seconds": round(speech_seconds, 3),
                "speech_ratio": round(
                    speech_seconds / total_speech_seconds, 4
                ) if total_speech_seconds > 0 else 0.0,
                "num_turns": num_turns,
                "avg_turn_seconds": round(avg_turn_seconds, 3),
                "longest_turn_seconds": round(longest_turn_seconds, 3),
            }
        )

    return {
        "num_speakers_detected": len(grouped),
        "total_speech_seconds": round(total_speech_seconds, 3),
        "speakers": speakers_payload,
    }


def run_basic_qa(
    normalized_segments: list[Any],
    reconciliation_stats: dict[str, Any],
) -> dict[str, Any]:
    timeline_consistent = all(seg.end > seg.start for seg in normalized_segments)
    labels_ok = all(seg.speaker.startswith("speaker_") for seg in normalized_segments)
    has_segments = len(normalized_segments) > 0
    transcript_assignment_ratio = float(
        reconciliation_stats.get("transcript_assignment_ratio", 0.0)
    )

    return {
        "timeline_consistent": timeline_consistent,
        "labels_ok": labels_ok,
        "has_segments": has_segments,
        "transcript_assignment_ratio": transcript_assignment_ratio,
    }


def ensure_qa_passes(qa_results: dict[str, Any]) -> None:
    if not qa_results["has_segments"]:
        raise RecoverableError("QA failed: no normalized speaker segments")
    if not qa_results["timeline_consistent"]:
        raise RecoverableError("QA failed: inconsistent timeline")
    if not qa_results["labels_ok"]:
        raise RecoverableError("QA failed: invalid canonical labels")


def cleanup_temp_dir(temp_dir: Path) -> None:
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    inputs_payload = load_json(INPUTS_JSON_PATH)
    if not isinstance(inputs_payload, list):
        raise FatalError("inputs/inputs.json must contain a JSON list")

    episode = select_episode_for_diarization(inputs_payload)
    episode_id = str(episode.get("id", "")).strip()
    if not episode_id:
        raise FatalError("Selected episode has no valid 'id'")

    retries_before = int(episode.get("retries", 0) or 0)
    paths = get_episode_paths(episode_id)

    precheck_required_files(paths)

    status_payload = load_status_json(paths["status_path"])
    ensure_episode_ready_for_diarization(episode, status_payload)
    save_status_json(paths["status_path"], mark_status_started(status_payload))

    diarization_dir = paths["diarization_dir"]
    diarization_dir.mkdir(parents=True, exist_ok=True)

    try:
        audio_path = prepare_temp_audio_path(paths["temp_dir"], episode)
        transcript_preview = load_json(paths["transcript_preview_path"])
        hf_token = get_hf_token()

        raw_segments, engine_metadata = run_pyannote_diarization(
            audio_path=audio_path,
            hf_token=hf_token,
            use_gpu_if_available=USE_GPU_IF_AVAILABLE,
        )

        normalized_segments, label_mapping, normalization_warnings = normalize_speaker_segments(
            raw_segments=raw_segments,
            min_segment_seconds=MIN_SEGMENT_SECONDS,
            merge_gap_seconds=MERGE_GAP_SECONDS,
        )

        if not normalized_segments:
            raise RecoverableError("No normalized speaker segments produced")

        utterances, reconciliation_stats = assign_speakers_to_transcript(
            transcript_preview=transcript_preview,
            speaker_segments=normalized_segments,
            low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
        )

        speaker_metrics = compute_speaker_metrics(utterances)
        qa_results = run_basic_qa(normalized_segments, reconciliation_stats)
        ensure_qa_passes(qa_results)

        write_diarization_raw_json(
            output_dir=diarization_dir,
            raw_segments=raw_segments,
            engine_metadata=engine_metadata,
        )
        write_speaker_segments_json(
            output_dir=diarization_dir,
            speaker_segments=normalized_segments,
            label_mapping=label_mapping,
            normalization_warnings=normalization_warnings,
        )
        write_transcript_with_speakers_json(
            output_dir=diarization_dir,
            utterances=utterances,
            reconciliation_stats=reconciliation_stats,
        )
        write_subtitles_diarized_srt(
            output_dir=diarization_dir,
            utterances=utterances,
        )
        write_per_speaker_outputs(
            output_dir=diarization_dir,
            utterances=utterances,
        )
        write_speaker_metrics_json(
            output_dir=diarization_dir,
            utterances=utterances,
        )

        diarization_metadata = {
            "action": ACTION_NAME,
            "version": ACTION_VERSION,
            "engine_metadata": engine_metadata,
            "normalization": {
                "min_segment_seconds": MIN_SEGMENT_SECONDS,
                "merge_gap_seconds": MERGE_GAP_SECONDS,
                "warnings": normalization_warnings,
                "label_mapping": label_mapping,
            },
            "reconciliation_stats": reconciliation_stats,
            "qa": qa_results,
            "finished_at": utc_now_iso(),
        }
        write_diarization_metadata_json(
            output_dir=diarization_dir,
            metadata=diarization_metadata,
        )

        result_payload = {
            "result": "success",
            "status_before": "diarization",
            "status_after": "alignment",
            "retries_before": retries_before,
            "retries_after": 0,
            "num_speakers_detected": speaker_metrics["num_speakers_detected"],
            "warnings": normalization_warnings,
            "finished_at": utc_now_iso(),
        }
        write_diarization_result_json(
            output_dir=diarization_dir,
            result_payload=result_payload,
        )

        status_payload = load_status_json(paths["status_path"])
        status_payload = mark_status_success(status_payload, episode_id)
        save_status_json(paths["status_path"], status_payload)

        inputs_payload = load_json(INPUTS_JSON_PATH)
        inputs_payload = update_episode_in_inputs(
            inputs_payload,
            episode_id,
            new_status="alignment",
            retries=0,
        )
        write_json(INPUTS_JSON_PATH, inputs_payload)

        return 0

    except FatalError as exc:
        retries_after = retries_before + 1

        status_payload = load_status_json(paths["status_path"])
        status_payload = mark_status_failure(
            status_payload,
            error_message=str(exc),
            retries=retries_after,
            ruined=True,
        )
        save_status_json(paths["status_path"], status_payload)

        inputs_payload = load_json(INPUTS_JSON_PATH)
        inputs_payload = update_episode_in_inputs(
            inputs_payload,
            episode_id,
            new_status="ruined",
            retries=retries_after,
        )
        write_json(INPUTS_JSON_PATH, inputs_payload)

        write_diarization_result_json(
            output_dir=diarization_dir,
            result_payload={
                "result": "fatal_error",
                "status_before": "diarization",
                "status_after": "ruined",
                "retries_before": retries_before,
                "retries_after": retries_after,
                "error": str(exc),
                "finished_at": utc_now_iso(),
            },
        )
        return 2

    except (
        RecoverableError,
        AudioPreparationError,
        PyannotePipelineLoadError,
        PyannoteDiarizationError,
    ) as exc:
        retries_after = retries_before + 1

        status_payload = load_status_json(paths["status_path"])
        status_payload = mark_status_failure(
            status_payload,
            error_message=str(exc),
            retries=retries_after,
            ruined=False,
        )
        save_status_json(paths["status_path"], status_payload)

        inputs_payload = load_json(INPUTS_JSON_PATH)
        inputs_payload = update_episode_in_inputs(
            inputs_payload,
            episode_id,
            new_status="diarization",
            retries=retries_after,
        )
        write_json(INPUTS_JSON_PATH, inputs_payload)

        write_diarization_result_json(
            output_dir=diarization_dir,
            result_payload={
                "result": "recoverable_error",
                "status_before": "diarization",
                "status_after": "diarization",
                "retries_before": retries_before,
                "retries_after": retries_after,
                "error": str(exc),
                "finished_at": utc_now_iso(),
            },
        )
        return 1

    finally:
        cleanup_temp_dir(paths["temp_dir"])


if __name__ == "__main__":
    raise SystemExit(main())
