from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import (
    LOW_CONFIDENCE_THRESHOLD,
    MAX_RETRIES,
    MERGE_GAP_SECONDS,
    MIN_SEGMENT_SECONDS,
    MIN_TRANSCRIPT_ASSIGNMENT_RATIO,
    POSTPROCESS_DROP_MICROSEGMENTS_SECONDS,
    POSTPROCESS_MAX_ABA_WINDOW_SECONDS,
    POSTPROCESS_MAX_BRIDGE_SECONDS,
    POSTPROCESS_MERGE_GAP_SECONDS,
    POSTPROCESS_MIN_SPEAKER_RATIO,
    POSTPROCESS_MIN_SPEAKER_SECONDS,
    USE_GPU_IF_AVAILABLE,
)
from .debug_report import build_debug_report
from .engine_pyannote import run_pyannote_diarization
from .metrics import compute_speaker_metrics
from .normalize_segments import normalize_speaker_segments
from .postprocess_segments import postprocess_speaker_segments
from .qa import run_diarization_qa
from .reconcile_with_transcript import assign_speakers_to_transcript
from .write_outputs import (
    ensure_diarization_dir,
    write_diarization_metadata_json,
    write_diarization_raw_json,
    write_diarization_result_json,
    write_per_speaker_outputs,
    write_speaker_metrics_json,
    write_speaker_segments_json,
    write_subtitles_diarized_srt,
    write_transcript_with_speakers_json,
)

from voicecaster.transcription.run import ContentError, NetworkError, download_file, ffprobe_audio

INPUTS_PATH = Path("inputs/inputs.json")
WORK_DIR = Path("work")


def load_inputs() -> list[dict]:
    if not INPUTS_PATH.exists():
        raise RuntimeError(f"Inputs file not found: {INPUTS_PATH}")
    data = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("inputs/inputs.json must contain a JSON array.")
    return data


def save_inputs(data: list[dict]) -> None:
    INPUTS_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def select_episode() -> dict:
    data = load_inputs()
    for ep in data:
        if ep.get("status") == "diarization":
            return ep
    raise RuntimeError("No episode with status=diarization found.")


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


def load_transcript_segments(work_episode_dir: Path) -> dict[str, Any]:
    path = work_episode_dir / "02_transcription" / "transcript_segments.json"
    if not path.exists():
        raise RuntimeError(f"Missing transcript segments file: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Invalid transcript segments JSON list: {path}")

    return {"segments": data}


def load_transcript_preview_if_exists(work_episode_dir: Path) -> dict[str, Any] | None:
    path = work_episode_dir / "02_transcription" / "transcript_preview.json"
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return data


def ensure_required_paths(work_episode_dir: Path) -> None:
    required_paths = [
        work_episode_dir / "02_transcription" / "transcript_segments.json",
        work_episode_dir / "02_transcription" / "full_transcript.srt",
        work_episode_dir / "02_transcription" / "full_transcript.txt",
        work_episode_dir / "02_transcription" / "transcription_metadata.json",
    ]

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required transcription artifacts: {missing}")


def is_network_error(exc: BaseException) -> bool:
    return isinstance(exc, NetworkError)


def download_audio_with_existing_pipeline(url: str, target: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    target.parent.mkdir(parents=True, exist_ok=True)

    download_info = download_file(url, target)
    audio_probe = ffprobe_audio(target)

    if not audio_probe.get("ffprobe_ok", False):
        raise ContentError("Downloaded resource is not a valid audio file for diarization.")

    return target, download_info, audio_probe


def main() -> int:
    temp_audio: Path | None = None

    try:
        episode = select_episode()
    except Exception:
        print("[diarization] No work to do.")
        return 0

    episode_id = episode["id"]
    url = episode["url"]

    print(f"[diarization] Processing episode: {episode_id}")

    work_episode_dir = WORK_DIR / episode_id
    diarization_dir = ensure_diarization_dir(work_episode_dir)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set")

    try:
        ensure_required_paths(work_episode_dir)

        temp_audio = work_episode_dir / "99_temp" / "audio_for_diarization"
        temp_audio.parent.mkdir(parents=True, exist_ok=True)

        audio_path, download_info, audio_probe = download_audio_with_existing_pipeline(
            url,
            temp_audio,
        )

        raw_segments, engine_metadata = run_pyannote_diarization(
            audio_path,
            hf_token=hf_token,
            use_gpu_if_available=USE_GPU_IF_AVAILABLE,
        )

        normalized_segments, label_map, norm_warnings = normalize_speaker_segments(
            raw_segments,
            min_segment_seconds=MIN_SEGMENT_SECONDS,
            merge_gap_seconds=MERGE_GAP_SECONDS,
        )

        processed_segments, postprocess_report = postprocess_speaker_segments(
            normalized_segments,
            min_speaker_ratio=POSTPROCESS_MIN_SPEAKER_RATIO,
            min_speaker_seconds=POSTPROCESS_MIN_SPEAKER_SECONDS,
            merge_gap_seconds=POSTPROCESS_MERGE_GAP_SECONDS,
            max_bridge_seconds=POSTPROCESS_MAX_BRIDGE_SECONDS,
            max_aba_window_seconds=POSTPROCESS_MAX_ABA_WINDOW_SECONDS,
            drop_microsegments_seconds=POSTPROCESS_DROP_MICROSEGMENTS_SECONDS,
        )

        transcript_data = load_transcript_segments(work_episode_dir)
        transcript_preview = load_transcript_preview_if_exists(work_episode_dir)

        utterances, assignment_stats = assign_speakers_to_transcript(
            transcript_data,
            processed_segments,
            low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
        )

        metrics_payload = compute_speaker_metrics(
            processed_segments,
            utterances,
        )

        qa_result = run_diarization_qa(
            processed_segments,
            utterances,
            min_assignment_ratio=MIN_TRANSCRIPT_ASSIGNMENT_RATIO,
        )

        if not qa_result.passed:
            raise RuntimeError(
                f"Diarization QA failed: {[issue.to_dict() for issue in qa_result.issues]}"
            )

        write_diarization_raw_json(diarization_dir, raw_segments, engine_metadata)
        write_speaker_segments_json(diarization_dir, processed_segments)
        write_transcript_with_speakers_json(diarization_dir, utterances)
        write_subtitles_diarized_srt(diarization_dir, utterances)
        write_per_speaker_outputs(diarization_dir, utterances)
        write_speaker_metrics_json(diarization_dir, metrics_payload)

        debug_payload = build_debug_report(
            engine_metadata={
                **engine_metadata,
                "download": download_info,
                "audio_probe": audio_probe,
            },
            label_map=label_map,
            normalization_warnings=norm_warnings,
            assignment_stats=assignment_stats,
            qa_result=qa_result,
        )
        debug_payload["postprocess"] = postprocess_report

        if transcript_preview is not None:
            debug_payload["transcript_preview"] = transcript_preview

        write_diarization_metadata_json(diarization_dir, debug_payload)

        write_diarization_result_json(
            diarization_dir,
            {
                "result": "success",
                "status_before": "diarization",
                "status_after": "alignment",
                "retries_before": int(episode.get("retries", 0)),
                "retries_after": 0,
                "num_speakers_detected": metrics_payload.get("num_speakers_detected", 0),
            },
        )

        update_episode_status(episode_id, "alignment")

        print("[diarization] SUCCESS")
        return 0

    except Exception as exc:
        print(f"[diarization] ERROR: {repr(exc)}")

        retries_after = increment_retries(episode_id)
        if retries_after > MAX_RETRIES:
            mark_ruined(episode_id)

        try:
            write_diarization_result_json(
                diarization_dir,
                {
                    "result": "error",
                    "status_before": "diarization",
                    "status_after": "ruined" if retries_after > MAX_RETRIES else "diarization",
                    "retries_before": int(episode.get("retries", 0)),
                    "retries_after": retries_after,
                    "error": repr(exc),
                    "retry_consumed": not is_network_error(exc),
                },
            )
        except Exception as write_exc:
            print(f"[diarization] Failed to write diarization_result.json: {repr(write_exc)}")

        return 1

    finally:
        try:
            if temp_audio and temp_audio.exists():
                temp_audio.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
