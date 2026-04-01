
# src/voicecaster/diarization/run.py

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import (
    LOW_CONFIDENCE_THRESHOLD,
    MERGE_GAP_SECONDS,
    MIN_SEGMENT_SECONDS,
    MIN_TRANSCRIPT_ASSIGNMENT_RATIO,
    USE_GPU_IF_AVAILABLE,
)
from .engine_pyannote import run_pyannote_diarization
from .normalize_segments import normalize_speaker_segments
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

INPUTS_PATH = Path("inputs/inputs.json")
WORK_DIR = Path("work")


# -------------------------
# Helpers
# -------------------------

def load_inputs() -> list[dict]:
    return json.loads(INPUTS_PATH.read_text(encoding="utf-8"))


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
    save_inputs(data)


def increment_retries(episode_id: str) -> None:
    data = load_inputs()
    for ep in data:
        if ep.get("id") == episode_id:
            ep["retries"] = int(ep.get("retries", 0)) + 1
    save_inputs(data)


def mark_ruined(episode_id: str) -> None:
    data = load_inputs()
    for ep in data:
        if ep.get("id") == episode_id:
            ep["status"] = "ruined"
    save_inputs(data)


def load_transcript_preview(work_episode_dir: Path) -> dict[str, Any]:
    path = work_episode_dir / "02_transcription" / "transcript_preview.json"
    if not path.exists():
        raise RuntimeError("Missing transcript_preview.json")
    return json.loads(path.read_text(encoding="utf-8"))


def fake_download_audio(url: str, target: Path) -> Path:
    """
    ⚠️ Placeholder.
    Debes conectar esto con tu downloader real de intake/transcription.
    """
    raise NotImplementedError("Integrate with your existing audio downloader.")


# -------------------------
# QA mínimo
# -------------------------

def run_basic_qa(
    speaker_segments,
    utterances,
    assignment_stats,
) -> None:
    if not speaker_segments:
        raise RuntimeError("No speaker segments produced.")

    if not utterances:
        raise RuntimeError("No transcript utterances.")

    if assignment_stats["assignment_ratio"] < MIN_TRANSCRIPT_ASSIGNMENT_RATIO:
        raise RuntimeError(
            f"Low assignment ratio: {assignment_stats['assignment_ratio']}"
        )


# -------------------------
# MAIN
# -------------------------

def main() -> int:
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
        # -------------------------
        # 1. AUDIO (TEMP)
        # -------------------------
        temp_audio = work_episode_dir / "99_temp" / "audio.wav"
        temp_audio.parent.mkdir(parents=True, exist_ok=True)

        audio_path = fake_download_audio(url, temp_audio)

        # -------------------------
        # 2. DIARIZATION ENGINE
        # -------------------------
        raw_segments, engine_metadata = run_pyannote_diarization(
            audio_path,
            hf_token=hf_token,
            use_gpu_if_available=USE_GPU_IF_AVAILABLE,
        )

        # -------------------------
        # 3. NORMALIZATION
        # -------------------------
        speaker_segments, label_map, norm_warnings = normalize_speaker_segments(
            raw_segments,
            min_segment_seconds=MIN_SEGMENT_SECONDS,
            merge_gap_seconds=MERGE_GAP_SECONDS,
        )

        # -------------------------
        # 4. TRANSCRIPT
        # -------------------------
        transcript_preview = load_transcript_preview(work_episode_dir)

        utterances, assignment_stats = assign_speakers_to_transcript(
            transcript_preview,
            speaker_segments,
            low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
        )

        # -------------------------
        # 5. QA
        # -------------------------
        run_basic_qa(speaker_segments, utterances, assignment_stats)

        # -------------------------
        # 6. OUTPUTS
        # -------------------------
        write_diarization_raw_json(diarization_dir, raw_segments, engine_metadata)
        write_speaker_segments_json(diarization_dir, speaker_segments)
        write_transcript_with_speakers_json(diarization_dir, utterances)
        write_subtitles_diarized_srt(diarization_dir, utterances)
        write_per_speaker_outputs(diarization_dir, utterances)
        write_speaker_metrics_json(diarization_dir, speaker_segments, utterances)

        write_diarization_metadata_json(
            diarization_dir,
            {
                "engine": engine_metadata,
                "normalization_warnings": norm_warnings,
                "assignment_stats": assignment_stats,
            },
        )

        write_diarization_result_json(
            diarization_dir,
            {
                "result": "success",
                "status_before": "diarization",
                "status_after": "alignment",
            },
        )

        # -------------------------
        # 7. STATE
        # -------------------------
        update_episode_status(episode_id, "alignment")

        print("[diarization] SUCCESS")
        return 0

    except Exception as exc:
        print(f"[diarization] ERROR: {repr(exc)}")

        increment_retries(episode_id)

        if int(episode.get("retries", 0)) > 10:
            mark_ruined(episode_id)

        return 1

    finally:
        # cleanup audio
        try:
            if temp_audio.exists():
                temp_audio.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
