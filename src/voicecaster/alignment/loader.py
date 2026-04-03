from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import AlignmentPaths


class AlignmentInputError(RuntimeError):
    """Raised when required alignment inputs are missing or malformed."""


def build_alignment_paths(work_root: Path, episode_id: str) -> AlignmentPaths:
    """
    Build canonical filesystem paths for 04_alignment.

    Real repo contract:
    - transcription authority: transcript_segments.json
    - transcription preview: transcript_preview.json (not used for alignment)
    - diarization authority: speaker_segments.json
    """
    episode_root = work_root / episode_id
    stage_dir = episode_root / "04_alignment"

    return AlignmentPaths(
        episode_root=episode_root,
        stage_dir=stage_dir,
        transcript_segments_json=episode_root / "02_transcription" / "transcript_segments.json",
        transcript_srt=episode_root / "02_transcription" / "full_transcript.srt",
        speaker_segments_json=episode_root / "03_diarization" / "speaker_segments.json",
        speaker_metrics_json=episode_root / "03_diarization" / "speaker_metrics.json",
        diarization_metadata_json=episode_root / "03_diarization" / "diarization_metadata.json",
        aligned_words_json=stage_dir / "aligned_words.json",
        aligned_utterances_json=stage_dir / "aligned_utterances.json",
        subtitles_speakers_srt=stage_dir / "subtitles_speakers.srt",
        alignment_metadata_json=stage_dir / "alignment_metadata.json",
        alignment_result_json=stage_dir / "alignment_result.json",
        alignment_preview_json=stage_dir / "alignment_preview.json",
    )


def ensure_stage_dir(paths: AlignmentPaths) -> None:
    paths.stage_dir.mkdir(parents=True, exist_ok=True)


def load_json_file(path: Path) -> Any:
    if not path.exists():
        raise AlignmentInputError(f"Missing required file: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise AlignmentInputError(f"Invalid JSON in file: {path}") from exc


def load_transcript_segments(path: Path) -> dict[str, Any]:
    """
    Load full transcription authority.

    Supported shapes:
    1. {"segments": [...]}
    2. [...]
    """
    data = load_json_file(path)

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        return {"segments": data}

    raise AlignmentInputError(
        f"transcript_segments.json must be a JSON object or a JSON list: {path}"
    )


def load_speaker_segments(path: Path) -> dict[str, Any]:
    """
    Load diarization authority.

    Supported shapes:
    1. {"segments": [...]}
    2. [...]
    """
    data = load_json_file(path)

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        return {"segments": data}

    raise AlignmentInputError(
        f"speaker_segments.json must be a JSON object or a JSON list: {path}"
    )


def validate_required_inputs(transcript_raw: dict[str, Any], speakers_raw: dict[str, Any]) -> None:
    if "segments" not in transcript_raw:
        raise AlignmentInputError("transcript input missing required key: 'segments'")

    if not isinstance(transcript_raw["segments"], list):
        raise AlignmentInputError("transcript input 'segments' must be a list")

    if "segments" not in speakers_raw:
        raise AlignmentInputError("speaker input missing required key: 'segments'")

    if not isinstance(speakers_raw["segments"], list):
        raise AlignmentInputError("speaker input 'segments' must be a list")

    if len(transcript_raw["segments"]) == 0:
        raise AlignmentInputError("transcript input contains no transcript segments")

    if len(speakers_raw["segments"]) == 0:
        raise AlignmentInputError("speaker input contains no speaker segments")
