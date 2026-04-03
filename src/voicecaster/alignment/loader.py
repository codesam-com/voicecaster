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

    Note:
    - transcript_preview_json currently points to the existing repo artifact.
    - This file may be only a preview summary, not the full transcript authority.
    """
    episode_root = work_root / episode_id
    stage_dir = episode_root / "04_alignment"

    return AlignmentPaths(
        episode_root=episode_root,
        stage_dir=stage_dir,
        transcript_preview_json=episode_root / "02_transcription" / "transcript_preview.json",
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
    """
    Ensure 04_alignment output directory exists.
    """
    paths.stage_dir.mkdir(parents=True, exist_ok=True)


def load_json_file(path: Path) -> Any:
    """
    Load JSON from disk.

    Accepts either:
    - dict root
    - list root
    """
    if not path.exists():
        raise AlignmentInputError(f"Missing required file: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise AlignmentInputError(f"Invalid JSON in file: {path}") from exc


def load_transcript_preview(path: Path) -> dict[str, Any]:
    """
    Load transcription-side artifact.

    Supported shapes:
    1. Full form:
       {"segments": [...]}

    2. Preview-only form currently observed in repo:
       {
         "episode_id": "...",
         "language": "...",
         "num_segments": ...,
         "first_segments": [...],
         "last_segments": [...]
       }

    Important:
    The preview-only form is NOT sufficient for alignment because it does not
    contain the full timeline. We keep it as-is so the normalizer can emit a
    precise and truthful error.
    """
    data = load_json_file(path)

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        # Accept legacy/direct list form as a full transcript representation.
        return {"segments": data}

    raise AlignmentInputError(
        f"transcript input must be a JSON object or JSON list: {path}"
    )


def load_speaker_segments(path: Path) -> dict[str, Any]:
    """
    Load diarization-side artifact.

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
    """
    Validate minimum structural contract for alignment.

    We distinguish two transcript cases:
    - full transcript: OK
    - preview-only transcript: fail with explicit message
    """
    if "segments" not in transcript_raw:
        if "first_segments" in transcript_raw or "last_segments" in transcript_raw:
            raise AlignmentInputError(
                "transcript_preview.json is a preview-only artifact "
                "(uses first_segments/last_segments) and does not contain the full "
                "transcript timeline required by alignment"
            )
        raise AlignmentInputError("transcript input missing required key: 'segments'")

    if not isinstance(transcript_raw["segments"], list):
        raise AlignmentInputError("transcript input 'segments' must be a list")

    if "segments" not in speakers_raw:
        raise AlignmentInputError("speaker_segments.json missing required key: 'segments'")

    if not isinstance(speakers_raw["segments"], list):
        raise AlignmentInputError("speaker_segments.json 'segments' must be a list")

    if len(transcript_raw["segments"]) == 0:
        raise AlignmentInputError("transcript input contains no transcript segments")

    if len(speakers_raw["segments"]) == 0:
        raise AlignmentInputError("speaker_segments.json contains no speaker segments")
