from __future__ import annotations

import json
from pathlib import Path

from .schemas import AlignmentPaths


def build_alignment_paths(episode_dir: Path) -> AlignmentPaths:
    transcription_dir = episode_dir / "02_transcription"
    diarization_dir = episode_dir / "03_diarization"
    alignment_dir = episode_dir / "04_alignment"

    return AlignmentPaths(
        episode_dir=episode_dir,
        transcription_dir=transcription_dir,
        diarization_dir=diarization_dir,
        alignment_dir=alignment_dir,
        transcript_preview_path=transcription_dir / "transcript_preview.json",
        speaker_segments_path=diarization_dir / "speaker_segments.json",
        aligned_words_path=alignment_dir / "aligned_words.json",
        aligned_utterances_path=alignment_dir / "aligned_utterances.json",
        subtitles_speakers_path=alignment_dir / "subtitles_speakers.srt",
        alignment_metadata_path=alignment_dir / "alignment_metadata.json",
        alignment_result_path=alignment_dir / "alignment_result.json",
        alignment_preview_path=alignment_dir / "alignment_preview.json",
        status_json_path=episode_dir / "status.json",
    )


def validate_required_files(paths: AlignmentPaths) -> None:
    required_files = [
        paths.transcript_preview_path,
        paths.speaker_segments_path,
        paths.status_json_path,
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required alignment input files: {', '.join(missing)}"
        )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_transcript_preview_json(path: Path) -> dict:
    return load_json(path)


def load_speaker_segments_json(path: Path) -> dict:
    return load_json(path)


def validate_raw_transcript_preview(payload: dict) -> None:
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("transcript_preview.json must contain a non-empty 'segments' list")

    for i, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"Transcript segment at index {i} is not an object")
        for key in ("start", "end", "text"):
            if key not in segment:
                raise ValueError(
                    f"Transcript segment at index {i} is missing required key '{key}'"
                )


def validate_raw_speaker_segments(payload: dict) -> None:
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("speaker_segments.json must contain a non-empty 'segments' list")

    for i, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"Speaker segment at index {i} is not an object")
        for key in ("start", "end", "speaker"):
            if key not in segment:
                raise ValueError(
                    f"Speaker segment at index {i} is missing required key '{key}'"
                )
