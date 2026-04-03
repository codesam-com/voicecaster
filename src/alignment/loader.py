from __future__ import annotations

import json
from pathlib import Path

from .schemas import SpeakerSegment, TranscriptSegment, Word


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_transcript_preview(path: Path) -> list[TranscriptSegment]:
    data = load_json(path)

    raw_segments = data.get("segments", [])
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("Transcript vacío o sin campo 'segments' válido")

    segments: list[TranscriptSegment] = []

    for seg in raw_segments:
        raw_words = seg.get("words", []) or []
        words = [
            Word(
                start=float(w["start"]),
                end=float(w["end"]),
                word=str(w["word"]),
                probability=w.get("probability"),
            )
            for w in raw_words
            if "start" in w and "end" in w and "word" in w
        ]

        segments.append(
            TranscriptSegment(
                id=int(seg["id"]),
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=str(seg.get("text", "")),
                words=words,
            )
        )

    if not segments:
        raise ValueError("Transcript vacío tras parseo")

    return segments


def load_speaker_segments(path: Path) -> list[SpeakerSegment]:
    data = load_json(path)

    raw_segments = data.get("segments", [])
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("Diarization vacía o sin campo 'segments' válido")

    segments = [
        SpeakerSegment(
            start=float(s["start"]),
            end=float(s["end"]),
            speaker=str(s["speaker"]),
        )
        for s in raw_segments
        if "start" in s and "end" in s and "speaker" in s
    ]

    if not segments:
        raise ValueError("Diarization vacía tras parseo")

    return segments
