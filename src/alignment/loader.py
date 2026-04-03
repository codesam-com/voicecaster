import json
from pathlib import Path

from .schemas import TranscriptSegment, Word, SpeakerSegment


def load_transcript_preview(path: Path) -> list[TranscriptSegment]:
    data = json.loads(path.read_text())

    segments = []
    for seg in data.get("segments", []):
        words = [
            Word(
                start=w["start"],
                end=w["end"],
                word=w["word"],
                probability=w.get("probability"),
            )
            for w in seg.get("words", [])
        ]

        segments.append(
            TranscriptSegment(
                id=seg["id"],
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                words=words,
            )
        )

    if not segments:
        raise ValueError("Transcript vacío")

    return segments


def load_speaker_segments(path: Path) -> list[SpeakerSegment]:
    data = json.loads(path.read_text())

    segments = [
        SpeakerSegment(
            start=s["start"],
            end=s["end"],
            speaker=s["speaker"],
        )
        for s in data.get("segments", [])
    ]

    if not segments:
        raise ValueError("Diarization vacía")

    return segments
