# voicecaster/diarization/models.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RawSpeakerSegment:
    """
    Segmento crudo devuelto por el motor de diarización, convertido
    a un formato neutral del proyecto.
    """
    start: float
    end: float
    speaker_raw: str
    confidence: float | None = None
    engine: str = "pyannote"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration"] = self.duration
        return payload


@dataclass(slots=True)
class SpeakerSegment:
    """
    Segmento ya normalizado y canónico del sistema.
    Este es el tipo que debe alimentar la autoridad de 03.
    """
    segment_id: str
    start: float
    end: float
    duration: float
    speaker: str
    speaker_confidence: float | None = None
    source: str = "normalized_diarization"
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TranscriptWord:
    """
    Unidad opcional a nivel palabra, heredada del ASR y enriquecida
    posteriormente con speaker cuando sea posible.
    """
    word: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    confidence: float | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TranscriptUtterance:
    """
    Bloque textual procedente de transcription, con speaker asignado
    de forma preliminar por 03.
    """
    utterance_id: str
    start: float
    end: float
    duration: float
    text: str
    speaker: str | None = None
    speaker_confidence: float | None = None
    assignment_source: str | None = None
    overlap_stats: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    words: list[TranscriptWord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["words"] = [w.to_dict() for w in self.words]
        return payload
