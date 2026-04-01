# src/voicecaster/diarization/models.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RawSpeakerSegment:
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
        data = asdict(self)
        data["duration"] = self.duration
        return data


@dataclass
class SpeakerSegment:
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


@dataclass
class TranscriptUtterance:
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
    words: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
