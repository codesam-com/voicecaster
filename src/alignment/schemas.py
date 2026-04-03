from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# =========================
# INPUT MODELS
# =========================

@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: Optional[float] = None


@dataclass
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


# =========================
# INTERNAL / OUTPUT MODELS
# =========================

@dataclass
class AlignedWord:
    word_id: str
    source_segment_id: int
    index_in_segment: int
    start: float
    end: float
    duration: float
    word: str
    probability: Optional[float]
    speaker: str
    speaker_confidence: float
    assignment_method: str
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Utterance:
    utterance_id: str
    source_segment_ids: list[int]
    start: float
    end: float
    duration: float
    speaker: str
    speaker_confidence: float
    assignment_method: str
    assignment_basis: dict[str, Any]
    text: str
    word_count: int
    words: list[AlignedWord] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["words"] = [w.to_dict() for w in self.words]
        return data


@dataclass
class AlignmentMetadata:
    algorithm_version: str
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    quality_metrics: dict[str, Any]
    timing_metrics: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
