# =========================================
# FILE: src/voicecaster/alignment/schemas.py
# =========================================

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TranscriptWord:
    source_segment_id: int
    index_in_segment: int
    word: str
    start: float
    end: float
    probability: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class TranscriptSegment:
    segment_id: int
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class SpeakerSegment:
    segment_id: int
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class SpeakerCandidate:
    speaker: str
    overlap_seconds: float
    overlap_ratio: float
    distance_to_center: float


@dataclass(slots=True)
class AlignedWord:
    word_id: str
    source_segment_id: int
    index_in_segment: int
    word: str
    start: float
    end: float
    probability: float | None
    speaker: str
    speaker_confidence: float | None
    assignment_method: str
    flags: list[str] = field(default_factory=list)
    candidates: list[SpeakerCandidate] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class SegmentSpeakerAssignment:
    source_segment_id: int
    start: float
    end: float
    speaker: str
    speaker_confidence: float | None
    assignment_method: str
    flags: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class AlignedUtterance:
    utterance_id: str
    source_segment_ids: list[int]
    start: float
    end: float
    speaker: str
    speaker_confidence: float | None
    assignment_method: str
    text: str
    word_count: int
    words: list[AlignedWord] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class AlignmentMetrics:
    algorithm_version: str
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    quality_metrics: dict[str, Any]
    timing_metrics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AlignmentPaths:
    episode_root: Path
    stage_dir: Path

    transcript_preview_json: Path
    speaker_segments_json: Path
    speaker_metrics_json: Path | None = None
    diarization_metadata_json: Path | None = None

    aligned_words_json: Path | None = None
    aligned_utterances_json: Path | None = None
    subtitles_speakers_srt: Path | None = None
    alignment_metadata_json: Path | None = None
    alignment_result_json: Path | None = None
    alignment_preview_json: Path | None = None


def dataclass_to_dict(obj: Any) -> Any:
    """
    Recursive serialization helper for dataclasses and nested structures.
    """
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    return obj
