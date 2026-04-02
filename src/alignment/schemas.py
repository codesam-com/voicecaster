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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "text": self.text,
            "words": [word.to_dict() for word in self.words],
        }


@dataclass(slots=True)
class SpeakerSegment:
    segment_id: int
    start: float
    end: float
    speaker: str
    confidence: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "speaker": self.speaker,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class AlignedWordCandidate:
    speaker: str
    overlap_seconds: float
    overlap_ratio: float
    distance_to_center: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AlignedWord:
    word_id: str
    source_segment_id: int
    index_in_segment: int
    word: str
    start: float
    end: float
    duration: float
    probability: float | None
    speaker: str
    speaker_confidence: float
    assignment_method: str
    flags: list[str] = field(default_factory=list)
    candidates: list[AlignedWordCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "word_id": self.word_id,
            "source_segment_id": self.source_segment_id,
            "index_in_segment": self.index_in_segment,
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "probability": self.probability,
            "speaker": self.speaker,
            "speaker_confidence": self.speaker_confidence,
            "assignment_method": self.assignment_method,
            "flags": list(self.flags),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(slots=True)
class SegmentSpeakerAssignment:
    source_segment_id: int
    start: float
    end: float
    speaker: str
    speaker_confidence: float
    assignment_method: str
    flags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_segment_id": self.source_segment_id,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "speaker": self.speaker,
            "speaker_confidence": self.speaker_confidence,
            "assignment_method": self.assignment_method,
            "flags": list(self.flags),
        }


@dataclass(slots=True)
class AlignedUtterance:
    utterance_id: str
    source_segment_ids: list[int]
    start: float
    end: float
    duration: float
    speaker: str
    speaker_confidence: float
    assignment_method: str
    text: str
    word_count: int
    words: list[AlignedWord] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "source_segment_ids": list(self.source_segment_ids),
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "speaker": self.speaker,
            "speaker_confidence": self.speaker_confidence,
            "assignment_method": self.assignment_method,
            "text": self.text,
            "word_count": self.word_count,
            "words": [word.to_dict() for word in self.words],
            "flags": list(self.flags),
        }


@dataclass(slots=True)
class AlignmentConfig:
    nearest_speaker_gap_tolerance: float = 0.35
    min_words_per_split_chunk: int = 2
    min_chunk_duration: float = 0.35
    noise_bridge_max_duration: float = 0.30
    merge_same_speaker_gap_max: float = 0.60
    preview_limit: int = 50


@dataclass(slots=True)
class AlignmentPaths:
    episode_dir: Path
    transcription_dir: Path
    diarization_dir: Path
    alignment_dir: Path
    transcript_preview_path: Path
    speaker_segments_path: Path
    aligned_words_path: Path
    aligned_utterances_path: Path
    subtitles_speakers_path: Path
    alignment_metadata_path: Path
    alignment_result_path: Path
    alignment_preview_path: Path
    status_json_path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "episode_dir": str(self.episode_dir),
            "transcription_dir": str(self.transcription_dir),
            "diarization_dir": str(self.diarization_dir),
            "alignment_dir": str(self.alignment_dir),
            "transcript_preview_path": str(self.transcript_preview_path),
            "speaker_segments_path": str(self.speaker_segments_path),
            "aligned_words_path": str(self.aligned_words_path),
            "aligned_utterances_path": str(self.aligned_utterances_path),
            "subtitles_speakers_path": str(self.subtitles_speakers_path),
            "alignment_metadata_path": str(self.alignment_metadata_path),
            "alignment_result_path": str(self.alignment_result_path),
            "alignment_preview_path": str(self.alignment_preview_path),
            "status_json_path": str(self.status_json_path),
        }
