from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict


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
    words: List[Word] = field(default_factory=list)


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


# =========================
# INTERNAL MODELS
# =========================

@dataclass
class AlignedWord:
    start: float
    end: float
    word: str
    probability: Optional[float]
    speaker: str
    assignment_method: str


@dataclass
class Utterance:
    start: float
    end: float
    speaker: str
    text: str
    words: List[AlignedWord]


# =========================
# OUTPUT MODELS
# =========================

@dataclass
class AlignmentMetadata:
    word_assignment_ratio: float
    utterance_count: int
    unknown_word_ratio: float
    warnings: List[str]
