from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AlignedWord:
    word_id: str
    utterance_id: str
    turn_id: str | None
    speaker: str | None
    word: str
    start: float | None
    end: float | None
    duration: float | None
    probability: float | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlignedUtterance:
    utterance_id: str
    source_utterance_ids: list[str]
    start: float
    end: float
    duration: float
    speaker: str | None
    speaker_confidence: float | None
    text: str
    words: list[dict[str, Any]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    assignment_source: str | None = None
    overlap_stats: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlignedTurn:
    turn_id: str
    speaker: str | None
    start: float
    end: float
    duration: float
    utterance_ids: list[str]
    text: str
    num_utterances: int
    num_words: int
    avg_speaker_confidence: float | None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QAIssue:
    severity: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QAResult:
    passed: bool
    checks: dict[str, bool]
    warnings: list[QAIssue] = field(default_factory=list)
    issues: list[QAIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "warnings": [w.to_dict() for w in self.warnings],
            "issues": [i.to_dict() for i in self.issues],
        }
