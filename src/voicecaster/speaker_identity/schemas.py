from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SpeakerEpisodeSummary:
    speaker: str
    speech_seconds: float
    num_turns: int
    num_utterances: int
    num_words: int
    first_seen: float | None
    last_seen: float | None
    share_of_speech: float
    identity_state: str
    review_required: bool
    usable_for_identity: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IdentityCandidate:
    candidate_type: str
    speaker_id: str | None
    display_name: str
    voice_score: float
    text_score: float
    context_score: float
    final_score: float
    decision_band: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IdentityDecision:
    speaker: str
    proposed_identity: str | None
    proposed_display_name: str
    confidence: float
    identity_state: str
    review_required: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IdentityQAIssue:
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IdentityQAResult:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    issues: list[IdentityQAIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "issues": [issue.to_dict() for issue in self.issues],
        }
