from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .models import SpeakerSegment, TranscriptUtterance


@dataclass
class QAIssue:
    severity: str  # info | warning | error
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QAResult:
    passed: bool
    issues: list[QAIssue]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "stats": self.stats,
        }


def run_diarization_qa(
    speaker_segments: list[SpeakerSegment],
    utterances: list[TranscriptUtterance],
    *,
    min_assignment_ratio: float,
    max_reasonable_speakers: int = 8,
    max_short_segment_ratio: float = 0.35,
    max_low_confidence_ratio: float = 0.35,
) -> QAResult:
    issues: list[QAIssue] = []

    stats = _build_stats(speaker_segments, utterances)

    if not speaker_segments:
        issues.append(
            QAIssue(
                severity="error",
                code="no_speaker_segments",
                message="No speaker segments were produced.",
            )
        )

    if not utterances:
        issues.append(
            QAIssue(
                severity="error",
                code="no_utterances",
                message="No transcript utterances were available after reconciliation.",
            )
        )

    invalid_segments = [
        seg.segment_id
        for seg in speaker_segments
        if seg.start < 0 or seg.end <= seg.start or seg.duration <= 0
    ]
    if invalid_segments:
        issues.append(
            QAIssue(
                severity="error",
                code="invalid_segment_timing",
                message="One or more normalized speaker segments have invalid timing.",
                details={"segment_ids": invalid_segments[:20]},
            )
        )

    unordered_segments = _count_unordered_segments(speaker_segments)
    if unordered_segments > 0:
        issues.append(
            QAIssue(
                severity="error",
                code="unordered_segments",
                message="Speaker segments are not strictly ordered by time.",
                details={"count": unordered_segments},
            )
        )

    assignment_ratio = stats["assignment_ratio"]
    if assignment_ratio < min_assignment_ratio:
        issues.append(
            QAIssue(
                severity="error",
                code="low_assignment_ratio",
                message="Too many transcript utterances could not be assigned to a speaker.",
                details={
                    "assignment_ratio": assignment_ratio,
                    "required_minimum": min_assignment_ratio,
                },
            )
        )

    num_speakers = stats["num_speakers"]
    if num_speakers > max_reasonable_speakers:
        issues.append(
            QAIssue(
                severity="warning",
                code="too_many_speakers_detected",
                message="Detected speaker count is unusually high.",
                details={
                    "num_speakers": num_speakers,
                    "max_reasonable_speakers": max_reasonable_speakers,
                },
            )
        )

    short_segment_ratio = stats["short_segment_ratio"]
    if short_segment_ratio > max_short_segment_ratio:
        issues.append(
            QAIssue(
                severity="warning",
                code="high_short_segment_ratio",
                message="A high fraction of segments are very short, suggesting over-fragmentation.",
                details={
                    "short_segment_ratio": short_segment_ratio,
                    "threshold": max_short_segment_ratio,
                },
            )
        )

    rapid_switches = _count_rapid_alternations(speaker_segments)
    if rapid_switches > 0:
        issues.append(
            QAIssue(
                severity="warning",
                code="rapid_speaker_alternation",
                message="Detected suspicious rapid alternation between speakers.",
                details={"count": rapid_switches},
            )
        )

    low_conf_ratio = stats["low_confidence_ratio"]
    if low_conf_ratio > max_low_confidence_ratio:
        issues.append(
            QAIssue(
                severity="warning",
                code="high_low_confidence_ratio",
                message="A large fraction of utterances were assigned with low confidence.",
                details={
                    "low_confidence_ratio": low_conf_ratio,
                    "threshold": max_low_confidence_ratio,
                },
            )
        )

    dominant_ratio = stats["dominant_speaker_ratio"]
    if dominant_ratio is not None and dominant_ratio > 0.95 and num_speakers > 1:
        issues.append(
            QAIssue(
                severity="warning",
                code="dominant_speaker_extreme",
                message="One speaker dominates almost the entire episode despite multiple detected speakers.",
                details={"dominant_speaker_ratio": dominant_ratio},
            )
        )

    passed = not any(issue.severity == "error" for issue in issues)

    return QAResult(
        passed=passed,
        issues=issues,
        stats=stats,
    )


def _build_stats(
    speaker_segments: list[SpeakerSegment],
    utterances: list[TranscriptUtterance],
) -> dict[str, Any]:
    speakers = sorted({seg.speaker for seg in speaker_segments})
    num_speakers = len(speakers)

    total_segment_seconds = round(sum(seg.duration for seg in speaker_segments), 3)
    short_segments = [seg for seg in speaker_segments if seg.duration < 1.0]
    short_segment_ratio = (
        round(len(short_segments) / len(speaker_segments), 4)
        if speaker_segments
        else 0.0
    )

    assigned_utterances = [utt for utt in utterances if utt.speaker]
    assignment_ratio = (
        round(len(assigned_utterances) / len(utterances), 4) if utterances else 0.0
    )

    low_confidence_utterances = [
        utt for utt in utterances if "low_confidence_assignment" in utt.flags
    ]
    low_confidence_ratio = (
        round(len(low_confidence_utterances) / len(utterances), 4)
        if utterances
        else 0.0
    )

    seconds_by_speaker: dict[str, float] = {}
    for seg in speaker_segments:
        seconds_by_speaker.setdefault(seg.speaker, 0.0)
        seconds_by_speaker[seg.speaker] += seg.duration

    dominant_speaker_ratio = None
    if total_segment_seconds > 0 and seconds_by_speaker:
        dominant_speaker_ratio = round(
            max(seconds_by_speaker.values()) / total_segment_seconds,
            4,
        )

    return {
        "num_speakers": num_speakers,
        "total_segment_seconds": total_segment_seconds,
        "assignment_ratio": assignment_ratio,
        "short_segment_ratio": short_segment_ratio,
        "low_confidence_ratio": low_confidence_ratio,
        "dominant_speaker_ratio": dominant_speaker_ratio,
    }


def _count_unordered_segments(speaker_segments: list[SpeakerSegment]) -> int:
    count = 0
    previous_end: float | None = None

    for seg in speaker_segments:
        if previous_end is not None and seg.start < previous_end:
            count += 1
        previous_end = seg.end

    return count


def _count_rapid_alternations(speaker_segments: list[SpeakerSegment]) -> int:
    count = 0

    for i in range(1, len(speaker_segments) - 1):
        prev_seg = speaker_segments[i - 1]
        curr_seg = speaker_segments[i]
        next_seg = speaker_segments[i + 1]

        if (
            prev_seg.speaker == next_seg.speaker
            and curr_seg.speaker != prev_seg.speaker
            and curr_seg.duration < 1.0
        ):
            count += 1

    return count
