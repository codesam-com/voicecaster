from __future__ import annotations

from collections import defaultdict

from .schemas import (
    AlignedWord,
    SegmentSpeakerAssignment,
    SpeakerSegment,
    TranscriptSegment,
)


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def group_words_by_source_segment(
    aligned_words: list[AlignedWord],
) -> dict[int, list[AlignedWord]]:
    grouped: dict[int, list[AlignedWord]] = defaultdict(list)
    for word in aligned_words:
        grouped[word.source_segment_id].append(word)
    return dict(grouped)


def assign_segment_from_words(
    segment: TranscriptSegment,
    words: list[AlignedWord],
) -> SegmentSpeakerAssignment:
    speaker_scores: dict[str, float] = defaultdict(float)

    for word in words:
        speaker_scores[word.speaker] += word.duration

    speaker, score = max(
        speaker_scores.items(),
        key=lambda item: (item[1], item[0]),
    )

    total = sum(speaker_scores.values()) or 1.0
    confidence = score / total

    flags: list[str] = []
    if len(speaker_scores) > 1:
        flags.append("multi_speaker_source_segment")

    return SegmentSpeakerAssignment(
        source_segment_id=segment.segment_id,
        start=segment.start,
        end=segment.end,
        speaker=speaker,
        speaker_confidence=confidence,
        assignment_method="word_majority_duration",
        flags=flags,
    )


def assign_segment_from_overlap(
    segment: TranscriptSegment,
    speaker_segments: list[SpeakerSegment],
) -> SegmentSpeakerAssignment:
    speaker_scores: dict[str, float] = defaultdict(float)

    for speaker_segment in speaker_segments:
        overlap = _overlap_seconds(
            segment.start,
            segment.end,
            speaker_segment.start,
            speaker_segment.end,
        )
        if overlap > 0:
            speaker_scores[speaker_segment.speaker] += overlap

    if not speaker_scores:
        return SegmentSpeakerAssignment(
            source_segment_id=segment.segment_id,
            start=segment.start,
            end=segment.end,
            speaker="UNKNOWN",
            speaker_confidence=0.0,
            assignment_method="unknown",
            flags=["unassigned_segment"],
        )

    speaker, score = max(
        speaker_scores.items(),
        key=lambda item: (item[1], item[0]),
    )
    total = sum(speaker_scores.values()) or 1.0

    return SegmentSpeakerAssignment(
        source_segment_id=segment.segment_id,
        start=segment.start,
        end=segment.end,
        speaker=speaker,
        speaker_confidence=score / total,
        assignment_method="segment_overlap",
        flags=[],
    )


def assign_speakers_to_segments(
    transcript_segments: list[TranscriptSegment],
    aligned_words: list[AlignedWord],
    speaker_segments: list[SpeakerSegment],
) -> list[SegmentSpeakerAssignment]:
    grouped_words = group_words_by_source_segment(aligned_words)
    assignments: list[SegmentSpeakerAssignment] = []

    for segment in transcript_segments:
        segment_words = grouped_words.get(segment.segment_id, [])
        if segment_words:
            assignments.append(assign_segment_from_words(segment, segment_words))
        else:
            assignments.append(assign_segment_from_overlap(segment, speaker_segments))

    return assignments
