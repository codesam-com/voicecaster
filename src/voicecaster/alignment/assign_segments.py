# =========================================
# FILE: src/voicecaster/alignment/assign_segments.py
# =========================================

from __future__ import annotations

from collections import defaultdict

from .schemas import AlignedWord, SegmentSpeakerAssignment, SpeakerSegment, TranscriptSegment


UNKNOWN_SPEAKER = "UNKNOWN"


def assign_speakers_to_segments(
    transcript_segments: list[TranscriptSegment],
    aligned_words: list[AlignedWord],
    speaker_segments: list[SpeakerSegment],
) -> list[SegmentSpeakerAssignment]:
    """
    Assign dominant speaker to each transcript segment.
    """
    words_by_segment: dict[int, list[AlignedWord]] = defaultdict(list)
    for word in aligned_words:
        words_by_segment[word.source_segment_id].append(word)

    assignments: list[SegmentSpeakerAssignment] = []

    for segment in transcript_segments:
        segment_words = words_by_segment.get(segment.segment_id, [])

        if segment_words:
            assignment = _assign_segment_from_words(segment, segment_words, speaker_segments)
        else:
            assignment = _assign_segment_from_overlap(segment, speaker_segments)

        assignments.append(assignment)

    return assignments


def _assign_segment_from_words(
    segment: TranscriptSegment,
    words: list[AlignedWord],
    speaker_segments: list[SpeakerSegment],
) -> SegmentSpeakerAssignment:
    speaker_durations: dict[str, float] = defaultdict(float)

    for word in words:
        speaker_durations[word.speaker] += max(0.0, word.end - word.start)

    ranked = sorted(
        speaker_durations.items(),
        key=lambda item: (-item[1], item[0]),
    )

    winner_speaker = ranked[0][0]
    total_duration = sum(speaker_durations.values())
    winner_duration = ranked[0][1]
    confidence = None if total_duration <= 0.0 else winner_duration / total_duration

    details = {
        "speaker_duration_votes": [
            {"speaker": speaker, "duration": duration}
            for speaker, duration in ranked
        ]
    }

    return SegmentSpeakerAssignment(
        source_segment_id=segment.segment_id,
        start=segment.start,
        end=segment.end,
        speaker=winner_speaker,
        speaker_confidence=confidence,
        assignment_method="word_duration_vote",
        flags=[],
        details=details,
    )


def _assign_segment_from_overlap(
    segment: TranscriptSegment,
    speaker_segments: list[SpeakerSegment],
) -> SegmentSpeakerAssignment:
    speaker_overlaps: dict[str, float] = defaultdict(float)

    for seg in speaker_segments:
        overlap_start = max(segment.start, seg.start)
        overlap_end = min(segment.end, seg.end)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > 0.0:
            speaker_overlaps[seg.speaker] += overlap

    if not speaker_overlaps:
        return SegmentSpeakerAssignment(
            source_segment_id=segment.segment_id,
            start=segment.start,
            end=segment.end,
            speaker=UNKNOWN_SPEAKER,
            speaker_confidence=None,
            assignment_method="unassigned",
            flags=["unassigned_segment"],
            details={},
        )

    ranked = sorted(
        speaker_overlaps.items(),
        key=lambda item: (-item[1], item[0]),
    )

    winner_speaker = ranked[0][0]
    total_overlap = sum(speaker_overlaps.values())
    winner_overlap = ranked[0][1]
    confidence = None if total_overlap <= 0.0 else winner_overlap / total_overlap

    return SegmentSpeakerAssignment(
        source_segment_id=segment.segment_id,
        start=segment.start,
        end=segment.end,
        speaker=winner_speaker,
        speaker_confidence=confidence,
        assignment_method="segment_overlap_fallback",
        flags=["segment_overlap_fallback"],
        details={
            "speaker_overlap_votes": [
                {"speaker": speaker, "overlap_seconds": overlap}
                for speaker, overlap in ranked
            ]
        },
    )
