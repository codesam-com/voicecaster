from __future__ import annotations

from .schemas import (
    AlignedWord,
    AlignedWordCandidate,
    AlignmentConfig,
    SpeakerSegment,
    TranscriptSegment,
    TranscriptWord,
)


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _center_distance(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    a_center = (a_start + a_end) / 2.0
    b_center = (b_start + b_end) / 2.0
    return abs(a_center - b_center)


def find_overlapping_speaker_candidates(
    word: TranscriptWord,
    speaker_segments: list[SpeakerSegment],
) -> list[AlignedWordCandidate]:
    candidates: list[AlignedWordCandidate] = []

    word_duration = max(1e-9, word.duration)

    for segment in speaker_segments:
        overlap = _overlap_seconds(word.start, word.end, segment.start, segment.end)
        if overlap <= 0:
            continue

        candidates.append(
            AlignedWordCandidate(
                speaker=segment.speaker,
                overlap_seconds=overlap,
                overlap_ratio=overlap / word_duration,
                distance_to_center=_center_distance(
                    word.start,
                    word.end,
                    segment.start,
                    segment.end,
                ),
            )
        )

    candidates.sort(
        key=lambda item: (
            -item.overlap_seconds,
            -item.overlap_ratio,
            item.distance_to_center,
            item.speaker,
        )
    )
    return candidates


def pick_best_candidate(
    candidates: list[AlignedWordCandidate],
    previous_speaker: str | None = None,
) -> AlignedWordCandidate | None:
    if not candidates:
        return None

    best = candidates[0]
    tied = [
        candidate
        for candidate in candidates
        if candidate.overlap_seconds == best.overlap_seconds
        and candidate.overlap_ratio == best.overlap_ratio
        and candidate.distance_to_center == best.distance_to_center
    ]

    if previous_speaker:
        for candidate in tied:
            if candidate.speaker == previous_speaker:
                return candidate

    return best


def assign_unknown_or_nearest(
    word: TranscriptWord,
    speaker_segments: list[SpeakerSegment],
    config: AlignmentConfig,
) -> tuple[str, float, str, list[str], list[AlignedWordCandidate]]:
    nearest_segment = None
    nearest_gap = None

    for segment in speaker_segments:
        if segment.end < word.start:
            gap = word.start - segment.end
        elif segment.start > word.end:
            gap = segment.start - word.end
        else:
            gap = 0.0

        if nearest_gap is None or gap < nearest_gap:
            nearest_gap = gap
            nearest_segment = segment

    if (
        nearest_segment is not None
        and nearest_gap is not None
        and nearest_gap <= config.nearest_speaker_gap_tolerance
    ):
        return (
            nearest_segment.speaker,
            0.5,
            "nearest_span",
            ["nearest_span_assignment"],
            [],
        )

    return (
        "UNKNOWN",
        0.0,
        "unknown",
        ["unassigned_word"],
        [],
    )


def assign_speakers_to_words(
    transcript_segments: list[TranscriptSegment],
    speaker_segments: list[SpeakerSegment],
    config: AlignmentConfig,
) -> list[AlignedWord]:
    aligned_words: list[AlignedWord] = []
    previous_speaker: str | None = None
    word_counter = 0

    for segment in transcript_segments:
        for word in segment.words:
            candidates = find_overlapping_speaker_candidates(word, speaker_segments)
            best = pick_best_candidate(candidates, previous_speaker=previous_speaker)

            if best is not None:
                speaker = best.speaker
                speaker_confidence = min(1.0, best.overlap_ratio)
                assignment_method = "max_overlap"
                flags: list[str] = []
            else:
                speaker, speaker_confidence, assignment_method, flags, nearest_candidates = (
                    assign_unknown_or_nearest(word, speaker_segments, config)
                )
                if nearest_candidates:
                    candidates = nearest_candidates

            aligned_word = AlignedWord(
                word_id=f"w_{word_counter:06d}",
                source_segment_id=word.source_segment_id,
                index_in_segment=word.index_in_segment,
                word=word.word,
                start=word.start,
                end=word.end,
                duration=word.duration,
                probability=word.probability,
                speaker=speaker,
                speaker_confidence=speaker_confidence,
                assignment_method=assignment_method,
                flags=flags,
                candidates=candidates,
            )
            aligned_words.append(aligned_word)

            previous_speaker = aligned_word.speaker
            word_counter += 1

    return aligned_words
