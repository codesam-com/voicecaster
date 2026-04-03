# =========================================
# FILE: src/voicecaster/alignment/assign_words.py
# =========================================

from __future__ import annotations

from .schemas import AlignedWord, SpeakerCandidate, SpeakerSegment, TranscriptSegment


UNKNOWN_SPEAKER = "UNKNOWN"


def assign_speakers_to_words(
    transcript_segments: list[TranscriptSegment],
    speaker_segments: list[SpeakerSegment],
    nearest_gap_tolerance: float,
) -> list[AlignedWord]:
    """
    Assign a speaker to each transcript word using temporal overlap against speaker segments.
    """
    aligned_words: list[AlignedWord] = []
    word_counter = 0
    previous_speaker: str | None = None

    for segment in transcript_segments:
        for word in segment.words:
            candidates = find_overlapping_speakers_for_word(word.start, word.end, speaker_segments)
            if candidates:
                chosen = choose_best_speaker_candidate(candidates, previous_speaker)
                speaker = chosen.speaker
                confidence = chosen.overlap_ratio
                method = "max_overlap"
                flags: list[str] = []
            else:
                nearest = assign_word_by_nearest_span(word.start, word.end, speaker_segments, nearest_gap_tolerance)
                if nearest is None:
                    speaker = UNKNOWN_SPEAKER
                    confidence = None
                    method = "unassigned"
                    flags = ["unassigned_word"]
                    candidates = []
                else:
                    speaker = nearest.speaker
                    confidence = None
                    method = "nearest_span"
                    flags = ["nearest_span_fallback"]
                    candidates = []

            aligned_word = AlignedWord(
                word_id=f"w_{word_counter:06d}",
                source_segment_id=word.source_segment_id,
                index_in_segment=word.index_in_segment,
                word=word.word,
                start=word.start,
                end=word.end,
                probability=word.probability,
                speaker=speaker,
                speaker_confidence=confidence,
                assignment_method=method,
                flags=flags,
                candidates=candidates,
            )

            aligned_words.append(aligned_word)
            previous_speaker = aligned_word.speaker
            word_counter += 1

    return aligned_words


def find_overlapping_speakers_for_word(
    word_start: float,
    word_end: float,
    speaker_segments: list[SpeakerSegment],
) -> list[SpeakerCandidate]:
    """
    Return speaker candidates that overlap the word interval.
    """
    duration = max(1e-9, word_end - word_start)
    word_center = (word_start + word_end) / 2.0
    candidates: list[SpeakerCandidate] = []

    for seg in speaker_segments:
        overlap_start = max(word_start, seg.start)
        overlap_end = min(word_end, seg.end)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap <= 0.0:
            continue

        seg_center = (seg.start + seg.end) / 2.0
        candidates.append(
            SpeakerCandidate(
                speaker=seg.speaker,
                overlap_seconds=overlap,
                overlap_ratio=overlap / duration,
                distance_to_center=abs(word_center - seg_center),
            )
        )

    return candidates


def choose_best_speaker_candidate(
    candidates: list[SpeakerCandidate],
    previous_speaker: str | None,
) -> SpeakerCandidate:
    """
    Deterministic tie-breaking:
    1. overlap_seconds desc
    2. overlap_ratio desc
    3. closest center
    4. previous speaker match
    5. lexicographic speaker id
    """
    def sort_key(candidate: SpeakerCandidate) -> tuple[float, float, float, int, str]:
        previous_match_rank = 0 if previous_speaker is not None and candidate.speaker == previous_speaker else 1
        return (
            -candidate.overlap_seconds,
            -candidate.overlap_ratio,
            candidate.distance_to_center,
            previous_match_rank,
            candidate.speaker,
        )

    return sorted(candidates, key=sort_key)[0]


def assign_word_by_nearest_span(
    word_start: float,
    word_end: float,
    speaker_segments: list[SpeakerSegment],
    nearest_gap_tolerance: float,
) -> SpeakerSegment | None:
    """
    Fallback assignment using nearest speaker segment when no overlap exists.
    """
    word_center = (word_start + word_end) / 2.0
    nearest_seg: SpeakerSegment | None = None
    nearest_distance = float("inf")

    for seg in speaker_segments:
        if word_end < seg.start:
            gap = seg.start - word_end
        elif word_start > seg.end:
            gap = word_start - seg.end
        else:
            gap = 0.0

        seg_center = (seg.start + seg.end) / 2.0
        distance = abs(word_center - seg_center)

        if gap <= nearest_gap_tolerance:
            if gap < nearest_distance:
                nearest_distance = gap
                nearest_seg = seg
            elif gap == nearest_distance and nearest_seg is not None:
                if (distance, seg.speaker) < (
                    abs(word_center - ((nearest_seg.start + nearest_seg.end) / 2.0)),
                    nearest_seg.speaker,
                ):
                    nearest_seg = seg

    return nearest_seg
