from __future__ import annotations

from .schemas import TranscriptSegment, SpeakerSegment, AlignedWord


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _safe_duration(start: float, end: float) -> float:
    return max(0.0, end - start)


def assign_words(
    transcript_segments: list[TranscriptSegment],
    speaker_segments: list[SpeakerSegment],
    tolerance: float = 0.35,
) -> list[AlignedWord]:
    aligned: list[AlignedWord] = []
    global_word_index = 0

    for seg in transcript_segments:
        for idx_in_segment, word in enumerate(seg.words):
            word_duration = _safe_duration(word.start, word.end)

            best_speaker = None
            best_overlap = 0.0

            for spk in speaker_segments:
                ov = _overlap(word.start, word.end, spk.start, spk.end)
                if ov > best_overlap:
                    best_overlap = ov
                    best_speaker = spk.speaker

            if best_speaker is not None and best_overlap > 0:
                method = "max_overlap"
                confidence = 1.0 if word_duration == 0 else min(1.0, best_overlap / word_duration)
                speaker = best_speaker
                flags: list[str] = []
            else:
                nearest = min(
                    speaker_segments,
                    key=lambda s: min(abs(word.start - s.end), abs(word.end - s.start)),
                )
                gap = min(abs(word.start - nearest.end), abs(word.end - nearest.start))

                if gap <= tolerance:
                    speaker = nearest.speaker
                    method = "nearest_span"
                    confidence = 0.5
                    flags = ["gap_fallback"]
                else:
                    speaker = "UNKNOWN"
                    method = "unassigned"
                    confidence = 0.0
                    flags = ["unassigned_word"]

            aligned.append(
                AlignedWord(
                    word_id=f"w_{global_word_index:06d}",
                    source_segment_id=seg.id,
                    index_in_segment=idx_in_segment,
                    start=word.start,
                    end=word.end,
                    duration=_safe_duration(word.start, word.end),
                    word=word.word,
                    probability=word.probability,
                    speaker=speaker,
                    speaker_confidence=confidence,
                    assignment_method=method,
                    flags=flags,
                )
            )
            global_word_index += 1

    return aligned
