from .schemas import TranscriptSegment, SpeakerSegment, AlignedWord


def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_words(
    transcript_segments: list[TranscriptSegment],
    speaker_segments: list[SpeakerSegment],
    tolerance: float = 0.35,
) -> list[AlignedWord]:

    aligned = []

    for seg in transcript_segments:
        for word in seg.words:

            best_speaker = None
            best_overlap = 0.0

            for spk in speaker_segments:
                ov = _overlap(word.start, word.end, spk.start, spk.end)

                if ov > best_overlap:
                    best_overlap = ov
                    best_speaker = spk.speaker

            if best_speaker is None or best_overlap == 0:
                # fallback nearest
                nearest = min(
                    speaker_segments,
                    key=lambda s: min(
                        abs(word.start - s.end),
                        abs(word.end - s.start),
                    ),
                )

                gap = min(
                    abs(word.start - nearest.end),
                    abs(word.end - nearest.start),
                )

                if gap <= tolerance:
                    best_speaker = nearest.speaker
                    method = "nearest"
                else:
                    best_speaker = "UNKNOWN"
                    method = "unassigned"
            else:
                method = "overlap"

            aligned.append(
                AlignedWord(
                    start=word.start,
                    end=word.end,
                    word=word.word,
                    probability=word.probability,
                    speaker=best_speaker,
                    assignment_method=method,
                )
            )

    return aligned
