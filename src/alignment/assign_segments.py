from collections import defaultdict

from .schemas import TranscriptSegment, AlignedWord


def assign_segments(
    transcript_segments: list[TranscriptSegment],
    aligned_words: list[AlignedWord],
):
    segment_speakers = []

    idx = 0

    for seg in transcript_segments:
        scores = defaultdict(float)

        seg_words = seg.words
        for _ in seg_words:
            w = aligned_words[idx]
            duration = w.end - w.start
            scores[w.speaker] += duration
            idx += 1

        if scores:
            speaker = max(scores.items(), key=lambda x: x[1])[0]
        else:
            speaker = "UNKNOWN"

        segment_speakers.append(speaker)

    return segment_speakers
