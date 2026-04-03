from __future__ import annotations

from collections import defaultdict

from .schemas import TranscriptSegment, SpeakerSegment, AlignedWord


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_segments(
    transcript_segments: list[TranscriptSegment],
    aligned_words: list[AlignedWord],
    speaker_segments: list[SpeakerSegment],
) -> list[dict]:
    results: list[dict] = []

    words_by_segment: dict[int, list[AlignedWord]] = {}
    for word in aligned_words:
        words_by_segment.setdefault(word.source_segment_id, []).append(word)

    for seg in transcript_segments:
        seg_words = words_by_segment.get(seg.id, [])

        if seg_words:
            scores = defaultdict(float)
            for w in seg_words:
                scores[w.speaker] += w.duration

            speaker = max(scores.items(), key=lambda x: x[1])[0]
            total = sum(scores.values()) or 1.0
            confidence = scores[speaker] / total
            method = "word_majority_duration"
            candidates = [
                {
                    "speaker": spk,
                    "score": score,
                    "ratio": score / total,
                }
                for spk, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ]
        else:
            scores = defaultdict(float)
            for spk in speaker_segments:
                ov = _overlap(seg.start, seg.end, spk.start, spk.end)
                if ov > 0:
                    scores[spk.speaker] += ov

            if scores:
                speaker = max(scores.items(), key=lambda x: x[1])[0]
                total = sum(scores.values()) or 1.0
                confidence = scores[speaker] / total
                method = "segment_overlap_fallback"
                candidates = [
                    {
                        "speaker": spk,
                        "score": score,
                        "ratio": score / total,
                    }
                    for spk, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
                ]
            else:
                speaker = "UNKNOWN"
                confidence = 0.0
                method = "unassigned"
                candidates = []

        results.append(
            {
                "segment_id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "speaker": speaker,
                "speaker_confidence": confidence,
                "assignment_method": method,
                "speaker_candidates": candidates,
            }
        )

    return results
