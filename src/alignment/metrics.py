from __future__ import annotations

from statistics import median

from .schemas import AlignmentMetadata, AlignedWord, SpeakerSegment, TranscriptSegment, Utterance


def compute_metrics(
    transcript_segments: list[TranscriptSegment],
    speaker_segments: list[SpeakerSegment],
    aligned_words: list[AlignedWord],
    utterances: list[Utterance],
    algorithm_version: str,
) -> AlignmentMetadata:
    total_words = len(aligned_words)
    total_segments = len(transcript_segments)
    total_speaker_segments = len(speaker_segments)

    unknown_words = sum(1 for w in aligned_words if w.speaker == "UNKNOWN")
    unknown_utterances = sum(1 for u in utterances if u.speaker == "UNKNOWN")

    multi_speaker_source_segment_ids = set()
    by_segment: dict[int, set[str]] = {}
    for word in aligned_words:
        by_segment.setdefault(word.source_segment_id, set()).add(word.speaker)

    for segment_id, speakers in by_segment.items():
        normalized = {s for s in speakers if s != "UNKNOWN"}
        if len(normalized) > 1:
            multi_speaker_source_segment_ids.add(segment_id)

    word_assignment_ratio = 1.0 - (unknown_words / total_words) if total_words else 0.0
    utterance_assignment_ratio = 1.0 - (unknown_utterances / len(utterances)) if utterances else 0.0
    unknown_word_ratio = unknown_words / total_words if total_words else 0.0
    unknown_utterance_ratio = unknown_utterances / len(utterances) if utterances else 0.0
    multi_speaker_source_segment_ratio = (
        len(multi_speaker_source_segment_ids) / total_segments if total_segments else 0.0
    )

    warnings: list[str] = []
    if unknown_word_ratio > 0.01:
        warnings.append("high_unknown_word_ratio")
    if unknown_utterance_ratio > 0.01:
        warnings.append("high_unknown_utterance_ratio")
    if multi_speaker_source_segment_ratio > 0.15:
        warnings.append("high_multi_speaker_segment_ratio")
    if total_words == 0:
        warnings.append("missing_word_timestamps")

    word_durations = [w.duration for w in aligned_words if w.duration > 0]
    utt_durations = [u.duration for u in utterances if u.duration > 0]

    return AlignmentMetadata(
        algorithm_version=algorithm_version,
        input_summary={
            "transcript_segments": total_segments,
            "transcript_words": total_words,
            "speaker_segments": total_speaker_segments,
            "speakers_detected": len({s.speaker for s in speaker_segments}),
        },
        output_summary={
            "aligned_words": total_words,
            "aligned_utterances": len(utterances),
            "unknown_words": unknown_words,
            "unknown_utterances": unknown_utterances,
        },
        quality_metrics={
            "word_assignment_ratio": word_assignment_ratio,
            "utterance_assignment_ratio": utterance_assignment_ratio,
            "multi_speaker_source_segment_ratio": multi_speaker_source_segment_ratio,
            "unknown_word_ratio": unknown_word_ratio,
            "unknown_utterance_ratio": unknown_utterance_ratio,
        },
        timing_metrics={
            "median_word_duration": median(word_durations) if word_durations else 0.0,
            "median_utterance_duration": median(utt_durations) if utt_durations else 0.0,
        },
        warnings=warnings,
    )
