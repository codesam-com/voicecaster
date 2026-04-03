# =========================================
# FILE: src/voicecaster/alignment/metrics.py
# =========================================

from __future__ import annotations

from statistics import median

from .schemas import AlignedUtterance, AlignedWord, AlignmentMetrics, SpeakerSegment, TranscriptSegment


def compute_alignment_metrics(
    transcript_segments: list[TranscriptSegment],
    speaker_segments: list[SpeakerSegment],
    aligned_words: list[AlignedWord],
    utterances: list[AlignedUtterance],
    algorithm_version: str,
    high_unknown_word_ratio_warning: float,
    high_unknown_utterance_ratio_warning: float,
    high_multi_speaker_segment_ratio_warning: float,
) -> AlignmentMetrics:
    """
    Compute alignment summary metrics and warnings.
    """
    total_words = len(aligned_words)
    total_utterances = len(utterances)
    total_segments = len(transcript_segments)

    unknown_words = sum(1 for word in aligned_words if word.speaker == "UNKNOWN")
    unknown_utterances = sum(1 for utt in utterances if utt.speaker == "UNKNOWN")

    word_assignment_ratio = 0.0 if total_words == 0 else 1.0 - (unknown_words / total_words)
    utterance_assignment_ratio = 0.0 if total_utterances == 0 else 1.0 - (unknown_utterances / total_utterances)

    multi_speaker_source_segments = 0
    segments_with_words = 0

    for segment in transcript_segments:
        speakers = {word.speaker for word in aligned_words if word.source_segment_id == segment.segment_id}
        if speakers:
            segments_with_words += 1
            if len(speakers) > 1:
                multi_speaker_source_segments += 1

    multi_speaker_source_segment_ratio = (
        0.0 if segments_with_words == 0 else multi_speaker_source_segments / segments_with_words
    )

    split_segment_ratio = (
        0.0
        if total_segments == 0
        else max(0, total_utterances - total_segments) / total_segments
    )

    unknown_word_ratio = 0.0 if total_words == 0 else unknown_words / total_words
    unknown_utterance_ratio = 0.0 if total_utterances == 0 else unknown_utterances / total_utterances

    word_durations = [max(0.0, word.end - word.start) for word in aligned_words]
    utterance_durations = [max(0.0, utt.end - utt.start) for utt in utterances]

    warnings: list[str] = []

    if total_words == 0:
        warnings.append("missing_word_timestamps")

    if unknown_word_ratio > high_unknown_word_ratio_warning:
        warnings.append("high_unknown_word_ratio")

    if unknown_utterance_ratio > high_unknown_utterance_ratio_warning:
        warnings.append("high_unknown_utterance_ratio")

    if multi_speaker_source_segment_ratio > high_multi_speaker_segment_ratio_warning:
        warnings.append("high_multi_speaker_segment_ratio")

    if total_segments > 0 and total_utterances / total_segments > 1.35:
        warnings.append("excessive_fragmentation_after_alignment")

    return AlignmentMetrics(
        algorithm_version=algorithm_version,
        input_summary={
            "transcript_segments": total_segments,
            "transcript_words": total_words,
            "speaker_segments": len(speaker_segments),
            "speakers_detected": len({seg.speaker for seg in speaker_segments}),
        },
        output_summary={
            "aligned_words": total_words,
            "aligned_utterances": total_utterances,
            "unknown_words": unknown_words,
            "unknown_utterances": unknown_utterances,
        },
        quality_metrics={
            "word_assignment_ratio": word_assignment_ratio,
            "utterance_assignment_ratio": utterance_assignment_ratio,
            "multi_speaker_source_segment_ratio": multi_speaker_source_segment_ratio,
            "split_segment_ratio": split_segment_ratio,
            "unknown_word_ratio": unknown_word_ratio,
            "unknown_utterance_ratio": unknown_utterance_ratio,
        },
        timing_metrics={
            "median_word_duration": 0.0 if not word_durations else median(word_durations),
            "median_utterance_duration": 0.0 if not utterance_durations else median(utterance_durations),
        },
        warnings=warnings,
    )
