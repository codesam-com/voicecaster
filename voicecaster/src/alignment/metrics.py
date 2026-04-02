from __future__ import annotations

from .schemas import AlignedUtterance, AlignedWord, SpeakerSegment, TranscriptSegment


def collect_alignment_warnings(
    transcript_segments: list[TranscriptSegment],
    aligned_words: list[AlignedWord],
    utterances: list[AlignedUtterance],
) -> list[str]:
    warnings: list[str] = []

    total_words = len(aligned_words)
    unknown_words = sum(1 for word in aligned_words if word.speaker == "UNKNOWN")
    total_utterances = len(utterances)
    unknown_utterances = sum(1 for utt in utterances if utt.speaker == "UNKNOWN")

    if total_words == 0:
        warnings.append("missing_word_timestamps")

    if total_words > 0 and (unknown_words / total_words) > 0.01:
        warnings.append("high_unknown_word_ratio")

    if total_utterances > 0 and (unknown_utterances / total_utterances) > 0.01:
        warnings.append("high_unknown_utterance_ratio")

    multi_speaker_segments = 0
    for segment in transcript_segments:
        speakers = {word.speaker for word in aligned_words if word.source_segment_id == segment.segment_id}
        if len(speakers) > 1:
            multi_speaker_segments += 1

    if transcript_segments and (multi_speaker_segments / len(transcript_segments)) > 0.15:
        warnings.append("high_multi_speaker_segment_ratio")

    if transcript_segments and len(utterances) / len(transcript_segments) > 1.35:
        warnings.append("excessive_fragmentation_after_alignment")

    return warnings


def compute_alignment_metadata(
    transcript_segments: list[TranscriptSegment],
    speaker_segments: list[SpeakerSegment],
    aligned_words: list[AlignedWord],
    utterances: list[AlignedUtterance],
) -> dict:
    transcript_word_count = sum(len(segment.words) for segment in transcript_segments)
    unknown_words = sum(1 for word in aligned_words if word.speaker == "UNKNOWN")
    unknown_utterances = sum(1 for utt in utterances if utt.speaker == "UNKNOWN")
    multi_speaker_source_segments = 0
    split_source_segments = 0

    utterance_source_segment_counts: dict[int, int] = {}
    for utterance in utterances:
        for source_segment_id in set(utterance.source_segment_ids):
            utterance_source_segment_counts[source_segment_id] = (
                utterance_source_segment_counts.get(source_segment_id, 0) + 1
            )

    for segment in transcript_segments:
        speakers = {word.speaker for word in aligned_words if word.source_segment_id == segment.segment_id}
        if len(speakers) > 1:
            multi_speaker_source_segments += 1
        if utterance_source_segment_counts.get(segment.segment_id, 0) > 1:
            split_source_segments += 1

    merged_utterances = sum(
        1 for utterance in utterances if "merged_same_speaker" in utterance.flags
    )

    warnings = collect_alignment_warnings(
        transcript_segments=transcript_segments,
        aligned_words=aligned_words,
        utterances=utterances,
    )

    return {
        "algorithm_version": "04_alignment_v1",
        "input_summary": {
            "transcript_segments": len(transcript_segments),
            "transcript_words": transcript_word_count,
            "speaker_segments": len(speaker_segments),
            "speakers_detected": len({segment.speaker for segment in speaker_segments}),
        },
        "output_summary": {
            "aligned_words": len(aligned_words),
            "aligned_utterances": len(utterances),
            "unknown_words": unknown_words,
            "unknown_utterances": unknown_utterances,
        },
        "quality_metrics": {
            "word_assignment_ratio": 0.0 if not aligned_words else (len(aligned_words) - unknown_words) / len(aligned_words),
            "utterance_assignment_ratio": 0.0 if not utterances else (len(utterances) - unknown_utterances) / len(utterances),
            "multi_speaker_source_segment_ratio": 0.0 if not transcript_segments else multi_speaker_source_segments / len(transcript_segments),
            "split_segment_ratio": 0.0 if not transcript_segments else split_source_segments / len(transcript_segments),
            "merge_utterance_ratio": 0.0 if not utterances else merged_utterances / len(utterances),
            "unknown_word_ratio": 0.0 if not aligned_words else unknown_words / len(aligned_words),
            "unknown_utterance_ratio": 0.0 if not utterances else unknown_utterances / len(utterances),
        },
        "warnings": warnings,
    }


def build_alignment_preview(
    utterances: list[AlignedUtterance],
    metadata: dict,
    preview_limit: int = 50,
) -> dict:
    return {
        "algorithm_version": metadata.get("algorithm_version", "04_alignment_v1"),
        "summary": metadata.get("output_summary", {}),
        "quality_metrics": metadata.get("quality_metrics", {}),
        "warnings": metadata.get("warnings", []),
        "preview_utterances": [
            utterance.to_dict() for utterance in utterances[:preview_limit]
        ],
    }
