# =========================================
# FILE: src/voicecaster/alignment/split_merge.py
# =========================================

from __future__ import annotations

from collections import defaultdict

from .schemas import AlignedUtterance, AlignedWord, SegmentSpeakerAssignment, TranscriptSegment


UNKNOWN_SPEAKER = "UNKNOWN"


def split_segments_into_utterances(
    transcript_segments: list[TranscriptSegment],
    aligned_words: list[AlignedWord],
    segment_assignments: list[SegmentSpeakerAssignment],
    min_words_per_chunk: int,
    min_chunk_duration: float,
    noise_bridge_max_duration: float,
) -> list[AlignedUtterance]:
    """
    Split transcript segments into monoespeaker utterances.
    """
    words_by_segment: dict[int, list[AlignedWord]] = defaultdict(list)
    for word in aligned_words:
        words_by_segment[word.source_segment_id].append(word)

    assignments_by_segment = {
        assignment.source_segment_id: assignment
        for assignment in segment_assignments
    }

    utterances: list[AlignedUtterance] = []
    utterance_counter = 0

    for segment in transcript_segments:
        segment_words = words_by_segment.get(segment.segment_id, [])
        default_assignment = assignments_by_segment[segment.segment_id]

        if not segment_words:
            utterances.append(
                AlignedUtterance(
                    utterance_id=f"utt_{utterance_counter:06d}",
                    source_segment_ids=[segment.segment_id],
                    start=segment.start,
                    end=segment.end,
                    speaker=default_assignment.speaker,
                    speaker_confidence=default_assignment.speaker_confidence,
                    assignment_method=default_assignment.assignment_method,
                    text=segment.text,
                    word_count=0,
                    words=[],
                    flags=["no_words_in_segment"],
                )
            )
            utterance_counter += 1
            continue

        grouped = _group_consecutive_words_by_speaker(segment_words)
        grouped = _absorb_noise_bridges(grouped, noise_bridge_max_duration)

        for group in grouped:
            if len(group) < min_words_per_chunk and _group_duration(group) < min_chunk_duration:
                # Conservative fallback: keep micro chunk but flag it.
                flags = ["small_split_chunk"]
            else:
                flags = []

            text = _join_words_text(group)

            utterances.append(
                AlignedUtterance(
                    utterance_id=f"utt_{utterance_counter:06d}",
                    source_segment_ids=[segment.segment_id],
                    start=group[0].start,
                    end=group[-1].end,
                    speaker=group[0].speaker,
                    speaker_confidence=_mean_confidence(group),
                    assignment_method="word_group_split",
                    text=text,
                    word_count=len(group),
                    words=list(group),
                    flags=flags,
                )
            )
            utterance_counter += 1

    return utterances


def merge_adjacent_same_speaker_utterances(
    utterances: list[AlignedUtterance],
    max_gap: float,
) -> list[AlignedUtterance]:
    """
    Merge adjacent same-speaker utterances when the temporal gap is small.
    """
    if not utterances:
        return []

    merged: list[AlignedUtterance] = [utterances[0]]

    for current in utterances[1:]:
        previous = merged[-1]
        gap = max(0.0, current.start - previous.end)

        if previous.speaker == current.speaker and gap <= max_gap:
            merged[-1] = AlignedUtterance(
                utterance_id=previous.utterance_id,
                source_segment_ids=previous.source_segment_ids + current.source_segment_ids,
                start=previous.start,
                end=current.end,
                speaker=previous.speaker,
                speaker_confidence=_merge_confidences(previous.speaker_confidence, current.speaker_confidence),
                assignment_method="merge_same_speaker_gap",
                text=_merge_text(previous.text, current.text),
                word_count=previous.word_count + current.word_count,
                words=previous.words + current.words,
                flags=previous.flags + current.flags + ["merged_same_speaker"],
            )
        else:
            merged.append(current)

    return merged


def _group_consecutive_words_by_speaker(words: list[AlignedWord]) -> list[list[AlignedWord]]:
    groups: list[list[AlignedWord]] = []
    current_group: list[AlignedWord] = []

    for word in words:
        if not current_group or word.speaker == current_group[-1].speaker:
            current_group.append(word)
        else:
            groups.append(current_group)
            current_group = [word]

    if current_group:
        groups.append(current_group)

    return groups


def _absorb_noise_bridges(
    groups: list[list[AlignedWord]],
    noise_bridge_max_duration: float,
) -> list[list[AlignedWord]]:
    """
    Absorb A-B-A micro bridges when B is very short.
    """
    if len(groups) < 3:
        return groups

    result = [groups[0]]

    for idx in range(1, len(groups) - 1):
        prev_group = result[-1]
        current_group = groups[idx]
        next_group = groups[idx + 1]

        current_duration = _group_duration(current_group)

        if (
            len(current_group) == 1
            and current_duration <= noise_bridge_max_duration
            and prev_group[0].speaker == next_group[0].speaker
        ):
            prev_group.extend(current_group)
        else:
            result.append(current_group)

    # Append last group if not already merged into previous.
    last_group = groups[-1]
    if result[-1] is not last_group:
        result.append(last_group)

    # Re-group in case a merge created mixed speaker groups.
    flattened: list[AlignedWord] = [word for group in result for word in group]
    return _group_consecutive_words_by_speaker(flattened)


def _group_duration(group: list[AlignedWord]) -> float:
    if not group:
        return 0.0
    return max(0.0, group[-1].end - group[0].start)


def _join_words_text(words: list[AlignedWord]) -> str:
    return " ".join(word.word for word in words).strip()


def _mean_confidence(words: list[AlignedWord]) -> float | None:
    values = [word.speaker_confidence for word in words if word.speaker_confidence is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _merge_confidences(a: float | None, b: float | None) -> float | None:
    values = [value for value in (a, b) if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _merge_text(a: str, b: str) -> str:
    a = a.strip()
    b = b.strip()
    if not a:
        return b
    if not b:
        return a
    return f"{a} {b}"
