from __future__ import annotations

from collections import defaultdict

from .schemas import (
    AlignedUtterance,
    AlignedWord,
    AlignmentConfig,
    SegmentSpeakerAssignment,
    TranscriptSegment,
)


def _join_words(words: list[AlignedWord]) -> str:
    return " ".join(word.word for word in words if word.word).strip()


def _group_words_by_segment(aligned_words: list[AlignedWord]) -> dict[int, list[AlignedWord]]:
    grouped: dict[int, list[AlignedWord]] = defaultdict(list)
    for word in aligned_words:
        grouped[word.source_segment_id].append(word)
    return dict(grouped)


def collapse_noise_runs(
    words: list[AlignedWord],
    config: AlignmentConfig,
) -> list[AlignedWord]:
    # v1: passthrough simple.
    # La lógica fina de A-B-A se puede introducir después sin romper contratos.
    return words


def build_single_utterance_from_segment(
    segment: TranscriptSegment,
    segment_assignment: SegmentSpeakerAssignment,
    segment_words: list[AlignedWord],
    utterance_index: int,
) -> AlignedUtterance:
    text = _join_words(segment_words) or segment.text
    start = segment_words[0].start if segment_words else segment.start
    end = segment_words[-1].end if segment_words else segment.end

    return AlignedUtterance(
        utterance_id=f"utt_{utterance_index:06d}",
        source_segment_ids=[segment.segment_id],
        start=start,
        end=end,
        duration=max(0.0, end - start),
        speaker=segment_assignment.speaker,
        speaker_confidence=segment_assignment.speaker_confidence,
        assignment_method=segment_assignment.assignment_method,
        text=text,
        word_count=len(segment_words),
        words=segment_words,
        flags=list(segment_assignment.flags),
    )


def split_segment_into_speaker_runs(
    segment: TranscriptSegment,
    segment_words: list[AlignedWord],
    config: AlignmentConfig,
    utterance_index_start: int,
) -> list[AlignedUtterance]:
    if not segment_words:
        return []

    segment_words = collapse_noise_runs(segment_words, config)

    runs: list[list[AlignedWord]] = []
    current_run: list[AlignedWord] = []

    for word in segment_words:
        if not current_run:
            current_run = [word]
            continue

        if word.speaker == current_run[-1].speaker:
            current_run.append(word)
        else:
            runs.append(current_run)
            current_run = [word]

    if current_run:
        runs.append(current_run)

    utterances: list[AlignedUtterance] = []
    utterance_index = utterance_index_start

    for run in runs:
        run_start = run[0].start
        run_end = run[-1].end
        run_duration = max(0.0, run_end - run_start)

        if (
            len(run) < config.min_words_per_split_chunk
            or run_duration < config.min_chunk_duration
        ):
            # v1 conservador: no crea micro-utterances aisladas.
            # Se absorberán luego si el segmento acaba como monobloque.
            return []

        utterances.append(
            AlignedUtterance(
                utterance_id=f"utt_{utterance_index:06d}",
                source_segment_ids=[segment.segment_id],
                start=run_start,
                end=run_end,
                duration=run_duration,
                speaker=run[0].speaker,
                speaker_confidence=sum(word.speaker_confidence for word in run) / len(run),
                assignment_method="speaker_run_split",
                text=_join_words(run),
                word_count=len(run),
                words=run,
                flags=["split_from_source_segment"],
            )
        )
        utterance_index += 1

    return utterances


def merge_adjacent_same_speaker_utterances(
    utterances: list[AlignedUtterance],
    config: AlignmentConfig,
) -> list[AlignedUtterance]:
    if not utterances:
        return []

    merged: list[AlignedUtterance] = [utterances[0]]

    for current in utterances[1:]:
        previous = merged[-1]
        gap = current.start - previous.end

        if previous.speaker == current.speaker and gap <= config.merge_same_speaker_gap_max:
            previous.source_segment_ids.extend(current.source_segment_ids)
            previous.end = current.end
            previous.duration = max(0.0, previous.end - previous.start)
            previous.text = f"{previous.text} {current.text}".strip()
            previous.word_count += current.word_count
            previous.words.extend(current.words)
            previous.flags = sorted(set(previous.flags + current.flags + ["merged_same_speaker"]))
            previous.speaker_confidence = (
                previous.speaker_confidence + current.speaker_confidence
            ) / 2.0
        else:
            merged.append(current)

    return merged


def build_aligned_utterances(
    transcript_segments: list[TranscriptSegment],
    aligned_words: list[AlignedWord],
    segment_assignments: list[SegmentSpeakerAssignment],
    config: AlignmentConfig,
) -> list[AlignedUtterance]:
    words_by_segment = _group_words_by_segment(aligned_words)
    assignments_by_segment = {
        assignment.source_segment_id: assignment for assignment in segment_assignments
    }

    utterances: list[AlignedUtterance] = []
    utterance_index = 0

    for segment in transcript_segments:
        segment_words = words_by_segment.get(segment.segment_id, [])
        assignment = assignments_by_segment[segment.segment_id]

        unique_speakers = {word.speaker for word in segment_words}

        if len(unique_speakers) <= 1:
            utterances.append(
                build_single_utterance_from_segment(
                    segment=segment,
                    segment_assignment=assignment,
                    segment_words=segment_words,
                    utterance_index=utterance_index,
                )
            )
            utterance_index += 1
            continue

        split_utterances = split_segment_into_speaker_runs(
            segment=segment,
            segment_words=segment_words,
            config=config,
            utterance_index_start=utterance_index,
        )

        if split_utterances:
            utterances.extend(split_utterances)
            utterance_index += len(split_utterances)
        else:
            fallback = build_single_utterance_from_segment(
                segment=segment,
                segment_assignment=assignment,
                segment_words=segment_words,
                utterance_index=utterance_index,
            )
            fallback.flags = sorted(set(fallback.flags + ["split_aborted_due_to_micro_chunks"]))
            utterances.append(fallback)
            utterance_index += 1

    utterances.sort(key=lambda item: (item.start, item.end, item.utterance_id))
    return merge_adjacent_same_speaker_utterances(utterances, config)
