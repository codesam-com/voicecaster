from __future__ import annotations

from .schemas import AlignedWord, Utterance


def _join_words(words: list[AlignedWord]) -> str:
    return " ".join(w.word.strip() for w in words if w.word.strip()).strip()


def _mean_confidence(words: list[AlignedWord]) -> float:
    if not words:
        return 0.0
    return sum(w.speaker_confidence for w in words) / len(words)


def _make_utterance(words: list[AlignedWord], utterance_index: int) -> Utterance:
    text = _join_words(words)
    start = words[0].start
    end = words[-1].end

    return Utterance(
        utterance_id=f"utt_{utterance_index:06d}",
        source_segment_ids=sorted({w.source_segment_id for w in words}),
        start=start,
        end=end,
        duration=max(0.0, end - start),
        speaker=words[0].speaker,
        speaker_confidence=_mean_confidence(words),
        assignment_method="word_run",
        assignment_basis={
            "word_count": len(words),
            "source": "aligned_words",
        },
        text=text,
        word_count=len(words),
        words=words,
        flags=[],
    )


def build_utterances(words: list[AlignedWord]) -> list[Utterance]:
    if not words:
        return []

    utterances: list[Utterance] = []
    current: list[AlignedWord] = [words[0]]
    utterance_index = 0

    for word in words[1:]:
        prev = current[-1]
        if word.speaker == prev.speaker:
            current.append(word)
        else:
            utterances.append(_make_utterance(current, utterance_index))
            utterance_index += 1
            current = [word]

    utterances.append(_make_utterance(current, utterance_index))
    return utterances


def merge_adjacent_same_speaker_utterances(
    utterances: list[Utterance],
    max_gap: float = 0.60,
) -> list[Utterance]:
    if not utterances:
        return []

    merged: list[Utterance] = []
    current = utterances[0]

    for nxt in utterances[1:]:
        gap = nxt.start - current.end

        if nxt.speaker == current.speaker and gap <= max_gap:
            combined_words = current.words + nxt.words
            current = Utterance(
                utterance_id=current.utterance_id,
                source_segment_ids=sorted(set(current.source_segment_ids + nxt.source_segment_ids)),
                start=current.start,
                end=nxt.end,
                duration=max(0.0, nxt.end - current.start),
                speaker=current.speaker,
                speaker_confidence=(current.speaker_confidence + nxt.speaker_confidence) / 2,
                assignment_method="merged_same_speaker_gap",
                assignment_basis={
                    "left_utterance_id": current.utterance_id,
                    "right_utterance_id": nxt.utterance_id,
                    "gap": gap,
                },
                text=_join_words(combined_words),
                word_count=len(combined_words),
                words=combined_words,
                flags=sorted(set(current.flags + nxt.flags)),
            )
        else:
            merged.append(current)
            current = nxt

    merged.append(current)

    # Renumerado estable
    renumbered: list[Utterance] = []
    for idx, utt in enumerate(merged):
        renumbered.append(
            Utterance(
                utterance_id=f"utt_{idx:06d}",
                source_segment_ids=utt.source_segment_ids,
                start=utt.start,
                end=utt.end,
                duration=utt.duration,
                speaker=utt.speaker,
                speaker_confidence=utt.speaker_confidence,
                assignment_method=utt.assignment_method,
                assignment_basis=utt.assignment_basis,
                text=utt.text,
                word_count=utt.word_count,
                words=utt.words,
                flags=utt.flags,
            )
        )

    return renumbered
