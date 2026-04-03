from __future__ import annotations

from .config import TIMESTAMP_PRECISION
from .models import AlignedTurn, AlignedUtterance, AlignedWord


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), TIMESTAMP_PRECISION)


def flatten_aligned_words(
    utterances: list[AlignedUtterance],
    turns: list[AlignedTurn],
) -> list[AlignedWord]:
    utterance_to_turn: dict[str, str] = {}
    for turn in turns:
        for utterance_id in turn.utterance_ids:
            utterance_to_turn[utterance_id] = turn.turn_id

    words: list[AlignedWord] = []
    counter = 0

    for utt in utterances:
        turn_id = utterance_to_turn.get(utt.utterance_id)

        for word_item in utt.words:
            counter += 1
            start = _round_or_none(word_item.get("start"))
            end = _round_or_none(word_item.get("end"))
            duration = None
            if start is not None and end is not None and end >= start:
                duration = round(end - start, TIMESTAMP_PRECISION)

            word = AlignedWord(
                word_id=f"word_{counter:06d}",
                utterance_id=utt.utterance_id,
                turn_id=turn_id,
                speaker=word_item.get("speaker") or utt.speaker,
                word=str(word_item.get("word") or ""),
                start=start,
                end=end,
                duration=duration,
                probability=word_item.get("probability"),
                flags=[],
            )
            words.append(word)

    return words
