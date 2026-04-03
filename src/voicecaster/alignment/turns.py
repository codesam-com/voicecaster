from __future__ import annotations

from .config import TIMESTAMP_PRECISION
from .models import AlignedTurn, AlignedUtterance


def _round(value: float) -> float:
    return round(float(value), TIMESTAMP_PRECISION)


def _can_merge_into_turn(
    current_items: list[AlignedUtterance],
    candidate: AlignedUtterance,
    merge_gap_seconds: float,
    max_duration_seconds: float,
    max_chars: int,
) -> bool:
    if not current_items:
        return True

    last = current_items[-1]

    if last.speaker != candidate.speaker:
        return False

    gap = candidate.start - last.end
    if gap < 0:
        return False

    if gap > merge_gap_seconds:
        return False

    new_start = current_items[0].start
    new_end = candidate.end
    new_duration = new_end - new_start
    if new_duration > max_duration_seconds:
        return False

    current_text = " ".join(item.text for item in current_items if item.text).strip()
    new_text = f"{current_text} {candidate.text}".strip()
    if len(new_text) > max_chars:
        return False

    return True


def _build_turn(turn_id: str, utterances: list[AlignedUtterance]) -> AlignedTurn:
    start = utterances[0].start
    end = utterances[-1].end
    text = " ".join(item.text for item in utterances if item.text).strip()
    word_count = sum(len(item.words) for item in utterances)

    confidence_values = [
        float(item.speaker_confidence)
        for item in utterances
        if item.speaker_confidence is not None
    ]
    avg_confidence = None
    if confidence_values:
        avg_confidence = round(sum(confidence_values) / len(confidence_values), 4)

    return AlignedTurn(
        turn_id=turn_id,
        speaker=utterances[0].speaker,
        start=_round(start),
        end=_round(end),
        duration=_round(end - start),
        utterance_ids=[item.utterance_id for item in utterances],
        text=text,
        num_utterances=len(utterances),
        num_words=word_count,
        avg_speaker_confidence=avg_confidence,
        flags=[],
    )


def build_turns(
    utterances: list[AlignedUtterance],
    merge_gap_seconds: float,
    max_duration_seconds: float,
    max_chars: int,
) -> tuple[list[AlignedTurn], dict[str, float | int]]:
    turns: list[AlignedTurn] = []
    bucket: list[AlignedUtterance] = []

    for utt in utterances:
        if not bucket:
            bucket = [utt]
            continue

        if _can_merge_into_turn(
            bucket,
            utt,
            merge_gap_seconds=merge_gap_seconds,
            max_duration_seconds=max_duration_seconds,
            max_chars=max_chars,
        ):
            bucket.append(utt)
        else:
            turns.append(_build_turn(f"turn_{len(turns)+1:06d}", bucket))
            bucket = [utt]

    if bucket:
        turns.append(_build_turn(f"turn_{len(turns)+1:06d}", bucket))

    durations = [turn.duration for turn in turns]
    avg_duration = round(sum(durations) / len(durations), 3) if durations else 0.0
    max_duration = round(max(durations), 3) if durations else 0.0

    report = {
        "output_turns": len(turns),
        "avg_turn_duration_seconds": avg_duration,
        "max_turn_duration_seconds": max_duration,
    }

    return turns, report
