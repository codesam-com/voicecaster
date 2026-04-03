from __future__ import annotations

import re
from typing import Any

from .config import TIMESTAMP_PRECISION
from .models import AlignedUtterance


CRITICAL_FLAGS = {
    "no_speaker_overlap",
    "speaker_conflict",
    "timestamp_invalid",
}


def round_ts(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), TIMESTAMP_PRECISION)


def normalize_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def rebuild_text_from_words(words: list[dict[str, Any]]) -> str:
    raw = "".join(str(item.get("word") or "") for item in words)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def normalize_words(
    words: list[dict[str, Any]],
    speaker: str | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    normalized: list[dict[str, Any]] = []
    fixed_missing_speaker = 0
    fixed_conflicting_speaker = 0

    for item in words:
        if not isinstance(item, dict):
            continue

        word_speaker = item.get("speaker")
        start = round_ts(item.get("start")) if item.get("start") is not None else None
        end = round_ts(item.get("end")) if item.get("end") is not None else None

        normalized_item = {
            "word": str(item.get("word") or ""),
            "start": start,
            "end": end,
            "probability": item.get("probability"),
        }

        if word_speaker in ("", None):
            normalized_item["speaker"] = speaker
            if speaker is not None:
                fixed_missing_speaker += 1
        else:
            normalized_item["speaker"] = str(word_speaker)
            if speaker is not None and str(word_speaker) != speaker:
                normalized_item["speaker"] = speaker
                fixed_conflicting_speaker += 1

        normalized.append(normalized_item)

    report = {
        "words_without_speaker_fixed": fixed_missing_speaker,
        "word_speaker_conflicts_fixed": fixed_conflicting_speaker,
    }
    return normalized, report


def normalize_single_utterance(
    item: dict[str, Any],
    index: int,
    valid_speakers: set[str],
) -> tuple[AlignedUtterance, dict[str, int]]:
    source_utterance_id = str(item.get("utterance_id") or f"utt_{index:06d}")

    start = round_ts(float(item["start"]))
    end = round_ts(float(item["end"]))
    if start is None or end is None:
        raise RuntimeError(f"Utterance {source_utterance_id} is missing start/end.")

    if end <= start:
        raise RuntimeError(f"Utterance {source_utterance_id} has invalid timestamps.")

    speaker_raw = item.get("speaker")
    speaker = None if speaker_raw in ("", None) else str(speaker_raw)

    speaker_invalid = 0
    if speaker is not None and speaker not in valid_speakers:
        speaker = None
        speaker_invalid = 1

    flags = list(item.get("flags") or [])
    if speaker_invalid and "invalid_upstream_speaker" not in flags:
        flags.append("invalid_upstream_speaker")

    assignment_source = item.get("assignment_source")
    overlap_stats = dict(item.get("overlap_stats") or {})

    words_in = item.get("words") or []
    if not isinstance(words_in, list):
        words_in = []

    normalized_words, word_report = normalize_words(words_in, speaker)

    text = normalize_text(str(item.get("text") or ""))
    text_rebuilt_from_words = 0
    if not text and normalized_words:
        rebuilt = rebuild_text_from_words(normalized_words)
        if rebuilt:
            text = rebuilt
            if "text_rebuilt_from_words" not in flags:
                flags.append("text_rebuilt_from_words")
            text_rebuilt_from_words = 1

    utt = AlignedUtterance(
        utterance_id=source_utterance_id,
        source_utterance_ids=[source_utterance_id],
        start=start,
        end=end,
        duration=round_ts(end - start) or 0.0,
        speaker=speaker,
        speaker_confidence=item.get("speaker_confidence"),
        text=text,
        words=normalized_words,
        flags=flags,
        assignment_source=assignment_source,
        overlap_stats=overlap_stats,
        normalization={
            "merged_in_alignment": False,
            "text_trimmed": True,
            "word_speaker_normalized": True,
            "speaker_fallback_applied": False,
        },
    )

    report = {
        "text_rebuilt_from_words": text_rebuilt_from_words,
        "invalid_upstream_speakers_removed": speaker_invalid,
        **word_report,
    }
    return utt, report


def has_critical_flags(utt: AlignedUtterance) -> bool:
    return any(flag in CRITICAL_FLAGS for flag in utt.flags)


def can_merge_utterances(
    left: AlignedUtterance,
    right: AlignedUtterance,
    merge_gap_seconds: float,
    max_utterance_seconds: float,
    max_utterance_chars: int,
) -> bool:
    if left.speaker is None or right.speaker is None:
        return False

    if left.speaker != right.speaker:
        return False

    if has_critical_flags(left) or has_critical_flags(right):
        return False

    gap = round(float(right.start) - float(left.end), TIMESTAMP_PRECISION)

    if gap < 0:
        return False

    if gap > merge_gap_seconds:
        return False

    merged_start = left.start
    merged_end = right.end
    merged_duration = merged_end - merged_start
    if merged_duration > max_utterance_seconds:
        return False

    merged_text = normalize_text(f"{left.text} {right.text}")
    if len(merged_text) > max_utterance_chars:
        return False

    return True


def merge_utterances(
    left: AlignedUtterance,
    right: AlignedUtterance,
    merged_id: str,
) -> AlignedUtterance:
    merged_text = normalize_text(f"{left.text} {right.text}")
    merged_words = list(left.words) + list(right.words)

    confidence_values = [
        float(value)
        for value in [left.speaker_confidence, right.speaker_confidence]
        if value is not None
    ]
    merged_confidence = None
    if confidence_values:
        merged_confidence = round(sum(confidence_values) / len(confidence_values), 4)

    merged_flags = list(dict.fromkeys(list(left.flags) + list(right.flags)))
    merged_overlap_stats = {
        "merged_from": [left.utterance_id, right.utterance_id],
        "left_overlap_stats": left.overlap_stats,
        "right_overlap_stats": right.overlap_stats,
    }

    merged = AlignedUtterance(
        utterance_id=merged_id,
        source_utterance_ids=list(left.source_utterance_ids) + list(right.source_utterance_ids),
        start=left.start,
        end=right.end,
        duration=round_ts(right.end - left.start) or 0.0,
        speaker=left.speaker,
        speaker_confidence=merged_confidence,
        text=merged_text,
        words=merged_words,
        flags=merged_flags,
        assignment_source=left.assignment_source or right.assignment_source,
        overlap_stats=merged_overlap_stats,
        normalization={
            "merged_in_alignment": True,
            "text_trimmed": True,
            "word_speaker_normalized": True,
            "speaker_fallback_applied": bool(
                left.normalization.get("speaker_fallback_applied", False)
                or right.normalization.get("speaker_fallback_applied", False)
            ),
        },
    )
    return merged


def merge_adjacent_same_speaker_utterances(
    utterances: list[AlignedUtterance],
    merge_gap_seconds: float,
    max_utterance_seconds: float,
    max_utterance_chars: int,
) -> tuple[list[AlignedUtterance], int]:
    if not utterances:
        return [], 0

    merged: list[AlignedUtterance] = []
    current = utterances[0]
    merge_count = 0
    synthetic_counter = 1

    for candidate in utterances[1:]:
        if can_merge_utterances(
            current,
            candidate,
            merge_gap_seconds=merge_gap_seconds,
            max_utterance_seconds=max_utterance_seconds,
            max_utterance_chars=max_utterance_chars,
        ):
            current = merge_utterances(
                current,
                candidate,
                merged_id=f"utt_align_{synthetic_counter:06d}",
            )
            synthetic_counter += 1
            merge_count += 1
        else:
            merged.append(current)
            current = candidate

    merged.append(current)
    return merged, merge_count


def _neighbor_speaker(
    utterances: list[AlignedUtterance],
    idx: int,
    valid_speakers: set[str],
    max_gap_seconds: float = 2.0,
) -> str | None:
    current = utterances[idx]

    prev_speaker = None
    next_speaker = None

    if idx > 0:
        prev = utterances[idx - 1]
        gap_prev = current.start - prev.end
        if (
            prev.speaker in valid_speakers
            and gap_prev >= 0
            and gap_prev <= max_gap_seconds
        ):
            prev_speaker = prev.speaker

    if idx + 1 < len(utterances):
        nxt = utterances[idx + 1]
        gap_next = nxt.start - current.end
        if (
            nxt.speaker in valid_speakers
            and gap_next >= 0
            and gap_next <= max_gap_seconds
        ):
            next_speaker = nxt.speaker

    if prev_speaker and next_speaker and prev_speaker == next_speaker:
        return prev_speaker

    if prev_speaker:
        return prev_speaker

    if next_speaker:
        return next_speaker

    return None


def apply_speaker_fallbacks(
    utterances: list[AlignedUtterance],
    valid_speakers: set[str],
) -> tuple[list[AlignedUtterance], dict[str, int]]:
    fallback_applied = 0
    unresolved_before_forced = 0

    for idx, utt in enumerate(utterances):
        if utt.speaker in valid_speakers:
            continue

        fallback_speaker = _neighbor_speaker(utterances, idx, valid_speakers)
        if fallback_speaker is not None:
            utt.speaker = fallback_speaker
            utt.words = _rewrite_word_speakers(utt.words, fallback_speaker)
            if "speaker_fallback_neighbor" not in utt.flags:
                utt.flags.append("speaker_fallback_neighbor")
            utt.normalization["speaker_fallback_applied"] = True
            fallback_applied += 1

    unresolved_indices = [
        idx for idx, utt in enumerate(utterances) if utt.speaker not in valid_speakers
    ]
    unresolved_before_forced = len(unresolved_indices)

    forced_speaker = sorted(valid_speakers)[0] if valid_speakers else None
    if forced_speaker is not None:
        for idx in unresolved_indices:
            utt = utterances[idx]
            utt.speaker = forced_speaker
            utt.words = _rewrite_word_speakers(utt.words, forced_speaker)
            if "speaker_fallback_forced" not in utt.flags:
                utt.flags.append("speaker_fallback_forced")
            utt.normalization["speaker_fallback_applied"] = True

    report = {
        "speaker_fallback_applied": fallback_applied,
        "speaker_fallback_forced": unresolved_before_forced,
    }
    return utterances, report


def _rewrite_word_speakers(words: list[dict[str, Any]], speaker: str) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for item in words:
        word = dict(item)
        word["speaker"] = speaker
        rewritten.append(word)
    return rewritten


def normalize_utterances(
    utterance_dicts: list[dict[str, Any]],
    merge_gap_seconds: float,
    max_utterance_seconds: float,
    max_utterance_chars: int,
    valid_speakers: set[str],
) -> tuple[list[AlignedUtterance], dict[str, Any]]:
    normalized: list[AlignedUtterance] = []

    empty_utterances_removed = 0
    text_rebuilt_from_words = 0
    words_without_speaker_fixed = 0
    word_speaker_conflicts_fixed = 0
    invalid_upstream_speakers_removed = 0

    for idx, item in enumerate(utterance_dicts, start=1):
        if not isinstance(item, dict):
            continue

        utt, report = normalize_single_utterance(item, idx, valid_speakers=valid_speakers)

        if not utt.text and not utt.words:
            empty_utterances_removed += 1
            continue

        text_rebuilt_from_words += report["text_rebuilt_from_words"]
        words_without_speaker_fixed += report["words_without_speaker_fixed"]
        word_speaker_conflicts_fixed += report["word_speaker_conflicts_fixed"]
        invalid_upstream_speakers_removed += report["invalid_upstream_speakers_removed"]

        normalized.append(utt)

    normalized.sort(key=lambda x: (x.start, x.end, x.utterance_id))

    normalized, fallback_report = apply_speaker_fallbacks(
        normalized,
        valid_speakers=valid_speakers,
    )

    merged_output, merge_count = merge_adjacent_same_speaker_utterances(
        normalized,
        merge_gap_seconds=merge_gap_seconds,
        max_utterance_seconds=max_utterance_seconds,
        max_utterance_chars=max_utterance_chars,
    )

    report = {
        "input_utterances": len(utterance_dicts),
        "normalized_utterances_before_merge": len(normalized),
        "output_utterances": len(merged_output),
        "merged_same_speaker_utterances": merge_count,
        "empty_utterances_removed": empty_utterances_removed,
        "text_rebuilt_from_words": text_rebuilt_from_words,
        "words_without_speaker_fixed": words_without_speaker_fixed,
        "word_speaker_conflicts_fixed": word_speaker_conflicts_fixed,
        "invalid_upstream_speakers_removed": invalid_upstream_speakers_removed,
        **fallback_report,
    }

    return merged_output, report
