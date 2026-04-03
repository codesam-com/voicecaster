from __future__ import annotations

import re
from typing import Any

from .config import TIMESTAMP_PRECISION
from .models import AlignedUtterance


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
        normalized_item = {
            "word": str(item.get("word") or ""),
            "start": round_ts(item.get("start")) if item.get("start") is not None else None,
            "end": round_ts(item.get("end")) if item.get("end") is not None else None,
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


def normalize_single_utterance(item: dict[str, Any], index: int) -> tuple[AlignedUtterance, dict[str, int]]:
    source_utterance_id = str(item.get("utterance_id") or f"utt_{index:06d}")

    start = round_ts(float(item["start"]))
    end = round_ts(float(item["end"]))
    if start is None or end is None:
        raise RuntimeError(f"Utterance {source_utterance_id} is missing start/end.")

    if end <= start:
        raise RuntimeError(f"Utterance {source_utterance_id} has invalid timestamps.")

    speaker_raw = item.get("speaker")
    speaker = None if speaker_raw in ("", None) else str(speaker_raw)

    flags = list(item.get("flags") or [])
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
        },
    )

    report = {
        "text_rebuilt_from_words": text_rebuilt_from_words,
        **word_report,
    }
    return utt, report


def normalize_utterances(
    utterance_dicts: list[dict[str, Any]],
    merge_gap_seconds: float,
    max_utterance_seconds: float,
    max_utterance_chars: int,
) -> tuple[list[AlignedUtterance], dict[str, Any]]:
    del merge_gap_seconds
    del max_utterance_seconds
    del max_utterance_chars

    normalized: list[AlignedUtterance] = []

    empty_utterances_removed = 0
    text_rebuilt_from_words = 0
    words_without_speaker_fixed = 0
    word_speaker_conflicts_fixed = 0

    for idx, item in enumerate(utterance_dicts, start=1):
        if not isinstance(item, dict):
            continue

        utt, report = normalize_single_utterance(item, idx)

        if not utt.text and not utt.words:
            empty_utterances_removed += 1
            continue

        text_rebuilt_from_words += report["text_rebuilt_from_words"]
        words_without_speaker_fixed += report["words_without_speaker_fixed"]
        word_speaker_conflicts_fixed += report["word_speaker_conflicts_fixed"]

        normalized.append(utt)

    normalized.sort(key=lambda x: (x.start, x.end, x.utterance_id))

    report = {
        "input_utterances": len(utterance_dicts),
        "output_utterances": len(normalized),
        "merged_same_speaker_utterances": 0,
        "empty_utterances_removed": empty_utterances_removed,
        "text_rebuilt_from_words": text_rebuilt_from_words,
        "words_without_speaker_fixed": words_without_speaker_fixed,
        "word_speaker_conflicts_fixed": word_speaker_conflicts_fixed,
    }

    return normalized, report
