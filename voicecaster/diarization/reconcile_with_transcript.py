from __future__ import annotations

from typing import Any

from .models import SpeakerSegment, TranscriptUtterance, TranscriptWord


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_text(segment: dict[str, Any]) -> str:
    text = segment.get("text", "")
    if text is None:
        return ""
    return str(text).strip()


def _normalize_words(raw_words: list[dict[str, Any]] | None) -> list[TranscriptWord]:
    if not raw_words:
        return []

    words: list[TranscriptWord] = []

    for item in raw_words:
        if not isinstance(item, dict):
            continue

        word = str(item.get("word", "")).strip()
        if not word:
            continue

        words.append(
            TranscriptWord(
                word=word,
                start=_safe_float(item.get("start")),
                end=_safe_float(item.get("end")),
                confidence=_safe_float(item.get("probability")),
                speaker=None,
                flags=[],
            )
        )

    return words


def _extract_transcript_segments(transcript_preview: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Intenta encontrar la lista de segmentos en transcript_preview.json
    sin acoplarse demasiado a una única forma exacta.
    """
    candidates = [
        transcript_preview.get("segments"),
        transcript_preview.get("utterances"),
        transcript_preview.get("items"),
    ]

    for candidate in candidates:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]

    return []


def _overlap_seconds(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _candidate_overlaps(
    utt_start: float,
    utt_end: float,
    speaker_segments: list[SpeakerSegment],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for seg in speaker_segments:
        overlap = _overlap_seconds(utt_start, utt_end, seg.start, seg.end)
        if overlap <= 0.0:
            continue

        candidates.append(
            {
                "speaker": seg.speaker,
                "segment_id": seg.segment_id,
                "overlap_seconds": round(overlap, 3),
                "segment_start": seg.start,
                "segment_end": seg.end,
            }
        )

    candidates.sort(
        key=lambda x: (-x["overlap_seconds"], x["segment_start"], x["speaker"])
    )
    return candidates


def _best_speaker_assignment(
    utt_start: float,
    utt_end: float,
    speaker_segments: list[SpeakerSegment],
    low_confidence_threshold: float,
) -> tuple[str | None, float | None, dict[str, Any], list[str]]:
    duration = max(0.0, utt_end - utt_start)
    flags: list[str] = []

    if duration <= 0.0:
        return None, None, {"coverage_ratio": 0.0, "candidate_speakers": []}, ["invalid_utterance_timestamps"]

    candidates = _candidate_overlaps(utt_start, utt_end, speaker_segments)

    if not candidates:
        return None, 0.0, {"coverage_ratio": 0.0, "candidate_speakers": []}, ["no_speaker_overlap"]

    best = candidates[0]
    best_overlap = float(best["overlap_seconds"])
    coverage_ratio = round(best_overlap / duration, 4)

    if coverage_ratio < low_confidence_threshold:
        flags.append("low_confidence_assignment")

    if len(candidates) > 1:
        second_overlap = float(candidates[1]["overlap_seconds"])
        if best_overlap > 0 and (second_overlap / best_overlap) >= 0.8:
            flags.append("overlap_conflict")

    overlap_stats = {
        "best_speaker_overlap_seconds": round(best_overlap, 3),
        "coverage_ratio": coverage_ratio,
        "candidate_speakers": [
            {
                "speaker": item["speaker"],
                "overlap_seconds": item["overlap_seconds"],
                "segment_id": item["segment_id"],
            }
            for item in candidates
        ],
    }

    return best["speaker"], coverage_ratio, overlap_stats, flags


def _assign_speaker_to_words(
    words: list[TranscriptWord],
    speaker_segments: list[SpeakerSegment],
    fallback_speaker: str | None,
) -> list[TranscriptWord]:
    if not words:
        return words

    assigned: list[TranscriptWord] = []

    for word in words:
        if word.start is None or word.end is None or word.end <= word.start:
            word.speaker = fallback_speaker
            word.flags.append("fallback_speaker_assignment")
            assigned.append(word)
            continue

        candidates = _candidate_overlaps(word.start, word.end, speaker_segments)
        if not candidates:
            word.speaker = fallback_speaker
            word.flags.append("fallback_speaker_assignment")
            assigned.append(word)
            continue

        word.speaker = candidates[0]["speaker"]
        assigned.append(word)

    return assigned


def assign_speakers_to_transcript(
    transcript_preview: dict[str, Any],
    speaker_segments: list[SpeakerSegment],
    low_confidence_threshold: float,
) -> tuple[list[TranscriptUtterance], dict[str, Any]]:
    """
    Asigna speaker a cada bloque del transcript_preview basándose en solape temporal.

    Devuelve:
      1. lista de TranscriptUtterance
      2. estadísticas globales de reconciliación
    """
    raw_segments = _extract_transcript_segments(transcript_preview)

    utterances: list[TranscriptUtterance] = []
    total_segments = 0
    assigned_segments = 0
    low_conf_segments = 0
    no_overlap_segments = 0

    for idx, item in enumerate(raw_segments, start=1):
        start = _safe_float(item.get("start"))
        end = _safe_float(item.get("end"))
        text = _extract_text(item)

        if start is None or end is None or end <= start:
            continue

        total_segments += 1
        duration = round(end - start, 3)

        words = _normalize_words(item.get("words"))
        speaker, speaker_confidence, overlap_stats, flags = _best_speaker_assignment(
            utt_start=start,
            utt_end=end,
            speaker_segments=speaker_segments,
            low_confidence_threshold=low_confidence_threshold,
        )

        if speaker is not None:
            assigned_segments += 1
        else:
            no_overlap_segments += 1

        if "low_confidence_assignment" in flags:
            low_conf_segments += 1

        words = _assign_speaker_to_words(
            words=words,
            speaker_segments=speaker_segments,
            fallback_speaker=speaker,
        )

        utterances.append(
            TranscriptUtterance(
                utterance_id=f"utt_{idx:06d}",
                start=round(start, 3),
                end=round(end, 3),
                duration=duration,
                text=text,
                speaker=speaker,
                speaker_confidence=speaker_confidence,
                assignment_source="diarization_overlap_assignment" if speaker else None,
                overlap_stats=overlap_stats,
                flags=flags,
                words=words,
            )
        )

    transcript_assignment_ratio = round(
        (assigned_segments / total_segments), 4
    ) if total_segments > 0 else 0.0

    stats = {
        "total_transcript_segments": total_segments,
        "assigned_transcript_segments": assigned_segments,
        "unassigned_transcript_segments": no_overlap_segments,
        "low_confidence_segments": low_conf_segments,
        "transcript_assignment_ratio": transcript_assignment_ratio,
    }

    return utterances, stats
