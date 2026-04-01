from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .models import RawSpeakerSegment, SpeakerSegment


def _is_valid_raw_segment(segment: RawSpeakerSegment) -> bool:
    return (
        segment.start >= 0.0
        and segment.end > segment.start
        and segment.duration > 0.0
        and bool(segment.speaker_raw.strip())
    )


def _sort_raw_segments(
    raw_segments: Iterable[RawSpeakerSegment],
) -> list[RawSpeakerSegment]:
    return sorted(
        raw_segments,
        key=lambda s: (s.start, s.end, s.speaker_raw),
    )


def _merge_consecutive_same_speaker(
    segments: list[RawSpeakerSegment],
    merge_gap_seconds: float,
) -> tuple[list[RawSpeakerSegment], int]:
    """
    Fusiona segmentos consecutivos del mismo speaker_raw cuando la pausa
    entre ellos es pequeña o nula.
    """
    if not segments:
        return [], 0

    merged: list[RawSpeakerSegment] = []
    merge_count = 0

    current = RawSpeakerSegment(
        start=segments[0].start,
        end=segments[0].end,
        speaker_raw=segments[0].speaker_raw,
        confidence=segments[0].confidence,
        engine=segments[0].engine,
        extra=dict(segments[0].extra),
    )

    for seg in segments[1:]:
        same_speaker = seg.speaker_raw == current.speaker_raw
        gap = seg.start - current.end

        if same_speaker and gap <= merge_gap_seconds:
            current.end = max(current.end, seg.end)
            merge_count += 1
            continue

        merged.append(current)
        current = RawSpeakerSegment(
            start=seg.start,
            end=seg.end,
            speaker_raw=seg.speaker_raw,
            confidence=seg.confidence,
            engine=seg.engine,
            extra=dict(seg.extra),
        )

    merged.append(current)
    return merged, merge_count


def _filter_too_short_segments(
    segments: list[RawSpeakerSegment],
    min_segment_seconds: float,
) -> tuple[list[RawSpeakerSegment], int]:
    """
    Elimina segmentos demasiado cortos para ser útiles en la verdad canónica
    de diarización.
    """
    filtered = [seg for seg in segments if seg.duration >= min_segment_seconds]
    removed_count = len(segments) - len(filtered)
    return filtered, removed_count


def _build_canonical_label_mapping(
    segments: list[RawSpeakerSegment],
) -> dict[str, str]:
    """
    Crea mapping speaker_raw -> speaker_XX según primer orden de aparición.
    """
    seen: dict[str, str] = {}
    counter = 1

    for seg in segments:
        if seg.speaker_raw not in seen:
            seen[seg.speaker_raw] = f"speaker_{counter:02d}"
            counter += 1

    return seen


def _compute_confidence_for_group(
    speaker_raw: str,
    raw_segments: list[RawSpeakerSegment],
) -> float | None:
    """
    Si existen confidences numéricas en los segmentos crudos de ese speaker,
    calcula una media simple. Si no, devuelve None.
    """
    values = [
        seg.confidence
        for seg in raw_segments
        if seg.speaker_raw == speaker_raw and seg.confidence is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def normalize_speaker_segments(
    raw_segments: list[RawSpeakerSegment],
    min_segment_seconds: float,
    merge_gap_seconds: float,
) -> tuple[list[SpeakerSegment], dict[str, str], list[str]]:
    """
    Convierte la salida cruda del motor en la timeline canónica del sistema.

    Devuelve:
      1. lista de SpeakerSegment ya normalizada
      2. mapping speaker_raw -> speaker_XX
      3. lista de warnings globales
    """
    warnings: list[str] = []

    if not raw_segments:
        return [], {}, ["no_raw_segments"]

    invalid_count = sum(1 for seg in raw_segments if not _is_valid_raw_segment(seg))
    valid_segments = [seg for seg in raw_segments if _is_valid_raw_segment(seg)]

    if invalid_count > 0:
        warnings.append(f"invalid_raw_segments_removed:{invalid_count}")

    if not valid_segments:
        return [], {}, warnings + ["no_valid_raw_segments"]

    ordered = _sort_raw_segments(valid_segments)

    merged, merge_count = _merge_consecutive_same_speaker(
        ordered,
        merge_gap_seconds=merge_gap_seconds,
    )
    if merge_count > 0:
        warnings.append(f"consecutive_segments_merged:{merge_count}")

    filtered, short_removed_count = _filter_too_short_segments(
        merged,
        min_segment_seconds=min_segment_seconds,
    )
    if short_removed_count > 0:
        warnings.append(f"short_segments_removed:{short_removed_count}")

    if not filtered:
        return [], {}, warnings + ["no_segments_after_short_filter"]

    label_mapping = _build_canonical_label_mapping(filtered)

    # confidencia media por speaker_raw
    grouped: dict[str, list[RawSpeakerSegment]] = defaultdict(list)
    for seg in filtered:
        grouped[seg.speaker_raw].append(seg)

    normalized_segments: list[SpeakerSegment] = []

    for idx, seg in enumerate(filtered, start=1):
        canonical_speaker = label_mapping[seg.speaker_raw]
        confidence = _compute_confidence_for_group(seg.speaker_raw, grouped[seg.speaker_raw])

        flags: list[str] = []
        if seg.duration < (min_segment_seconds * 1.5):
            flags.append("short_segment")

        normalized_segments.append(
            SpeakerSegment(
                segment_id=f"spkseg_{idx:06d}",
                start=seg.start,
                end=seg.end,
                duration=round(seg.duration, 3),
                speaker=canonical_speaker,
                speaker_confidence=confidence,
                source="normalized_diarization",
                flags=flags,
            )
        )

    # comprobación final de orden temporal
    normalized_segments.sort(key=lambda s: (s.start, s.end, s.speaker))
    for i in range(1, len(normalized_segments)):
        prev = normalized_segments[i - 1]
        curr = normalized_segments[i]
        if curr.start < prev.start:
            warnings.append("timeline_order_anomaly_detected")
            break

    return normalized_segments, label_mapping, warnings
