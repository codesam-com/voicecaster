# src/voicecaster/diarization/normalize_segments.py

from __future__ import annotations

from collections import OrderedDict

from .config import SPEAKER_LABEL_PADDING, SPEAKER_LABEL_PREFIX
from .models import RawSpeakerSegment, SpeakerSegment


def _make_speaker_label(index: int) -> str:
    return f"{SPEAKER_LABEL_PREFIX}{index:0{SPEAKER_LABEL_PADDING}d}"


def normalize_speaker_segments(
    raw_segments: list[RawSpeakerSegment],
    min_segment_seconds: float,
    merge_gap_seconds: float,
) -> tuple[list[SpeakerSegment], dict[str, str], list[str]]:
    """
    Normalize raw diarization segments into canonical speaker_X segments.

    Returns:
        normalized_segments, raw_to_canonical_mapping, warnings
    """
    warnings: list[str] = []

    valid_raw = [
        seg
        for seg in raw_segments
        if seg.start >= 0.0 and seg.end > seg.start
    ]

    if len(valid_raw) != len(raw_segments):
        warnings.append("invalid_raw_segments_removed")

    valid_raw.sort(key=lambda s: (s.start, s.end, s.speaker_raw))

    if not valid_raw:
        return [], {}, warnings

    first_seen: OrderedDict[str, float] = OrderedDict()
    for seg in valid_raw:
        if seg.speaker_raw not in first_seen:
            first_seen[seg.speaker_raw] = seg.start

    sorted_speakers = sorted(first_seen.items(), key=lambda item: item[1])
    raw_to_canonical = {
        raw_label: _make_speaker_label(idx)
        for idx, (raw_label, _) in enumerate(sorted_speakers, start=1)
    }

    merged: list[tuple[float, float, str, float | None, list[str]]] = []

    for seg in valid_raw:
        canonical_speaker = raw_to_canonical[seg.speaker_raw]
        seg_flags: list[str] = []

        if seg.duration < min_segment_seconds:
            seg_flags.append("short_segment")

        if not merged:
            merged.append((seg.start, seg.end, canonical_speaker, seg.confidence, seg_flags))
            continue

        last_start, last_end, last_speaker, last_conf, last_flags = merged[-1]
        gap = seg.start - last_end

        if canonical_speaker == last_speaker and gap <= merge_gap_seconds:
            merged[-1] = (
                last_start,
                max(last_end, seg.end),
                last_speaker,
                _merge_confidences(last_conf, seg.confidence),
                _merge_flags(last_flags, seg_flags, extra_flag="merged_short_segments" if seg.duration < min_segment_seconds else None),
            )
        else:
            merged.append((seg.start, seg.end, canonical_speaker, seg.confidence, seg_flags))

    normalized_segments: list[SpeakerSegment] = []
    for idx, (start, end, speaker, confidence, flags) in enumerate(merged, start=1):
        normalized_segments.append(
            SpeakerSegment(
                segment_id=f"spkseg_{idx:06d}",
                start=round(start, 3),
                end=round(end, 3),
                duration=round(end - start, 3),
                speaker=speaker,
                speaker_confidence=confidence,
                source="normalized_diarization",
                flags=sorted(set(flags)),
            )
        )

    return normalized_segments, raw_to_canonical, warnings


def _merge_confidences(a: float | None, b: float | None) -> float | None:
    values = [v for v in (a, b) if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _merge_flags(
    flags_a: list[str],
    flags_b: list[str],
    extra_flag: str | None = None,
) -> list[str]:
    merged = list(flags_a) + list(flags_b)
    if extra_flag:
        merged.append(extra_flag)
    return merged
