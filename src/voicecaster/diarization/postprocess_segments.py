from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import SPEAKER_LABEL_PADDING, SPEAKER_LABEL_PREFIX
from .models import SpeakerSegment


def postprocess_speaker_segments(
    speaker_segments: list[SpeakerSegment],
    *,
    min_speaker_ratio: float,
    min_speaker_seconds: float,
    merge_gap_seconds: float,
    max_bridge_seconds: float,
    max_aba_window_seconds: float,
    drop_microsegments_seconds: float,
) -> tuple[list[SpeakerSegment], dict[str, Any]]:
    """
    Post-process normalized speaker segments.

    Goals:
    - merge neighboring same-speaker segments
    - smooth suspicious A-B-A alternations
    - prune residual tiny speakers
    - renumber speakers canonically after pruning

    Returns:
        processed_segments, postprocess_report
    """
    report: dict[str, Any] = {
        "steps": [],
        "counts_before": _build_counts(speaker_segments),
    }

    segments = _copy_segments(speaker_segments)

    segments, merge_report = _merge_same_speaker_neighbors(
        segments,
        merge_gap_seconds=merge_gap_seconds,
    )
    report["steps"].append({"merge_same_speaker_neighbors": merge_report})

    segments, aba_report = _smooth_aba_patterns(
        segments,
        max_bridge_seconds=max_bridge_seconds,
        max_aba_window_seconds=max_aba_window_seconds,
    )
    report["steps"].append({"smooth_aba_patterns": aba_report})

    segments, micro_report = _drop_microsegments(
        segments,
        drop_microsegments_seconds=drop_microsegments_seconds,
    )
    report["steps"].append({"drop_microsegments": micro_report})

    segments, prune_report = _prune_tiny_speakers(
        segments,
        min_speaker_ratio=min_speaker_ratio,
        min_speaker_seconds=min_speaker_seconds,
    )
    report["steps"].append({"prune_tiny_speakers": prune_report})

    segments, second_merge_report = _merge_same_speaker_neighbors(
        segments,
        merge_gap_seconds=merge_gap_seconds,
    )
    report["steps"].append({"merge_after_pruning": second_merge_report})

    segments, relabel_report = _relabel_speakers_by_first_seen(segments)
    report["steps"].append({"relabel_speakers": relabel_report})

    segments = _rebuild_segment_ids(segments)

    report["counts_after"] = _build_counts(segments)
    return segments, report


def _copy_segments(segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
    return [replace(seg, flags=list(seg.flags)) for seg in segments]


def _merge_same_speaker_neighbors(
    segments: list[SpeakerSegment],
    *,
    merge_gap_seconds: float,
) -> tuple[list[SpeakerSegment], dict[str, Any]]:
    if not segments:
        return [], {"merged_pairs": 0}

    merged: list[SpeakerSegment] = []
    merged_pairs = 0

    for seg in sorted(segments, key=lambda s: (s.start, s.end, s.speaker)):
        if not merged:
            merged.append(replace(seg))
            continue

        last = merged[-1]
        gap = seg.start - last.end

        if seg.speaker == last.speaker and gap <= merge_gap_seconds:
            merged_pairs += 1
            merged[-1] = SpeakerSegment(
                segment_id=last.segment_id,
                start=last.start,
                end=max(last.end, seg.end),
                duration=round(max(last.end, seg.end) - last.start, 3),
                speaker=last.speaker,
                speaker_confidence=_mean_optional(last.speaker_confidence, seg.speaker_confidence),
                source=last.source,
                flags=sorted(set(last.flags + seg.flags + ["merged_neighbor_segments"])),
            )
        else:
            merged.append(replace(seg))

    return merged, {"merged_pairs": merged_pairs}


def _smooth_aba_patterns(
    segments: list[SpeakerSegment],
    *,
    max_bridge_seconds: float,
    max_aba_window_seconds: float,
) -> tuple[list[SpeakerSegment], dict[str, Any]]:
    if len(segments) < 3:
        return segments, {"smoothed_triplets": 0}

    items = [replace(seg, flags=list(seg.flags)) for seg in segments]
    smoothed_triplets = 0
    i = 1

    while i < len(items) - 1:
        prev_seg = items[i - 1]
        curr_seg = items[i]
        next_seg = items[i + 1]

        total_window = next_seg.end - prev_seg.start
        is_aba = prev_seg.speaker == next_seg.speaker and curr_seg.speaker != prev_seg.speaker
        short_bridge = curr_seg.duration <= max_bridge_seconds
        compact_window = total_window <= max_aba_window_seconds

        if is_aba and short_bridge and compact_window:
            smoothed_triplets += 1

            merged_seg = SpeakerSegment(
                segment_id=prev_seg.segment_id,
                start=prev_seg.start,
                end=next_seg.end,
                duration=round(next_seg.end - prev_seg.start, 3),
                speaker=prev_seg.speaker,
                speaker_confidence=_mean_optional(
                    prev_seg.speaker_confidence,
                    next_seg.speaker_confidence,
                ),
                source=prev_seg.source,
                flags=sorted(
                    set(
                        prev_seg.flags
                        + curr_seg.flags
                        + next_seg.flags
                        + ["smoothed_aba_pattern"]
                    )
                ),
            )

            items[i - 1 : i + 2] = [merged_seg]
            i = max(1, i - 1)
            continue

        i += 1

    return items, {"smoothed_triplets": smoothed_triplets}


def _drop_microsegments(
    segments: list[SpeakerSegment],
    *,
    drop_microsegments_seconds: float,
) -> tuple[list[SpeakerSegment], dict[str, Any]]:
    if not segments:
        return [], {"dropped_microsegments": 0}

    kept: list[SpeakerSegment] = []
    dropped = 0

    for seg in segments:
        if seg.duration < drop_microsegments_seconds:
            dropped += 1
            continue
        kept.append(seg)

    return kept, {"dropped_microsegments": dropped}


def _prune_tiny_speakers(
    segments: list[SpeakerSegment],
    *,
    min_speaker_ratio: float,
    min_speaker_seconds: float,
) -> tuple[list[SpeakerSegment], dict[str, Any]]:
    if not segments:
        return [], {
            "pruned_speakers": [],
            "speaker_seconds_before": {},
            "total_seconds": 0.0,
        }

    total_seconds = sum(seg.duration for seg in segments)
    seconds_by_speaker: dict[str, float] = {}
    for seg in segments:
        seconds_by_speaker.setdefault(seg.speaker, 0.0)
        seconds_by_speaker[seg.speaker] += seg.duration

    pruned_speakers = []
    for speaker, seconds in seconds_by_speaker.items():
        ratio = seconds / total_seconds if total_seconds > 0 else 0.0
        if seconds < min_speaker_seconds or ratio < min_speaker_ratio:
            pruned_speakers.append(speaker)

    if not pruned_speakers:
        return segments, {
            "pruned_speakers": [],
            "speaker_seconds_before": {k: round(v, 3) for k, v in seconds_by_speaker.items()},
            "total_seconds": round(total_seconds, 3),
        }

    kept = [seg for seg in segments if seg.speaker not in pruned_speakers]

    return kept, {
        "pruned_speakers": sorted(pruned_speakers),
        "speaker_seconds_before": {k: round(v, 3) for k, v in seconds_by_speaker.items()},
        "total_seconds": round(total_seconds, 3),
    }


def _relabel_speakers_by_first_seen(
    segments: list[SpeakerSegment],
) -> tuple[list[SpeakerSegment], dict[str, Any]]:
    if not segments:
        return [], {"mapping": {}}

    first_seen: dict[str, float] = {}
    for seg in segments:
        if seg.speaker not in first_seen:
            first_seen[seg.speaker] = seg.start

    ordered = sorted(first_seen.items(), key=lambda item: item[1])
    mapping = {
        old: f"{SPEAKER_LABEL_PREFIX}{idx:0{SPEAKER_LABEL_PADDING}d}"
        for idx, (old, _) in enumerate(ordered, start=1)
    }

    relabeled = [
        SpeakerSegment(
            segment_id=seg.segment_id,
            start=seg.start,
            end=seg.end,
            duration=seg.duration,
            speaker=mapping[seg.speaker],
            speaker_confidence=seg.speaker_confidence,
            source=seg.source,
            flags=list(seg.flags),
        )
        for seg in segments
    ]

    return relabeled, {"mapping": mapping}


def _rebuild_segment_ids(segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
    rebuilt: list[SpeakerSegment] = []

    for idx, seg in enumerate(sorted(segments, key=lambda s: (s.start, s.end, s.speaker)), start=1):
        rebuilt.append(
            SpeakerSegment(
                segment_id=f"spkseg_{idx:06d}",
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                duration=round(seg.end - seg.start, 3),
                speaker=seg.speaker,
                speaker_confidence=seg.speaker_confidence,
                source=seg.source,
                flags=sorted(set(seg.flags)),
            )
        )

    return rebuilt


def _build_counts(segments: list[SpeakerSegment]) -> dict[str, Any]:
    total_seconds = round(sum(seg.duration for seg in segments), 3)
    speakers = sorted({seg.speaker for seg in segments})

    seconds_by_speaker: dict[str, float] = {}
    for seg in segments:
        seconds_by_speaker.setdefault(seg.speaker, 0.0)
        seconds_by_speaker[seg.speaker] += seg.duration

    return {
        "num_segments": len(segments),
        "num_speakers": len(speakers),
        "total_seconds": total_seconds,
        "speaker_seconds": {k: round(v, 3) for k, v in seconds_by_speaker.items()},
    }


def _mean_optional(a: float | None, b: float | None) -> float | None:
    vals = [v for v in (a, b) if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)
