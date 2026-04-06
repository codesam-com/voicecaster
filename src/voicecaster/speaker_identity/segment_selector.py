from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class SelectedSegment:
    speaker: str
    source: str
    start: float
    end: float
    duration: float
    text: str
    num_words: int
    selection_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_num_words(item: dict[str, Any]) -> int:
    words = item.get("words")
    if isinstance(words, list):
        return len(words)

    text = _safe_text(item.get("text"))
    if not text:
        return 0
    return len(text.split())


def select_segments_for_speaker(
    speaker: str,
    aligned_utterances: list[dict[str, Any]],
    *,
    min_segment_seconds: float,
    min_num_words: int,
    max_segments: int,
) -> list[SelectedSegment]:
    candidates: list[SelectedSegment] = []

    for item in aligned_utterances:
        item_speaker = _safe_text(item.get("speaker"))
        if item_speaker != speaker:
            continue

        start = _safe_float(item.get("start"))
        end = _safe_float(item.get("end"))
        duration = max(0.0, end - start)
        text = _safe_text(item.get("text"))
        num_words = _safe_num_words(item)

        if duration < min_segment_seconds:
            continue
        if num_words < min_num_words:
            continue
        if not text:
            continue

        candidates.append(
            SelectedSegment(
                speaker=speaker,
                source="aligned_utterances",
                start=start,
                end=end,
                duration=round(duration, 3),
                text=text,
                num_words=num_words,
                selection_reason="passed_basic_filters",
            )
        )

    if not candidates:
        return []

    candidates.sort(key=lambda x: x.start)

    if len(candidates) <= max_segments:
        return candidates

    selected: list[SelectedSegment] = []
    selected.append(candidates[0])

    middle_count = max(0, max_segments - 2)
    if middle_count > 0:
        step = len(candidates) / (middle_count + 1)
        for i in range(1, middle_count + 1):
            idx = int(round(i * step))
            idx = max(1, min(len(candidates) - 2, idx))
            selected.append(candidates[idx])

    if max_segments > 1:
        selected.append(candidates[-1])

    dedup: list[SelectedSegment] = []
    seen: set[tuple[float, float]] = set()
    for seg in selected:
        key = (seg.start, seg.end)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(seg)

    if len(dedup) < max_segments:
        remaining = [
            seg for seg in candidates
            if (seg.start, seg.end) not in {(x.start, x.end) for x in dedup}
        ]
        remaining.sort(key=lambda x: (-x.duration, x.start))

        for seg in remaining:
            if len(dedup) >= max_segments:
                break
            dedup.append(seg)

    dedup.sort(key=lambda x: x.start)
    return dedup[:max_segments]
