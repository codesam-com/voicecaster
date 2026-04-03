from __future__ import annotations

from typing import Any

from .schemas import SpeakerSegment, TranscriptSegment, TranscriptWord


class AlignmentNormalizationError(RuntimeError):
    """Raised when raw alignment inputs cannot be normalized safely."""


def _coerce_float(value: object, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AlignmentNormalizationError(
            f"Invalid float for field '{field_name}': {value!r}"
        ) from exc


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_probability(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_transcript_segments(transcript_raw: dict[str, Any]) -> list[TranscriptSegment]:
    """
    Convert raw transcript JSON into normalized internal transcript segments.

    Accepted full form:
        {"segments": [...]}

    Explicitly rejected preview-only form:
        {"first_segments": [...], "last_segments": [...], ...}

    Reason:
    first_segments/last_segments is only a sample view, not the full episode
    timeline, so it cannot be used to build alignment.
    """
    if "segments" not in transcript_raw:
        if "first_segments" in transcript_raw or "last_segments" in transcript_raw:
            raise AlignmentNormalizationError(
                "Cannot normalize transcript_preview.json for alignment because it is "
                "preview-only and does not contain the full 'segments' timeline"
            )
        raise AlignmentNormalizationError(
            "Transcript input missing required key 'segments'"
        )

    raw_segments = transcript_raw.get("segments", [])
    if not isinstance(raw_segments, list):
        raise AlignmentNormalizationError("Transcript input 'segments' must be a list")

    normalized: list[TranscriptSegment] = []

    for idx, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise AlignmentNormalizationError(
                f"Transcript segment at index {idx} is not an object"
            )

        segment_id_raw = raw_segment.get("id", idx)
        try:
            segment_id = int(segment_id_raw)
        except (TypeError, ValueError):
            segment_id = idx

        start = _coerce_float(raw_segment.get("start", 0.0), "segment.start")
        end = _coerce_float(raw_segment.get("end", 0.0), "segment.end")
        text = _normalize_text(raw_segment.get("text", ""))

        if end < start:
            raise AlignmentNormalizationError(
                f"Transcript segment {idx} has invalid interval: start={start}, end={end}"
            )

        raw_words = raw_segment.get("words", []) or []
        if not isinstance(raw_words, list):
            raise AlignmentNormalizationError(
                f"Transcript segment {idx} field 'words' must be a list"
            )

        words: list[TranscriptWord] = []

        for word_idx, raw_word in enumerate(raw_words):
            if not isinstance(raw_word, dict):
                raise AlignmentNormalizationError(
                    f"Transcript word at segment {idx}, index {word_idx} is not an object"
                )

            word_start = _coerce_float(raw_word.get("start", start), "word.start")
            word_end = _coerce_float(raw_word.get("end", word_start), "word.end")

            if word_end < word_start:
                raise AlignmentNormalizationError(
                    f"Transcript word at segment {idx}, index {word_idx} "
                    f"has invalid interval: start={word_start}, end={word_end}"
                )

            word_text = _normalize_text(
                raw_word.get("word", raw_word.get("text", ""))
            )

            words.append(
                TranscriptWord(
                    source_segment_id=segment_id,
                    index_in_segment=word_idx,
                    word=word_text,
                    start=word_start,
                    end=word_end,
                    probability=_coerce_probability(raw_word.get("probability")),
                )
            )

        if text or words:
            normalized.append(
                TranscriptSegment(
                    segment_id=segment_id,
                    start=start,
                    end=end,
                    text=text,
                    words=words,
                )
            )

    normalized.sort(key=lambda seg: (seg.start, seg.end, seg.segment_id))
    validate_monotonic_timeline(normalized, kind="transcript")
    return normalized


def normalize_speaker_segments(speakers_raw: dict[str, Any]) -> list[SpeakerSegment]:
    """
    Convert raw speaker JSON into normalized internal speaker segments.

    Real repo-compatible forms observed:
    - {"segments": [...]}
    - [...]
    """
    raw_segments = speakers_raw.get("segments", [])
    if not isinstance(raw_segments, list):
        raise AlignmentNormalizationError("Speaker input 'segments' must be a list")

    normalized: list[SpeakerSegment] = []

    for idx, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise AlignmentNormalizationError(
                f"Speaker segment at index {idx} is not an object"
            )

        start = _coerce_float(raw_segment.get("start", 0.0), "speaker.start")
        end = _coerce_float(raw_segment.get("end", 0.0), "speaker.end")
        speaker = _normalize_text(raw_segment.get("speaker", "UNKNOWN")) or "UNKNOWN"

        if end < start:
            raise AlignmentNormalizationError(
                f"Speaker segment {idx} has invalid interval: start={start}, end={end}"
            )

        normalized.append(
            SpeakerSegment(
                segment_id=idx,
                start=start,
                end=end,
                speaker=speaker,
            )
        )

    normalized.sort(key=lambda seg: (seg.start, seg.end, seg.segment_id))
    validate_monotonic_timeline(normalized, kind="speaker")
    return normalized


def validate_monotonic_timeline(
    segments: list[TranscriptSegment] | list[SpeakerSegment],
    kind: str,
) -> None:
    """
    Validate sorted timeline.

    Overlaps are allowed.
    We only reject impossible intervals and negative starts.
    """
    for idx, segment in enumerate(segments):
        if segment.start < 0.0:
            raise AlignmentNormalizationError(
                f"{kind} segment {idx} has negative start time"
            )
        if segment.end < segment.start:
            raise AlignmentNormalizationError(
                f"{kind} segment {idx} has end < start"
            )
