# =========================================
# FILE: src/voicecaster/alignment/normalizer.py
# =========================================

from __future__ import annotations

from .schemas import SpeakerSegment, TranscriptSegment, TranscriptWord


class AlignmentNormalizationError(RuntimeError):
    """Raised when raw alignment inputs cannot be normalized safely."""


def _coerce_float(value: object, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AlignmentNormalizationError(f"Invalid float for field '{field_name}': {value!r}") from exc


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_transcript_segments(transcript_raw: dict) -> list[TranscriptSegment]:
    """
    Convert raw transcript JSON into normalized internal transcript segments.
    """
    normalized: list[TranscriptSegment] = []

    for idx, raw_segment in enumerate(transcript_raw.get("segments", [])):
        if not isinstance(raw_segment, dict):
            raise AlignmentNormalizationError(f"Transcript segment at index {idx} is not an object")

        start = _coerce_float(raw_segment.get("start", 0.0), "start")
        end = _coerce_float(raw_segment.get("end", 0.0), "end")
        text = _normalize_text(raw_segment.get("text", ""))

        if end < start:
            raise AlignmentNormalizationError(
                f"Transcript segment {idx} has invalid interval: start={start}, end={end}"
            )

        raw_words = raw_segment.get("words", []) or []
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

            probability_raw = raw_word.get("probability")
            probability = None if probability_raw is None else float(probability_raw)

            words.append(
                TranscriptWord(
                    source_segment_id=idx,
                    index_in_segment=word_idx,
                    word=_normalize_text(raw_word.get("word", "")),
                    start=word_start,
                    end=word_end,
                    probability=probability,
                )
            )

        # Keep useful segments even if text is empty but words exist.
        if text or words:
            normalized.append(
                TranscriptSegment(
                    segment_id=idx,
                    start=start,
                    end=end,
                    text=text,
                    words=words,
                )
            )

    normalized.sort(key=lambda seg: (seg.start, seg.end, seg.segment_id))
    validate_monotonic_timeline(normalized, kind="transcript")
    return normalized


def normalize_speaker_segments(speakers_raw: dict) -> list[SpeakerSegment]:
    """
    Convert raw speaker JSON into normalized internal speaker segments.
    """
    normalized: list[SpeakerSegment] = []

    for idx, raw_segment in enumerate(speakers_raw.get("segments", [])):
        if not isinstance(raw_segment, dict):
            raise AlignmentNormalizationError(f"Speaker segment at index {idx} is not an object")

        start = _coerce_float(raw_segment.get("start", 0.0), "start")
        end = _coerce_float(raw_segment.get("end", 0.0), "end")
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


def validate_monotonic_timeline(segments: list[TranscriptSegment] | list[SpeakerSegment], kind: str) -> None:
    """
    Validate sorted monotonic timeline.
    Overlaps are allowed because diarization and transcript may have adjacent/complex boundaries.
    This function only ensures no impossible negative intervals remain after normalization.
    """
    for idx, segment in enumerate(segments):
        if segment.start < 0.0:
            raise AlignmentNormalizationError(f"{kind} segment {idx} has negative start time")
        if segment.end < segment.start:
            raise AlignmentNormalizationError(f"{kind} segment {idx} has end < start")
