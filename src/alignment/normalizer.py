from __future__ import annotations

from .schemas import SpeakerSegment, TranscriptSegment, TranscriptWord


def _safe_float(value: object, *, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for '{field_name}': {value!r}") from exc


def _normalize_text(text: object) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split()).strip()


def normalize_transcript_segments(payload: dict) -> list[TranscriptSegment]:
    normalized: list[TranscriptSegment] = []

    for i, raw_segment in enumerate(payload.get("segments", [])):
        start = _safe_float(raw_segment.get("start"), field_name=f"segments[{i}].start")
        end = _safe_float(raw_segment.get("end"), field_name=f"segments[{i}].end")
        text = _normalize_text(raw_segment.get("text", ""))

        if end < start:
            continue

        raw_words = raw_segment.get("words") or []
        words: list[TranscriptWord] = []

        for j, raw_word in enumerate(raw_words):
            if not isinstance(raw_word, dict):
                continue

            word_text = _normalize_text(raw_word.get("word", ""))
            word_start = _safe_float(
                raw_word.get("start"),
                field_name=f"segments[{i}].words[{j}].start",
            )
            word_end = _safe_float(
                raw_word.get("end"),
                field_name=f"segments[{i}].words[{j}].end",
            )

            if word_end < word_start:
                continue

            probability_raw = raw_word.get("probability")
            probability = (
                None
                if probability_raw is None
                else _safe_float(
                    probability_raw,
                    field_name=f"segments[{i}].words[{j}].probability",
                )
            )

            words.append(
                TranscriptWord(
                    source_segment_id=i,
                    index_in_segment=j,
                    word=word_text,
                    start=word_start,
                    end=word_end,
                    probability=probability,
                )
            )

        if not text and not words:
            continue

        normalized.append(
            TranscriptSegment(
                segment_id=i,
                start=start,
                end=end,
                text=text,
                words=words,
            )
        )

    normalized.sort(key=lambda item: (item.start, item.end, item.segment_id))
    return normalized


def normalize_speaker_segments(payload: dict) -> list[SpeakerSegment]:
    normalized: list[SpeakerSegment] = []

    for i, raw_segment in enumerate(payload.get("segments", [])):
        start = _safe_float(raw_segment.get("start"), field_name=f"segments[{i}].start")
        end = _safe_float(raw_segment.get("end"), field_name=f"segments[{i}].end")
        speaker = _normalize_text(raw_segment.get("speaker"))

        if not speaker:
            speaker = "UNKNOWN"

        if end < start:
            continue

        confidence_raw = raw_segment.get("confidence")
        confidence = (
            None
            if confidence_raw is None
            else _safe_float(confidence_raw, field_name=f"segments[{i}].confidence")
        )

        normalized.append(
            SpeakerSegment(
                segment_id=i,
                start=start,
                end=end,
                speaker=speaker,
                confidence=confidence,
            )
        )

    normalized.sort(key=lambda item: (item.start, item.end, item.segment_id))
    return normalized


def validate_monotonic_timeline(
    transcript_segments: list[TranscriptSegment],
    speaker_segments: list[SpeakerSegment],
) -> None:
    if not transcript_segments:
        raise ValueError("Normalized transcript segments are empty")

    if not speaker_segments:
        raise ValueError("Normalized speaker segments are empty")

    for segment in transcript_segments:
        if segment.start < 0 or segment.end < 0:
            raise ValueError("Transcript contains negative timestamps")
        if segment.end < segment.start:
            raise ValueError("Transcript contains inverted segment timestamps")

        last_word_end = None
        for word in segment.words:
            if word.start < 0 or word.end < 0:
                raise ValueError("Transcript contains negative word timestamps")
            if word.end < word.start:
                raise ValueError("Transcript contains inverted word timestamps")
            if last_word_end is not None and word.start < last_word_end:
                # Permit a small amount of overlap noise later if needed.
                pass
            last_word_end = word.end

    for segment in speaker_segments:
        if segment.start < 0 or segment.end < 0:
            raise ValueError("Speaker segments contain negative timestamps")
        if segment.end < segment.start:
            raise ValueError("Speaker segments contain inverted timestamps")
