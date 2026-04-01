# src/voicecaster/diarization/reconcile_with_transcript.py

from __future__ import annotations

from typing import Any

from .models import SpeakerSegment, TranscriptUtterance


def assign_speakers_to_transcript(
    transcript_preview: dict[str, Any],
    speaker_segments: list[SpeakerSegment],
    low_confidence_threshold: float,
) -> tuple[list[TranscriptUtterance], dict[str, Any]]:
    """
    Assign dominant speaker to each ASR utterance using temporal overlap.
    """
    utterance_dicts = _extract_utterance_dicts(transcript_preview)
    utterances: list[TranscriptUtterance] = []

    assigned_count = 0
    low_confidence_count = 0

    for idx, item in enumerate(utterance_dicts, start=1):
        start = float(item["start"])
        end = float(item["end"])
        text = str(item.get("text", "")).strip()
        duration = max(0.0, end - start)

        overlap_rows = []
        for seg in speaker_segments:
            overlap_seconds = _compute_overlap(start, end, seg.start, seg.end)
            if overlap_seconds > 0:
                overlap_rows.append(
                    {
                        "speaker": seg.speaker,
                        "overlap_seconds": overlap_seconds,
                    }
                )

        overlap_rows.sort(key=lambda x: x["overlap_seconds"], reverse=True)

        flags: list[str] = []
        assigned_speaker: str | None = None
        speaker_confidence: float | None = None
        overlap_stats: dict[str, Any] = {
            "best_speaker_overlap_seconds": 0.0,
            "coverage_ratio": 0.0,
            "candidate_speakers": overlap_rows,
        }

        if overlap_rows and duration > 0:
            best = overlap_rows[0]
            assigned_speaker = best["speaker"]
            coverage_ratio = best["overlap_seconds"] / duration
            speaker_confidence = round(min(1.0, coverage_ratio), 4)

            overlap_stats["best_speaker_overlap_seconds"] = round(best["overlap_seconds"], 4)
            overlap_stats["coverage_ratio"] = round(coverage_ratio, 4)

            assigned_count += 1

            if speaker_confidence < low_confidence_threshold:
                flags.append("low_confidence_assignment")
                low_confidence_count += 1
        else:
            flags.append("no_speaker_overlap")

        words = _normalize_words(item.get("words", []), assigned_speaker)

        utterances.append(
            TranscriptUtterance(
                utterance_id=f"utt_{idx:06d}",
                start=round(start, 3),
                end=round(end, 3),
                duration=round(duration, 3),
                text=text,
                speaker=assigned_speaker,
                speaker_confidence=speaker_confidence,
                assignment_source="diarization_overlap_assignment" if assigned_speaker else None,
                overlap_stats=overlap_stats,
                flags=flags,
                words=words,
            )
        )

    total_utterances = len(utterances)
    assignment_ratio = (assigned_count / total_utterances) if total_utterances else 0.0

    stats = {
        "total_utterances": total_utterances,
        "assigned_utterances": assigned_count,
        "assignment_ratio": round(assignment_ratio, 4),
        "low_confidence_utterances": low_confidence_count,
    }

    return utterances, stats


def _extract_utterance_dicts(transcript_preview: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(transcript_preview.get("segments"), list):
        return transcript_preview["segments"]

    if isinstance(transcript_preview.get("utterances"), list):
        return transcript_preview["utterances"]

    raise ValueError("transcript_preview does not contain 'segments' or 'utterances' list.")


def _compute_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _normalize_words(words: list[dict[str, Any]], speaker: str | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in words:
        word = dict(item)
        if speaker is not None and "speaker" not in word:
            word["speaker"] = speaker
        normalized.append(word)
    return normalized
