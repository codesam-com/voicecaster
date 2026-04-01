# src/voicecaster/diarization/write_outputs.py

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .models import RawSpeakerSegment, SpeakerSegment, TranscriptUtterance


def ensure_diarization_dir(work_episode_dir: Path) -> Path:
    diarization_dir = work_episode_dir / "03_diarization"
    diarization_dir.mkdir(parents=True, exist_ok=True)
    (diarization_dir / "speakers").mkdir(parents=True, exist_ok=True)
    return diarization_dir


def write_diarization_raw_json(
    diarization_dir: Path,
    raw_segments: list[RawSpeakerSegment],
    metadata: dict[str, Any],
) -> Path:
    output_path = diarization_dir / "diarization_raw.json"
    payload = {
        "metadata": metadata,
        "segments": [seg.to_dict() for seg in raw_segments],
    }
    _write_json(output_path, payload)
    return output_path


def write_speaker_segments_json(
    diarization_dir: Path,
    speaker_segments: list[SpeakerSegment],
) -> Path:
    output_path = diarization_dir / "speaker_segments.json"
    _write_json(output_path, [seg.to_dict() for seg in speaker_segments])
    return output_path


def write_transcript_with_speakers_json(
    diarization_dir: Path,
    utterances: list[TranscriptUtterance],
) -> Path:
    output_path = diarization_dir / "transcript_with_speakers.json"
    _write_json(output_path, [utt.to_dict() for utt in utterances])
    return output_path


def write_subtitles_diarized_srt(
    diarization_dir: Path,
    utterances: list[TranscriptUtterance],
) -> Path:
    output_path = diarization_dir / "subtitles_diarized.srt"
    blocks: list[str] = []

    for idx, utt in enumerate(utterances, start=1):
        speaker_label = f"[{utt.speaker}]" if utt.speaker else "[unknown]"
        text = f"{speaker_label} {utt.text}".strip()
        blocks.append(
            "\n".join(
                [
                    str(idx),
                    f"{_format_srt_time(utt.start)} --> {_format_srt_time(utt.end)}",
                    text,
                ]
            )
        )

    output_path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    return output_path


def write_per_speaker_outputs(
    diarization_dir: Path,
    utterances: list[TranscriptUtterance],
) -> list[Path]:
    speakers_dir = diarization_dir / "speakers"
    grouped: dict[str, list[TranscriptUtterance]] = defaultdict(list)

    for utt in utterances:
        speaker = utt.speaker or "unknown"
        grouped[speaker].append(utt)

    written_paths: list[Path] = []

    for speaker, items in sorted(grouped.items()):
        srt_path = speakers_dir / f"{speaker}.srt"
        txt_path = speakers_dir / f"{speaker}.txt"
        json_path = speakers_dir / f"{speaker}.json"

        srt_blocks: list[str] = []
        full_text_parts: list[str] = []

        for idx, utt in enumerate(items, start=1):
            srt_blocks.append(
                "\n".join(
                    [
                        str(idx),
                        f"{_format_srt_time(utt.start)} --> {_format_srt_time(utt.end)}",
                        utt.text,
                    ]
                )
            )
            if utt.text:
                full_text_parts.append(utt.text)

        srt_path.write_text("\n\n".join(srt_blocks) + ("\n" if srt_blocks else ""), encoding="utf-8")
        txt_path.write_text("\n".join(full_text_parts).strip() + ("\n" if full_text_parts else ""), encoding="utf-8")

        speech_seconds = round(sum(utt.duration for utt in items), 3)
        payload = {
            "speaker": speaker,
            "num_utterances": len(items),
            "speech_seconds": speech_seconds,
            "first_seen": items[0].start if items else None,
            "last_seen": items[-1].end if items else None,
            "utterances": [utt.to_dict() for utt in items],
        }
        _write_json(json_path, payload)

        written_paths.extend([srt_path, txt_path, json_path])

    return written_paths


def write_speaker_metrics_json(
    diarization_dir: Path,
    speaker_segments: list[SpeakerSegment],
    utterances: list[TranscriptUtterance],
) -> Path:
    output_path = diarization_dir / "speaker_metrics.json"

    per_speaker_turns: dict[str, list[float]] = defaultdict(list)
    per_speaker_utterances: dict[str, list[TranscriptUtterance]] = defaultdict(list)

    for seg in speaker_segments:
        per_speaker_turns[seg.speaker].append(seg.duration)

    for utt in utterances:
        if utt.speaker:
            per_speaker_utterances[utt.speaker].append(utt)

    total_speech_seconds = round(sum(seg.duration for seg in speaker_segments), 3)

    speakers_payload = []
    for speaker in sorted(set(per_speaker_turns) | set(per_speaker_utterances)):
        turn_durations = per_speaker_turns.get(speaker, [])
        utterance_items = per_speaker_utterances.get(speaker, [])

        speech_seconds = round(sum(turn_durations), 3)
        num_turns = len(turn_durations)
        avg_turn_seconds = round(speech_seconds / num_turns, 3) if num_turns else 0.0
        sorted_turns = sorted(turn_durations)
        median_turn_seconds = (
            round(sorted_turns[len(sorted_turns) // 2], 3) if sorted_turns else 0.0
        )
        longest_turn_seconds = round(max(sorted_turns), 3) if sorted_turns else 0.0

        confidences = [
            utt.speaker_confidence
            for utt in utterance_items
            if utt.speaker_confidence is not None
        ]
        assignment_confidence_mean = (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        )

        low_confidence_segments = sum(
            1 for utt in utterance_items if "low_confidence_assignment" in utt.flags
        )

        speakers_payload.append(
            {
                "speaker": speaker,
                "speech_seconds": speech_seconds,
                "speech_ratio": round(speech_seconds / total_speech_seconds, 4) if total_speech_seconds else 0.0,
                "num_turns": num_turns,
                "avg_turn_seconds": avg_turn_seconds,
                "median_turn_seconds": median_turn_seconds,
                "longest_turn_seconds": longest_turn_seconds,
                "first_seen": min((utt.start for utt in utterance_items), default=None),
                "last_seen": max((utt.end for utt in utterance_items), default=None),
                "assignment_confidence_mean": assignment_confidence_mean,
                "low_confidence_segments": low_confidence_segments,
            }
        )

    payload = {
        "num_speakers_detected": len(speakers_payload),
        "total_speech_seconds": total_speech_seconds,
        "speakers": speakers_payload,
    }
    _write_json(output_path, payload)
    return output_path


def write_diarization_metadata_json(
    diarization_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    output_path = diarization_dir / "diarization_metadata.json"
    _write_json(output_path, metadata)
    return output_path


def write_diarization_result_json(
    diarization_dir: Path,
    result: dict[str, Any],
) -> Path:
    output_path = diarization_dir / "diarization_result.json"
    _write_json(output_path, result)
    return output_path


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _format_srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
