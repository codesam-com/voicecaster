from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .models import RawSpeakerSegment, SpeakerSegment, TranscriptUtterance


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _build_srt_block(index: int, start: float, end: float, text: str) -> str:
    start_str = _format_srt_timestamp(start)
    end_str = _format_srt_timestamp(end)
    return f"{index}\n{start_str} --> {end_str}\n{text.strip()}\n"


def write_diarization_raw_json(
    output_dir: Path,
    raw_segments: list[RawSpeakerSegment],
    engine_metadata: dict[str, Any],
) -> Path:
    path = output_dir / "diarization_raw.json"
    payload = {
        "engine_metadata": engine_metadata,
        "segments": [seg.to_dict() for seg in raw_segments],
    }
    _write_json(path, payload)
    return path


def write_speaker_segments_json(
    output_dir: Path,
    speaker_segments: list[SpeakerSegment],
    label_mapping: dict[str, str],
    normalization_warnings: list[str],
) -> Path:
    path = output_dir / "speaker_segments.json"
    payload = {
        "label_mapping": label_mapping,
        "warnings": normalization_warnings,
        "segments": [seg.to_dict() for seg in speaker_segments],
    }
    _write_json(path, payload)
    return path


def write_transcript_with_speakers_json(
    output_dir: Path,
    utterances: list[TranscriptUtterance],
    reconciliation_stats: dict[str, Any],
) -> Path:
    path = output_dir / "transcript_with_speakers.json"
    payload = {
        "stats": reconciliation_stats,
        "utterances": [utt.to_dict() for utt in utterances],
    }
    _write_json(path, payload)
    return path


def write_subtitles_diarized_srt(
    output_dir: Path,
    utterances: list[TranscriptUtterance],
) -> Path:
    path = output_dir / "subtitles_diarized.srt"
    _ensure_dir(path.parent)

    blocks: list[str] = []
    counter = 1

    for utt in utterances:
        if not utt.text.strip():
            continue
        speaker = utt.speaker or "unknown_speaker"
        text = f"[{speaker}] {utt.text}"
        blocks.append(_build_srt_block(counter, utt.start, utt.end, text))
        counter += 1

    path.write_text("\n".join(blocks).strip() + "\n", encoding="utf-8")
    return path


def write_per_speaker_outputs(
    output_dir: Path,
    utterances: list[TranscriptUtterance],
) -> list[Path]:
    speakers_dir = output_dir / "speakers"
    _ensure_dir(speakers_dir)

    grouped: dict[str, list[TranscriptUtterance]] = defaultdict(list)
    for utt in utterances:
        speaker = utt.speaker or "unknown_speaker"
        grouped[speaker].append(utt)

    written_paths: list[Path] = []

    for speaker, items in sorted(grouped.items(), key=lambda x: x[0]):
        srt_path = speakers_dir / f"{speaker}.srt"
        txt_path = speakers_dir / f"{speaker}.txt"
        json_path = speakers_dir / f"{speaker}.json"

        # SRT
        srt_blocks: list[str] = []
        for idx, utt in enumerate(items, start=1):
            if not utt.text.strip():
                continue
            srt_blocks.append(_build_srt_block(idx, utt.start, utt.end, utt.text))
        srt_path.write_text("\n".join(srt_blocks).strip() + "\n", encoding="utf-8")

        # TXT
        text_lines = [utt.text.strip() for utt in items if utt.text.strip()]
        txt_path.write_text("\n".join(text_lines).strip() + "\n", encoding="utf-8")

        # JSON resumen por speaker
        durations = [utt.duration for utt in items]
        payload = {
            "speaker": speaker,
            "num_utterances": len(items),
            "speech_seconds": round(sum(durations), 3),
            "first_seen": round(min((utt.start for utt in items), default=0.0), 3),
            "last_seen": round(max((utt.end for utt in items), default=0.0), 3),
            "utterances": [utt.to_dict() for utt in items],
        }
        _write_json(json_path, payload)

        written_paths.extend([srt_path, txt_path, json_path])

    return written_paths


def write_speaker_metrics_json(
    output_dir: Path,
    utterances: list[TranscriptUtterance],
) -> Path:
    path = output_dir / "speaker_metrics.json"

    grouped: dict[str, list[TranscriptUtterance]] = defaultdict(list)
    for utt in utterances:
        speaker = utt.speaker or "unknown_speaker"
        grouped[speaker].append(utt)

    total_speech_seconds = sum(utt.duration for utt in utterances)
    speakers_payload: list[dict[str, Any]] = []

    for speaker, items in sorted(grouped.items(), key=lambda x: x[0]):
        speech_seconds = sum(utt.duration for utt in items)
        num_turns = len(items)
        avg_turn_seconds = speech_seconds / num_turns if num_turns else 0.0
        longest_turn_seconds = max((utt.duration for utt in items), default=0.0)

        speakers_payload.append(
            {
                "speaker": speaker,
                "speech_seconds": round(speech_seconds, 3),
                "speech_ratio": round(
                    speech_seconds / total_speech_seconds, 4
                ) if total_speech_seconds > 0 else 0.0,
                "num_turns": num_turns,
                "avg_turn_seconds": round(avg_turn_seconds, 3),
                "longest_turn_seconds": round(longest_turn_seconds, 3),
            }
        )

    payload = {
        "num_speakers_detected": len(grouped),
        "total_speech_seconds": round(total_speech_seconds, 3),
        "speakers": speakers_payload,
    }
    _write_json(path, payload)
    return path


def write_diarization_metadata_json(
    output_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    path = output_dir / "diarization_metadata.json"
    _write_json(path, metadata)
    return path


def write_diarization_result_json(
    output_dir: Path,
    result_payload: dict[str, Any],
) -> Path:
    path = output_dir / "diarization_result.json"
    _write_json(path, result_payload)
    return path
