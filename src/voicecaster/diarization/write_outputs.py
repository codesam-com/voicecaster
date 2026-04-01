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

        srt_path.write_text(
            "\n\n".join(srt_blocks) + ("\n" if srt_blocks else ""),
            encoding="utf-8",
        )
        txt_path.write_text(
            "\n".join(full_text_parts).strip() + ("\n" if full_text_parts else ""),
            encoding="utf-8",
        )

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
    metrics_payload: dict[str, Any],
) -> Path:
    output_path = diarization_dir / "speaker_metrics.json"
    _write_json(output_path, metrics_payload)
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
