from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .models import AlignedTurn, AlignedUtterance, AlignedWord, QAResult


def ensure_alignment_dir(work_episode_dir: Path) -> Path:
    alignment_dir = work_episode_dir / "04_alignment"
    alignment_dir.mkdir(parents=True, exist_ok=True)
    return alignment_dir


def write_aligned_utterances_json(
    alignment_dir: Path,
    utterances: list[AlignedUtterance],
) -> Path:
    output_path = alignment_dir / "aligned_utterances.json"
    _write_json(output_path, [utt.to_dict() for utt in utterances])
    return output_path


def write_aligned_turns_json(
    alignment_dir: Path,
    turns: list[AlignedTurn],
) -> Path:
    output_path = alignment_dir / "aligned_turns.json"
    _write_json(output_path, [turn.to_dict() for turn in turns])
    return output_path


def write_aligned_words_json(
    alignment_dir: Path,
    words: list[AlignedWord],
) -> Path:
    output_path = alignment_dir / "aligned_words.json"
    _write_json(output_path, [word.to_dict() for word in words])
    return output_path


def write_speakers_index_json(
    alignment_dir: Path,
    payload: dict[str, Any],
) -> Path:
    output_path = alignment_dir / "speakers_index.json"
    _write_json(output_path, payload)
    return output_path


def write_alignment_metadata_json(
    alignment_dir: Path,
    payload: dict[str, Any],
) -> Path:
    output_path = alignment_dir / "alignment_metadata.json"
    _write_json(output_path, payload)
    return output_path


def write_alignment_result_json(
    alignment_dir: Path,
    payload: dict[str, Any],
) -> Path:
    output_path = alignment_dir / "alignment_result.json"
    _write_json(output_path, payload)
    return output_path


def write_qa_summary_json(
    alignment_dir: Path,
    qa_result: QAResult,
) -> Path:
    output_path = alignment_dir / "qa_summary.json"
    _write_json(output_path, qa_result.to_dict())
    return output_path


def write_transcript_final_txt(
    alignment_dir: Path,
    turns: list[AlignedTurn],
) -> Path:
    output_path = alignment_dir / "transcript_final.txt"
    chunks: list[str] = []

    for turn in turns:
        speaker_label = turn.speaker or "unknown"
        chunks.append(f"[{speaker_label}]")
        chunks.append(turn.text)
        chunks.append("")

    output_path.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")
    return output_path


def write_subtitles_final_srt(
    alignment_dir: Path,
    utterances: list[AlignedUtterance],
) -> Path:
    output_path = alignment_dir / "subtitles_final.srt"
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
