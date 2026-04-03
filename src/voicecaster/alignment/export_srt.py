# =========================================
# FILE: src/voicecaster/alignment/export_srt.py
# =========================================

from __future__ import annotations

from pathlib import Path

from .schemas import AlignedUtterance


def write_speaker_srt(path: Path, utterances: list[AlignedUtterance]) -> None:
    """
    Write speaker-aware SRT file from aligned utterances.
    """
    lines: list[str] = []

    for idx, utterance in enumerate(utterances, start=1):
        lines.append(str(idx))
        lines.append(
            f"{format_srt_timestamp(utterance.start)} --> {format_srt_timestamp(utterance.end)}"
        )
        lines.append(f"[{utterance.speaker}] {utterance.text}".strip())
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def format_srt_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp: HH:MM:SS,mmm
    """
    total_milliseconds = max(0, int(round(seconds * 1000)))
    hours = total_milliseconds // 3_600_000
    remainder = total_milliseconds % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    secs = remainder // 1_000
    millis = remainder % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
