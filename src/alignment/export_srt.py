from __future__ import annotations

from pathlib import Path

from .schemas import Utterance


def _format_srt_time(t: float) -> str:
    total_ms = int(round(t * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    seconds = (total_ms % 60_000) // 1000
    milliseconds = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def export_srt(path: Path, utterances: list[Utterance]) -> None:
    lines: list[str] = []

    for i, utt in enumerate(utterances, start=1):
        lines.append(str(i))
        lines.append(f"{_format_srt_time(utt.start)} --> {_format_srt_time(utt.end)}")
        lines.append(f"[{utt.speaker}] {utt.text}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
