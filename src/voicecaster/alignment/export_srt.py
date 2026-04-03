from __future__ import annotations

from pathlib import Path

from .schemas import SubtitleCue


def write_speaker_srt(path: Path, cues: list[SubtitleCue]) -> None:
    """
    Write speaker-aware SRT file from subtitle cues.
    """
    lines: list[str] = []

    for idx, cue in enumerate(cues, start=1):
        lines.append(str(idx))
        lines.append(
            f"{format_srt_timestamp(cue.start)} --> {format_srt_timestamp(cue.end)}"
        )

        cue_text_lines = cue.text.splitlines() if cue.text else [""]
        if cue_text_lines:
            first_line = f"[{cue.speaker}] {cue_text_lines[0]}".strip()
            lines.append(first_line)
            for extra_line in cue_text_lines[1:]:
                lines.append(extra_line)
        else:
            lines.append(f"[{cue.speaker}]")

        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    hours = total_milliseconds // 3_600_000
    remainder = total_milliseconds % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    secs = remainder // 1_000
    millis = remainder % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
