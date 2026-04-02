from __future__ import annotations

from pathlib import Path

from .schemas import AlignedUtterance


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    secs = remainder // 1_000
    millis = remainder % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def render_speaker_srt(
    utterances: list[AlignedUtterance],
) -> str:
    blocks: list[str] = []

    for i, utterance in enumerate(utterances, start=1):
        speaker = utterance.speaker or "UNKNOWN"
        text = utterance.text.strip()
        block = "\n".join(
            [
                str(i),
                f"{format_srt_timestamp(utterance.start)} --> {format_srt_timestamp(utterance.end)}",
                f"[{speaker}] {text}",
            ]
        )
        blocks.append(block)

    return "\n\n".join(blocks).strip() + "\n"


def write_speaker_srt(
    path: Path,
    utterances: list[AlignedUtterance],
) -> None:
    content = render_speaker_srt(utterances)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
