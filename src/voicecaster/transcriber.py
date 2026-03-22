from __future__ import annotations

from pathlib import Path
from datetime import timedelta

import whisper


def _format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    ms = int((seconds - total_seconds) * 1000)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def transcribe_to_srt(audio_path: Path, output_srt: Path) -> dict:
    model = whisper.load_model("base")

    result = model.transcribe(
        str(audio_path),
        verbose=False,
        word_timestamps=False,
    )

    segments = result.get("segments", [])

    lines = []
    for idx, seg in enumerate(segments, start=1):
        start = _format_timestamp(seg["start"])
        end = _format_timestamp(seg["end"])
        text = seg["text"].strip()

        lines.append(f"{idx}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    output_srt.write_text("\n".join(lines), encoding="utf-8")

    return {
        "language": result.get("language"),
        "num_segments": len(segments),
    }
