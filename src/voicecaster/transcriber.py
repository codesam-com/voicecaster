from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import whisper


def _format_timestamp(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000))
    hours = total_milliseconds // 3_600_000
    minutes = (total_milliseconds % 3_600_000) // 60_000
    secs = (total_milliseconds % 60_000) // 1000
    millis = total_milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def transcribe_to_srt(audio_path: Path, output_srt: Path, model_name: str = "base") -> dict:
    model = whisper.load_model(model_name)

    result = model.transcribe(
        str(audio_path),
        verbose=False,
        word_timestamps=False,
        condition_on_previous_text=True,
    )

    segments = result.get("segments", [])
    output_srt.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        start = _format_timestamp(float(seg["start"]))
        end = _format_timestamp(float(seg["end"]))
        text = str(seg["text"]).strip()

        if not text:
            continue

        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    output_srt.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    return {
        "language": result.get("language"),
        "num_segments": len([s for s in segments if str(s.get("text", "")).strip()]),
        "model_name": model_name,
        "text_preview": (result.get("text") or "").strip()[:500],
    }
