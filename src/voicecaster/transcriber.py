from __future__ import annotations

from pathlib import Path

import whisper

_MODEL_CACHE: dict[str, object] = {}


def _format_timestamp(seconds: float) -> str:
    total_milliseconds = int(round(max(seconds, 0.0) * 1000))
    hours = total_milliseconds // 3_600_000
    minutes = (total_milliseconds % 3_600_000) // 60_000
    secs = (total_milliseconds % 60_000) // 1000
    millis = total_milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = whisper.load_model(model_name)
    return _MODEL_CACHE[model_name]


def transcribe_to_srt(audio_path: Path, output_srt: Path, model_name: str = "base") -> dict:
    model = _get_model(model_name)

    result = model.transcribe(
        str(audio_path),
        verbose=False,
        word_timestamps=False,
        condition_on_previous_text=True,
    )

    segments = result.get("segments", [])
    output_srt.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    valid_segments = 0

    for idx, seg in enumerate(segments, start=1):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        start = _format_timestamp(float(seg["start"]))
        end = _format_timestamp(float(seg["end"]))

        lines.append(str(valid_segments + 1))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

        valid_segments += 1

    output_srt.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    full_text = (result.get("text") or "").strip()

    return {
        "language": result.get("language"),
        "num_segments": valid_segments,
        "model_name": model_name,
        "text_preview": full_text[:500],
        "full_text_characters": len(full_text),
    }
