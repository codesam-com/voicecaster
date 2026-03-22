from __future__ import annotations

import json
from pathlib import Path

import whisper

_MODEL_CACHE: dict[str, object] = {}


def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = whisper.load_model(model_name)
    return _MODEL_CACHE[model_name]


def _format_timestamp(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000))
    hours = total_milliseconds // 3_600_000
    minutes = (total_milliseconds % 3_600_000) // 60_000
    secs = (total_milliseconds % 60_000) // 1000
    millis = total_milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def transcribe_audio(audio_path: Path, model_name: str = "base") -> dict:
    model = _get_model(model_name)
    result = model.transcribe(
        str(audio_path),
        verbose=False,
        word_timestamps=False,
        condition_on_previous_text=True,
    )
    return result


def write_srt(result: dict, output_srt: Path) -> int:
    segments = result.get("segments", [])
    output_srt.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    kept = 0

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        kept += 1
        start = _format_timestamp(float(seg["start"]))
        end = _format_timestamp(float(seg["end"]))

        lines.append(str(kept))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    output_srt.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return kept


def write_plain_text(result: dict, output_txt: Path) -> None:
    text = (result.get("text") or "").strip()
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(text + "\n", encoding="utf-8")


def write_segments_json(result: dict, output_json: Path) -> int:
    raw_segments = result.get("segments", [])
    payload = []

    for seg in raw_segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        payload.append(
            {
                "id": seg.get("id"),
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": text,
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(payload)


def transcribe_bundle(audio_path: Path, work_dir: Path, model_name: str = "base") -> dict:
    result = transcribe_audio(audio_path, model_name=model_name)

    srt_segments = write_srt(result, work_dir / "full_transcript.srt")
    write_plain_text(result, work_dir / "full_transcript.txt")
    json_segments = write_segments_json(result, work_dir / "transcript_segments.json")

    text = (result.get("text") or "").strip()

    return {
        "language": result.get("language"),
        "model_name": model_name,
        "num_segments_srt": srt_segments,
        "num_segments_json": json_segments,
        "text_preview": text[:500],
        "text_characters": len(text),
    }
