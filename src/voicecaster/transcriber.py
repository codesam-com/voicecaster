from __future__ import annotations

import json
from pathlib import Path

import whisper


_MODEL_CACHE: dict[str, object] = {}


def _get_model(model_name: str = "large-v3"):
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


def transcribe_audio_large_v3(audio_path: Path) -> dict:
    model = _get_model("large-v3")

    result = model.transcribe(
        str(audio_path),
        verbose=False,
        word_timestamps=False,
        condition_on_previous_text=True,
        task="transcribe",
    )
    return result


def write_whisper_outputs(result: dict, work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)

    segments = result.get("segments", []) or []
    text = (result.get("text") or "").strip()

    whisper_raw_segments = []
    srt_lines: list[str] = []
    kept = 0

    for seg in segments:
        txt = str(seg.get("text", "")).strip()
        if not txt:
            continue

        start = float(seg["start"])
        end = float(seg["end"])

        whisper_raw_segments.append(
            {
                "id": seg.get("id"),
                "start": start,
                "end": end,
                "text": txt,
            }
        )

        kept += 1
        srt_lines.append(str(kept))
        srt_lines.append(f"{_format_timestamp(start)} --> {_format_timestamp(end)}")
        srt_lines.append(txt)
        srt_lines.append("")

    (work_dir / "whisper_raw_segments.json").write_text(
        json.dumps(whisper_raw_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (work_dir / "transcript_segments.json").write_text(
        json.dumps(whisper_raw_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (work_dir / "full_transcript.txt").write_text(text + "\n", encoding="utf-8")
    (work_dir / "full_transcript.srt").write_text(
        "\n".join(srt_lines).strip() + "\n",
        encoding="utf-8",
    )

    return {
        "language": result.get("language"),
        "model_name": "large-v3",
        "num_segments_srt": kept,
        "num_segments_json": len(whisper_raw_segments),
        "text_characters": len(text),
        "text_preview": text[:500],
    }
