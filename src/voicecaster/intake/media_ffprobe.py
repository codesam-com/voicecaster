# src/voicecaster/intake/media_ffprobe.py
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class FFprobeValidationError(Exception):
    """Raised when ffprobe cannot validate a media file."""


def run_ffprobe(file_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        str(file_path),
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise FFprobeValidationError(
            f"ffprobe falló con código {completed.returncode}: {stderr}"
        )

    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FFprobeValidationError("ffprobe devolvió JSON inválido.") from exc

    return data


def has_audio_stream(ffprobe_data: dict[str, Any]) -> bool:
    streams = ffprobe_data.get("streams", [])
    if not isinstance(streams, list):
        return False

    return any(stream.get("codec_type") == "audio" for stream in streams if isinstance(stream, dict))


def extract_audio_metadata(ffprobe_data: dict[str, Any], file_path: Path) -> dict[str, Any]:
    format_info = ffprobe_data.get("format", {}) if isinstance(ffprobe_data.get("format"), dict) else {}
    streams = ffprobe_data.get("streams", []) if isinstance(ffprobe_data.get("streams"), list) else []

    first_audio_stream = next(
        (stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "audio"),
        {},
    )

    duration_raw = format_info.get("duration")
    duration_seconds: float | None
    try:
        duration_seconds = round(float(duration_raw), 3) if duration_raw is not None else None
    except (TypeError, ValueError):
        duration_seconds = None

    sample_rate_raw = first_audio_stream.get("sample_rate")
    try:
        sample_rate = int(sample_rate_raw) if sample_rate_raw is not None else None
    except (TypeError, ValueError):
        sample_rate = None

    channels_raw = first_audio_stream.get("channels")
    try:
        channels = int(channels_raw) if channels_raw is not None else None
    except (TypeError, ValueError):
        channels = None

    bit_rate_raw = format_info.get("bit_rate")
    try:
        bit_rate = int(bit_rate_raw) if bit_rate_raw is not None else None
    except (TypeError, ValueError):
        bit_rate = None

    return {
        "ffprobe_ok": True,
        "duration_seconds": duration_seconds,
        "container": format_info.get("format_name"),
        "audio_codec": first_audio_stream.get("codec_name"),
        "sample_rate": sample_rate,
        "channels": channels,
        "bit_rate": bit_rate,
        "size_bytes": file_path.stat().st_size if file_path.exists() else None,
    }
