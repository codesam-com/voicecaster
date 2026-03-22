from __future__ import annotations

import json
import subprocess
from pathlib import Path


class AudioProbeError(Exception):
    pass


def probe_audio_file(audio_path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(audio_path),
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise AudioProbeError(
            f"ffprobe no pudo analizar el archivo: {completed.stderr.strip()}"
        )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AudioProbeError("ffprobe devolvió una salida JSON inválida.") from exc

    format_info = payload.get("format") or {}
    streams = payload.get("streams") or []

    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise AudioProbeError("No se detectaron streams de audio válidos.")

    duration_raw = format_info.get("duration")
    duration_seconds = float(duration_raw) if duration_raw is not None else None

    primary_audio = audio_streams[0]

    return {
        "duration_seconds": duration_seconds,
        "container_format": format_info.get("format_name"),
        "bit_rate": format_info.get("bit_rate"),
        "size_bytes": format_info.get("size"),
        "audio_codec": primary_audio.get("codec_name"),
        "sample_rate": primary_audio.get("sample_rate"),
        "channels": primary_audio.get("channels"),
        "channel_layout": primary_audio.get("channel_layout"),
    }
