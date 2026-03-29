# src/voicecaster/intake/validate_audio.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from .media_ffprobe import FFprobeValidationError, extract_audio_metadata, has_audio_stream, run_ffprobe


class AudioValidationError(Exception):
    """Raised when a downloaded file is not a workable audio file."""


def validate_audio(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        raise AudioValidationError("El archivo temporal descargado no existe.")

    ffprobe_data = run_ffprobe(file_path)

    if not has_audio_stream(ffprobe_data):
        raise AudioValidationError("ffprobe no detectó ningún stream de audio.")

    metadata = extract_audio_metadata(ffprobe_data, file_path)
    return {
        "ffprobe_raw": ffprobe_data,
        "source_metadata": metadata,
    }
