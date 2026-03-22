from __future__ import annotations

import subprocess
from pathlib import Path


def normalize_audio_for_diarization(input_audio: Path, output_audio: Path) -> Path:
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_audio),
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg normalization failed: {completed.stderr}")

    return output_audio
