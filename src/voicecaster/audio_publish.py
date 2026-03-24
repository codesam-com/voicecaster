from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _safe_filename(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", value.strip())
    value = re.sub(r"\s+", " ", value)
    return value[:180].strip()


def build_published_audio_filename(podcast_title: str, episode_title: str, suffix: str) -> str:
    return f"{_safe_filename(podcast_title)} - {_safe_filename(episode_title)}{suffix.lower()}"


def rewrite_audio_metadata(
    input_audio: Path,
    output_audio: Path,
    podcast_title: str,
    episode_title: str,
) -> Path:
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
        "-map",
        "0:a:0",
        "-vn",
        "-map_metadata",
        "-1",
        "-c:a",
        "copy",
        "-metadata",
        f"title={podcast_title}",
        "-metadata",
        f"subtitle={episode_title}",
        "-metadata",
        f"description={episode_title}",
    ]

    if output_audio.suffix.lower() == ".mp3":
        command.extend(["-id3v2_version", "3"])

    command.append(str(output_audio))

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg metadata rewrite failed: {completed.stderr}")

    return output_audio
