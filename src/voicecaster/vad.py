from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch


def _load_silero():
    torch.set_num_threads(1)
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps, _, read_audio, _, _ = utils
    return model, get_speech_timestamps, read_audio


def detect_vad_segments(audio_path: Path, output_json: Path) -> dict:
    model, get_speech_timestamps, read_audio = _load_silero()

    wav = read_audio(str(audio_path), sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(speech_timestamps, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_speech_seconds = round(
        sum(float(seg["end"]) - float(seg["start"]) for seg in speech_timestamps),
        3,
    )

    return {
        "num_vad_segments": len(speech_timestamps),
        "total_speech_seconds": total_speech_seconds,
    }


def cut_audio_with_ffmpeg(input_audio: Path, start: float, end: float, output_audio: Path) -> Path:
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
        "-ss",
        str(start),
        "-to",
        str(end),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_audio),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed: {completed.stderr}")

    return output_audio
