from __future__ import annotations

import json
import subprocess
from pathlib import Path

import soundfile as sf
import torch


_MODEL_CACHE: dict[str, object] = {}


def _load_silero():
    cache_key = "silero_vad"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    torch.set_num_threads(1)
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps, _, _, _, _ = utils
    _MODEL_CACHE[cache_key] = (model, get_speech_timestamps)
    return _MODEL_CACHE[cache_key]


def _ensure_mono_16k_wav(input_audio: Path, output_wav: Path) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_wav),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg VAD preparation failed: {completed.stderr}")

    return output_wav


def _load_waveform_for_silero(audio_path: Path) -> torch.Tensor:
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)

    if int(sample_rate) != 16000:
        raise ValueError(f"VAD esperaba 16000 Hz y recibió {sample_rate}")

    # shape soundfile: (time, channels)
    # Silero espera tensor 1D mono o forma compatible en samples
    mono = audio[:, 0]
    return torch.from_numpy(mono)


def detect_vad_segments(audio_path: Path, output_json: Path, temp_dir: Path) -> dict:
    model, get_speech_timestamps = _load_silero()

    prepared_wav = _ensure_mono_16k_wav(
        audio_path,
        temp_dir / "vad_input_16k.wav",
    )
    wav = _load_waveform_for_silero(prepared_wav)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
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
        "prepared_audio": str(prepared_wav),
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
