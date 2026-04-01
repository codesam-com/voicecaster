from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class AudioPreparationError(RuntimeError):
    pass


def _normalize_google_drive_url(url: str) -> str:
    """
    Convierte algunas variantes típicas de Google Drive a formato descargable directo.
    Si no reconoce patrón Drive, devuelve la URL tal cual.
    """
    parsed = urlparse(url)

    if "drive.google.com" not in parsed.netloc:
        return url

    # Caso 1: /file/d/<ID>/view
    parts = [p for p in parsed.path.split("/") if p]
    if "file" in parts and "d" in parts:
        try:
            file_id = parts[parts.index("d") + 1]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        except Exception:
            return url

    # Caso 2: ya viene con id en query
    query = parse_qs(parsed.query)
    file_id = query.get("id", [None])[0]
    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    return url


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; voicecaster/1.0)"
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response, destination.open("wb") as f:
            shutil.copyfileobj(response, f)
    except Exception as exc:  # noqa: BLE001
        raise AudioPreparationError(
            f"Failed to download audio from url={url}: {exc}"
        ) from exc


def _run_ffmpeg_to_wav_16k_mono(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise AudioPreparationError(f"Failed to execute ffmpeg: {exc}") from exc

    if completed.returncode != 0:
        raise AudioPreparationError(
            "ffmpeg conversion failed. "
            f"returncode={completed.returncode} "
            f"stderr={completed.stderr[-4000:]}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise AudioPreparationError(
            f"Prepared audio file was not created correctly: {output_path}"
        )


def prepare_temp_audio_path(temp_dir: Path, episode: dict) -> Path:
    """
    Descarga el audio del episodio y lo convierte a un WAV mono 16 kHz listo para pyannote.

    Estructura generada:
      temp_dir/
        source_audio.bin
        audio.wav
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    raw_url = str(episode.get("url", "")).strip()
    if not raw_url:
        raise AudioPreparationError("Episode has no valid 'url'")

    download_url = _normalize_google_drive_url(raw_url)

    source_audio_path = temp_dir / "source_audio.bin"
    prepared_audio_path = temp_dir / "audio.wav"

    _download_file(download_url, source_audio_path)
    _run_ffmpeg_to_wav_16k_mono(source_audio_path, prepared_audio_path)

    return prepared_audio_path
