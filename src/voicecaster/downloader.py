from __future__ import annotations

from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import (
    HTTP_CONNECT_TIMEOUT_SECONDS,
    HTTP_READ_TIMEOUT_SECONDS,
    MAX_DOWNLOAD_RETRIES,
)

class DownloadError(Exception):
    pass


class IncompatibleSourceError(Exception):
    pass


def _guess_extension(content_type: str | None, fallback_url: str) -> str:
    if content_type:
        lowered = content_type.lower()
        if "audio/mpeg" in lowered:
            return ".mp3"
        if "audio/mp4" in lowered or "audio/x-m4a" in lowered:
            return ".m4a"
        if "audio/wav" in lowered or "audio/x-wav" in lowered:
            return ".wav"
        if "audio/ogg" in lowered:
            return ".ogg"
        if "audio/flac" in lowered:
            return ".flac"

    parsed = urlparse(fallback_url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".mp4"}:
        return suffix

    return ".bin"


def _is_probably_downloadable_audio(response: requests.Response) -> bool:
    content_type = (response.headers.get("Content-Type") or "").lower()
    disposition = (response.headers.get("Content-Disposition") or "").lower()

    if content_type.startswith("audio/"):
        return True

    if "attachment" in disposition:
        return True

    if "application/octet-stream" in content_type:
        return True

    return False


@retry(
    stop=stop_after_attempt(MAX_DOWNLOAD_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def download_audio_to_workdir(url: str, target_dir: Path, episode_id: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "voicecaster/0.1 (+https://github.com/codesam-com/voicecaster)"
    }

    try:
        with requests.get(url, headers=headers, stream=True, timeout=(HTTP_CONNECT_TIMEOUT_SECONDS, HTTP_READ_TIMEOUT_SECONDS), allow_redirects=True) as response:
            response.raise_for_status()

            if not _is_probably_downloadable_audio(response):
                raise IncompatibleSourceError(
                    f"El recurso no parece audio descargable. "
                    f"content-type={response.headers.get('Content-Type')!r}, "
                    f"content-disposition={response.headers.get('Content-Disposition')!r}"
                )

            extension = _guess_extension(response.headers.get("Content-Type"), str(response.url))
            output_path = target_dir / f"{episode_id}_source{extension}"

            total_bytes = 0
            with output_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    total_bytes += len(chunk)

            if total_bytes == 0:
                raise DownloadError("La descarga terminó vacía.")

            return output_path

    except IncompatibleSourceError:
        raise
    except requests.HTTPError as exc:
        raise DownloadError(f"HTTP error durante la descarga: {exc}") from exc
    except requests.RequestException as exc:
        raise DownloadError(f"Error de red durante la descarga: {exc}") from exc
