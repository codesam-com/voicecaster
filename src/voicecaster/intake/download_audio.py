# src/voicecaster/intake/download_audio.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from .error_classification import build_error_payload, classify_http_status


class DownloadContentError(Exception):
    """Raised when the source responds but is not usable as content."""


def download_audio(url: str, destination_path: Path, timeout_seconds: int = 120) -> dict[str, Any]:
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout_seconds, allow_redirects=True) as response:
        if response.status_code != 200:
            error_type = classify_http_status(response.status_code)
            payload = build_error_payload(
                message=f"HTTP status no exitoso: {response.status_code}",
                error_type=error_type,
                extra={"status_code": response.status_code, "url": url},
            )
            raise DownloadContentError(str(payload))

        bytes_written = 0
        with destination_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                bytes_written += len(chunk)

        if bytes_written <= 0:
            payload = build_error_payload(
                message="La descarga terminó sin bytes útiles.",
                error_type="content",
                extra={"url": url},
            )
            raise DownloadContentError(str(payload))

        return {
            "final_url": response.url,
            "content_type": response.headers.get("Content-Type"),
            "content_length_header": response.headers.get("Content-Length"),
            "bytes_written": bytes_written,
        }
