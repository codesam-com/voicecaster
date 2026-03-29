# src/voicecaster/intake/normalize_source.py
from __future__ import annotations

from urllib.parse import parse_qs, urlparse


def normalize_source(url: str) -> dict[str, str]:
    url = url.strip()

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    if "drive.google.com" in hostname:
        query = parse_qs(parsed.query)
        file_id = query.get("id", [None])[0]

        if not file_id:
            path_parts = [part for part in parsed.path.split("/") if part]
            if "d" in path_parts:
                idx = path_parts.index("d")
                if idx + 1 < len(path_parts):
                    file_id = path_parts[idx + 1]

        if file_id:
            normalized = f"https://drive.google.com/uc?export=download&id={file_id}"
            return {
                "url_original": url,
                "url_normalized": normalized,
                "source_type": "google_drive",
                "drive_file_id": file_id,
            }

    return {
        "url_original": url,
        "url_normalized": url,
        "source_type": "generic_http",
    }
