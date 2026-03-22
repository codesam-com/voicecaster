from __future__ import annotations

import re
from urllib.parse import urlparse, parse_qs


def normalize_download_url(raw_url: str) -> str:
    if "drive.google.com" not in raw_url:
        return raw_url

    parsed = urlparse(raw_url)

    query = parse_qs(parsed.query)
    if "id" in query and query["id"]:
        file_id = query["id"][0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    match = re.search(r"/file/d/([^/]+)/", raw_url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    return raw_url
