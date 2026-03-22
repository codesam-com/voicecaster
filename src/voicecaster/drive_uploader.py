from __future__ import annotations

import io
import json
import os
from pathlib import Path

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]


def build_drive_service(service_account_json: str):
    info = json.loads(service_account_json)
    credentials = Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def upload_file_resumable(service, local_path: Path, filename: str, parent_folder_id: str):
    metadata = {"name": filename, "parents": [parent_folder_id]}
    media = MediaFileUpload(str(local_path), resumable=True)
    return (
        service.files()
        .create(body=metadata, media_body=media, fields="id,name,parents", supportsAllDrives=True)
        .execute()
    )
