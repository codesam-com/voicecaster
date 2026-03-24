from __future__ import annotations

import json
from pathlib import Path

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]


def build_drive_service(service_account_json: str):
    info = json.loads(service_account_json)
    credentials = Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def upload_file_resumable(service, local_path: Path, filename: str, parent_folder_id: str) -> dict:
    metadata = {"name": filename, "parents": [parent_folder_id]}
    media = MediaFileUpload(str(local_path), resumable=True)

    return (
        service.files()
        .create(
            body=metadata,
            media_body=media,
            fields="id,name,parents,webContentLink,webViewLink,mimeType,size",
            supportsAllDrives=True,
        )
        .execute()
    )


def create_anyone_reader_permission(service, file_id: str) -> None:
    (
        service.permissions()
        .create(
            fileId=file_id,
            body={
                "type": "anyone",
                "role": "reader",
                "allowFileDiscovery": False,
            },
            supportsAllDrives=True,
        )
        .execute()
    )


def publish_audio_to_podcasts_pending(
    service_account_json: str,
    local_path: Path,
    filename: str,
    pending_folder_id: str,
) -> dict:
    service = build_drive_service(service_account_json)

    uploaded = upload_file_resumable(
        service=service,
        local_path=local_path,
        filename=filename,
        parent_folder_id=pending_folder_id,
    )

    file_id = uploaded["id"]
    create_anyone_reader_permission(service, file_id)

    refreshed = (
        service.files()
        .get(
            fileId=file_id,
            fields="id,name,parents,webContentLink,webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )

    operational_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    return {
        "file_id": file_id,
        "name": refreshed.get("name"),
        "operational_url": operational_url,
        "web_view_url": refreshed.get("webViewLink"),
    }
