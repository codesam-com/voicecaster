from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_EPISODES_PATH = REPO_ROOT / "inputs" / "episodes.yaml"
PROCESSED_EPISODES_PATH = REPO_ROOT / "inputs" / "episodes_processed.yaml"
REVIEWS_DIR = REPO_ROOT / "reviews"
SPEAKERS_DIR = REPO_ROOT / "data" / "speakers"
WORK_DIR = REPO_ROOT / "work"
RUNTIME_CONTROL_PATH = REPO_ROOT / "data" / "runtime_control.json"

MAX_DOWNLOAD_RETRIES = 3
MAX_PROCESSING_RETRIES = 2

DRIVE_ROOT_FOLDER_ID = os.getenv("DRIVE_ROOT_FOLDER_ID", "").strip()
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

PREAUDIT_ARCHIVE_RETENTION_DAYS = int(os.getenv("PREAUDIT_ARCHIVE_RETENTION_DAYS", "5"))
HTTP_CONNECT_TIMEOUT_SECONDS = int(os.getenv("HTTP_CONNECT_TIMEOUT_SECONDS", "15"))
HTTP_READ_TIMEOUT_SECONDS = int(os.getenv("HTTP_READ_TIMEOUT_SECONDS", "120"))
