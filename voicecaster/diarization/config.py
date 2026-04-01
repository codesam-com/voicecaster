# voicecaster/diarization/config.py

from __future__ import annotations

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Identidad de la action
# -----------------------------------------------------------------------------

ACTION_NAME = "03_diarization"
ACTION_VERSION = "v1"

# -----------------------------------------------------------------------------
# Motor
# -----------------------------------------------------------------------------

DIARIZATION_ENGINE = "pyannote"
PYANNOTE_PRIMARY_PIPELINE = "pyannote/speaker-diarization-community-1"
PYANNOTE_FALLBACK_PIPELINE = "pyannote/speaker-diarization-3.1"

USE_GPU_IF_AVAILABLE = False

# -----------------------------------------------------------------------------
# Audio
# -----------------------------------------------------------------------------

AUDIO_TARGET_SAMPLE_RATE = 16000
AUDIO_TARGET_CHANNELS = 1
AUDIO_TARGET_FORMAT = "wav"

# -----------------------------------------------------------------------------
# Normalización
# -----------------------------------------------------------------------------

MIN_SEGMENT_SECONDS = 0.80
MERGE_GAP_SECONDS = 0.50
LOW_CONFIDENCE_THRESHOLD = 0.60
MIN_TRANSCRIPT_ASSIGNMENT_RATIO = 0.85

# -----------------------------------------------------------------------------
# Retries
# -----------------------------------------------------------------------------

MAX_RETRIES = 10

# -----------------------------------------------------------------------------
# Paths y entorno
# -----------------------------------------------------------------------------

HF_TOKEN_ENV_VAR = "HF_TOKEN"

PROJECT_ROOT = Path(os.getenv("GITHUB_WORKSPACE", ".")).resolve()
INPUTS_JSON_PATH = PROJECT_ROOT / "inputs" / "inputs.json"
WORK_DIR = PROJECT_ROOT / "work"

# Carpeta temporal recomendada por episodio:
TEMP_DIR_NAME = "99_temp"
OUTPUT_DIR_NAME = "03_diarization"

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------

def get_hf_token() -> str:
    token = os.getenv(HF_TOKEN_ENV_VAR, "").strip()
    if not token:
        raise RuntimeError(
            f"Missing required environment variable: {HF_TOKEN_ENV_VAR}"
        )
    return token
