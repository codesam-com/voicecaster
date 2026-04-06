from __future__ import annotations

from pathlib import Path

WORKFLOW_NAME = "speaker_identity"
TARGET_STATUS = "speaker_identity"
SUCCESS_STATUS = "review_preparation"
RUINED_STATUS = "ruined"

INPUTS_PATH = Path("inputs/inputs.json")
WORK_DIR = Path("work")

MAX_RETRIES = 10

ENABLE_TEXT_SUPPORT = True

MIN_SPEECH_SECONDS_FOR_ANALYSIS = 3.0
MIN_NUM_WORDS_FOR_ANALYSIS = 8

MIN_SELECTED_SEGMENT_SECONDS = 2.5
MIN_SELECTED_SEGMENT_WORDS = 6
MAX_SELECTED_SEGMENTS_PER_SPEAKER = 8

UNKNOWN_DISPLAY_NAME = "Desconocido"
