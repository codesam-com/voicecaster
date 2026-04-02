from __future__ import annotations

DIARIZATION_ENGINE = "pyannote"

PYANNOTE_PRIMARY_PIPELINE = "pyannote/speaker-diarization-community-1"
PYANNOTE_FALLBACK_PIPELINE = "pyannote/speaker-diarization-3.1"

USE_GPU_IF_AVAILABLE = False

AUDIO_TARGET_SAMPLE_RATE = 16000
AUDIO_TARGET_CHANNELS = 1

MIN_SEGMENT_SECONDS = 0.80
MERGE_GAP_SECONDS = 0.50
LOW_CONFIDENCE_THRESHOLD = 0.60
MIN_TRANSCRIPT_ASSIGNMENT_RATIO = 0.85

MAX_RETRIES = 10
SPEAKER_LABEL_PREFIX = "speaker_"
SPEAKER_LABEL_PADDING = 2

# -------------------------
# Post-processing
# -------------------------

# Remove speakers whose total share is below this ratio.
POSTPROCESS_MIN_SPEAKER_RATIO = 0.01  # 1%

# Or whose absolute speech time is below this threshold.
POSTPROCESS_MIN_SPEAKER_SECONDS = 12.0

# Merge same-speaker neighboring segments if the gap is small.
POSTPROCESS_MERGE_GAP_SECONDS = 0.75

# Merge/remove tiny bridge segments in A-B-A patterns when B is shorter than this.
POSTPROCESS_MAX_BRIDGE_SECONDS = 1.25

# Only smooth A-B-A if total bridge neighborhood is bounded.
POSTPROCESS_MAX_ABA_WINDOW_SECONDS = 8.0

# Drop extremely short standalone segments after smoothing.
POSTPROCESS_DROP_MICROSEGMENTS_SECONDS = 0.35
