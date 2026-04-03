# =========================================
# FILE: src/voicecaster/alignment/__init__.py
# =========================================

from .run import run_alignment

ALIGNMENT_ALGORITHM_VERSION = "04_alignment_v1"

__all__ = [
    "ALIGNMENT_ALGORITHM_VERSION",
    "run_alignment",
]
