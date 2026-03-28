from __future__ import annotations

import sys

from .config import INPUT_EPISODES_PATH, PROCESSED_EPISODES_PATH, REPO_ROOT
from .postaudit import run_postaudit
from .preaudit_alignment import run_preaudit_alignment
from .preaudit_diarization import run_preaudit_diarization
from .preaudit_intake import run_preaudit_intake
from .preaudit_review_prepare import run_preaudit_review_prepare
from .preaudit_transcription import run_preaudit_transcription


def _print_runtime_paths() -> None:
    print(f"REPO_ROOT={REPO_ROOT}")
    print(f"INPUT_EPISODES_PATH={INPUT_EPISODES_PATH}")
    print(f"PROCESSED_EPISODES_PATH={PROCESSED_EPISODES_PATH}")


def main() -> int:
    _print_runtime_paths()

    if len(sys.argv) < 2:
        print("Uso: python -m voicecaster.cli [preaudit-intake|preaudit-transcription|preaudit-diarization|preaudit-alignment|preaudit-review-prepare|postaudit]")
        return 1

    command = sys.argv[1].strip().lower()

    if command == "preaudit-intake":
        return run_preaudit_intake()
    if command == "preaudit-transcription":
        return run_preaudit_transcription()
    if command == "preaudit-diarization":
        return run_preaudit_diarization()
    if command == "preaudit-alignment":
        return run_preaudit_alignment()
    if command == "preaudit-review-prepare":
        return run_preaudit_review_prepare()
    if command == "postaudit":
        return run_postaudit()

    print(f"Comando no soportado: {command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
