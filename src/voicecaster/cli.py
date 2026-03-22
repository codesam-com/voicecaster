from __future__ import annotations

import sys

from .config import INPUT_EPISODES_PATH, PROCESSED_EPISODES_PATH, REPO_ROOT
from .postaudit import run_postaudit
from .preaudit import run_preaudit


def _print_runtime_paths() -> None:
    print(f"REPO_ROOT={REPO_ROOT}")
    print(f"INPUT_EPISODES_PATH={INPUT_EPISODES_PATH}")
    print(f"PROCESSED_EPISODES_PATH={PROCESSED_EPISODES_PATH}")


def main() -> int:
    _print_runtime_paths()

    if len(sys.argv) < 2:
        print("Uso: python -m voicecaster.cli [preaudit|postaudit]")
        return 1

    command = sys.argv[1].strip().lower()

    if command == "preaudit":
        return run_preaudit()
    if command == "postaudit":
        return run_postaudit()

    print(f"Comando no soportado: {command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
