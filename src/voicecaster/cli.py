from __future__ import annotations

import sys

from .postaudit import run_postaudit
from .preaudit import run_preaudit


def main() -> int:
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
