# src/voicecaster/intake/json_utils.py
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(path: Path, data: Any, *, indent: int = 2) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def write_json_atomic(path: Path, data: Any, *, indent: int = 2) -> None:
    ensure_parent_dir(path)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=str(path.parent),
        suffix=".tmp",
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=indent)
        tmp.write("\n")
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, path)
