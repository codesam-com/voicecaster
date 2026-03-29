# src/voicecaster/intake/inputs_load_and_save.py
from __future__ import annotations

from typing import Any

from .fs_paths import get_inputs_json_path
from .json_utils import read_json_file, write_json_atomic


def load_inputs() -> list[dict[str, Any]]:
    path = get_inputs_json_path()
    data = read_json_file(path)

    if not isinstance(data, list):
        raise ValueError("inputs/inputs.json debe contener un array JSON.")

    episodes: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Cada episodio de inputs/inputs.json debe ser un objeto JSON.")
        episodes.append(item)

    return episodes


def save_inputs(episodes: list[dict[str, Any]]) -> None:
    path = get_inputs_json_path()
    write_json_atomic(path, episodes, indent=2)
