# src/voicecaster/intake/fs_workdir.py
from __future__ import annotations

import shutil
from pathlib import Path

from .fs_paths import get_episode_temp_dir, get_episode_workdir


def reset_episode_workdir(episode_id: str) -> Path:
    workdir = get_episode_workdir(episode_id)
    if workdir.exists():
        shutil.rmtree(workdir)
    return workdir


def create_episode_workdir_structure(episode_id: str) -> Path:
    workdir = get_episode_workdir(episode_id)
    (workdir / "00_intake").mkdir(parents=True, exist_ok=True)
    (workdir / "01_logs").mkdir(parents=True, exist_ok=True)
    (workdir / "99_temp").mkdir(parents=True, exist_ok=True)
    return workdir


def write_workdir_readme(episode_id: str) -> None:
    workdir = get_episode_workdir(episode_id)
    readme_path = workdir / "README.md"
    content = f"""# work/{episode_id}

Carpeta de trabajo del episodio `{episode_id}` para el workflow `intake`.

## Estructura
- `00_intake/` → datos persistentes de intake
- `01_logs/` → logs y reportes técnicos
- `99_temp/` → temporales del run; no deben persistir al final
"""
    readme_path.write_text(content, encoding="utf-8")


def cleanup_temp_dir(episode_id: str) -> None:
    temp_dir = get_episode_temp_dir(episode_id)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
