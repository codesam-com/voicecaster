from __future__ import annotations

import json
from pathlib import Path


PHASE_DIRS = {
    "intake": "00_intake",
    "transcription": "01_transcription",
    "diarization": "02_diarization",
    "alignment": "03_alignment",
    "review": "04_review",
    "episode_outputs": "05_episode_outputs",
    "logs": "06_logs",
    "temp": "99_temp",
}


def build_work_layout(work_dir: Path) -> dict[str, Path]:
    layout = {name: work_dir / dirname for name, dirname in PHASE_DIRS.items()}
    work_dir.mkdir(parents=True, exist_ok=True)

    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)

    return layout


def write_work_readme(work_dir: Path, episode_id: str) -> None:
    content = f"""# work/{episode_id}

Directorio de trabajo del episodio.

## Regla principal
filesystem = source of truth

## Estructura
- 00_intake: ingestión
- 01_transcription: transcripción
- 02_diarization: diarización
- 03_alignment: cruce y asignación
- 04_review: revisión humana
- 05_episode_outputs: outputs estructurados de episodio
- 06_logs: reportes y logs
- 99_temp: temporales
"""
    (work_dir / "README.md").write_text(content, encoding="utf-8")


def write_status_json(work_dir: Path, payload: dict) -> None:
    (work_dir / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_work_layout(work_dir: Path, episode_id: str) -> dict[str, Path]:
    layout = build_work_layout(work_dir)
    write_work_readme(work_dir, episode_id)
    return layout
