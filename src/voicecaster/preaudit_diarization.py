from __future__ import annotations

from pathlib import Path

from .audio_normalize import normalize_audio_for_diarization
from .config import HF_TOKEN
from .diarizer import diarize_audio
from .preaudit_common import (
    append_report_note,
    finalize_workflow_failure,
    finalize_workflow_success,
    init_episode_context,
    preaudit_runtime_gate,
)
from .stage_selector import find_episode_for_workflow
from .status_manager import save_status_json

WORKFLOW_NAME = "preaudit-diarization"
STAGE_NAME = "diarization"


def _find_source_audio(intake_dir: Path) -> Path:
    candidates = sorted(p for p in intake_dir.iterdir() if p.is_file() and p.name.startswith("source_audio."))
    if not candidates:
        raise FileNotFoundError("No se encontró source_audio en 00_intake/")
    return candidates[0]


def run_preaudit_diarization() -> int:
    should_run, runtime_info = preaudit_runtime_gate("preaudit_diarization")
    if not should_run:
        print("Runtime control: skip diarization.")
        return 0

    ctx = find_episode_for_workflow(WORKFLOW_NAME)
    if ctx is None:
        print("No hay episodio elegible para diarization.")
        return 0

    layout, status_payload, report_payload = init_episode_context(ctx, WORKFLOW_NAME, STAGE_NAME, "normalize_for_diarization")

    try:
        append_report_note(report_payload, f"Runtime control workflow=preaudit_diarization decision={runtime_info.get('decision')}")
        audio_path = _find_source_audio(layout["intake"])

        normalized_audio = normalize_audio_for_diarization(audio_path, layout["temp"] / "audio_mono_16k.wav")
        append_report_note(report_payload, f"Audio normalizado para diarización: {normalized_audio.name}")

        status_payload["current_step"] = "pyannote"
        save_status_json(ctx.work_dir, status_payload)

        diarization_meta = diarize_audio(
            normalized_audio,
            layout["diarization"] / "speaker_timeline.rttm",
            layout["diarization"] / "speaker_segments.json",
            HF_TOKEN,
        )
        append_report_note(report_payload, f"Diarización generada ({diarization_meta['num_speakers_detected']} hablantes, {diarization_meta['num_segments']} segmentos)")
        append_report_note(report_payload, f"Tiempos diarización: {diarization_meta['timing_seconds']}")

        finalize_workflow_success(ctx, WORKFLOW_NAME, STAGE_NAME, layout, status_payload, report_payload)
        print(f"Diarization completado para {ctx.episode.id}")
        return 0

    except Exception as exc:
        finalize_workflow_failure(
            ctx, WORKFLOW_NAME, STAGE_NAME, status_payload.get("current_step", "normalize_for_diarization"),
            layout, status_payload, report_payload, exc
        )
        raise
