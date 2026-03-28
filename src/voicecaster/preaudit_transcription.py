from __future__ import annotations

from pathlib import Path

from .preaudit_common import (
    append_report_note,
    finalize_workflow_failure,
    finalize_workflow_success,
    init_episode_context,
    preaudit_runtime_gate,
)
from .stage_selector import find_episode_for_workflow
from .status_manager import save_status_json
from .transcriber import transcribe_audio_large_v3, write_whisper_outputs
from .vad import detect_vad_segments
from .whisperx_aligner import align_with_whisperx

WORKFLOW_NAME = "preaudit-transcription"
STAGE_NAME = "transcription"


def _find_source_audio(intake_dir: Path) -> Path:
    candidates = sorted(p for p in intake_dir.iterdir() if p.is_file() and p.name.startswith("source_audio."))
    if not candidates:
        raise FileNotFoundError("No se encontró source_audio en 00_intake/")
    return candidates[0]


def run_preaudit_transcription() -> int:
    should_run, runtime_info = preaudit_runtime_gate("preaudit_transcription")
    if not should_run:
        print("Runtime control: skip transcription.")
        return 0

    ctx = find_episode_for_workflow(WORKFLOW_NAME)
    if ctx is None:
        print("No hay episodio elegible para transcription.")
        return 0

    layout, status_payload, report_payload = init_episode_context(ctx, WORKFLOW_NAME, STAGE_NAME, "vad")

    try:
        append_report_note(report_payload, f"Runtime control workflow=preaudit_transcription decision={runtime_info.get('decision')}")
        audio_path = _find_source_audio(layout["intake"])

        vad_meta = detect_vad_segments(audio_path, layout["transcription"] / "vad_segments.json", layout["temp"])
        append_report_note(report_payload, f"VAD generado ({vad_meta['num_vad_segments']} segmentos)")
        append_report_note(report_payload, f"Voz detectada: {vad_meta['total_speech_seconds']} segundos")

        status_payload["current_step"] = "whisper_large_v3"
        save_status_json(ctx.work_dir, status_payload)

        whisper_result = transcribe_audio_large_v3(audio_path)
        transcription_meta = write_whisper_outputs(whisper_result, layout["transcription"])
        append_report_note(report_payload, f"Transcripción generada ({transcription_meta['num_segments_srt']} segmentos)")
        append_report_note(report_payload, f"Idioma detectado: {transcription_meta.get('language') or 'desconocido'}")
        append_report_note(report_payload, f"Texto transcrito: {transcription_meta.get('text_characters', 0)} caracteres")

        status_payload["current_step"] = "whisperx_alignment"
        save_status_json(ctx.work_dir, status_payload)

        whisperx_meta = align_with_whisperx(
            audio_path,
            layout["transcription"] / "transcript_segments.json",
            layout["transcription"] / "whisperx_aligned_segments.json",
            transcription_meta.get("language"),
        )
        append_report_note(report_payload, f"WhisperX alignment generado ({whisperx_meta['num_aligned_segments']} segmentos)")

        finalize_workflow_success(ctx, WORKFLOW_NAME, STAGE_NAME, layout, status_payload, report_payload)
        print(f"Transcription completado para {ctx.episode.id}")
        return 0

    except Exception as exc:
        finalize_workflow_failure(
            ctx, WORKFLOW_NAME, STAGE_NAME, status_payload.get("current_step", "vad"),
            layout, status_payload, report_payload, exc
        )
        raise
