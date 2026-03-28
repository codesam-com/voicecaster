from __future__ import annotations

from pathlib import Path

from .preaudit_common import (
    append_report_note,
    finalize_workflow_failure,
    finalize_workflow_success,
    init_episode_context,
    preaudit_runtime_gate,
)
from .redecode import redecode_doubtful_segments
from .speaker_alignment import (
    assign_speakers_to_transcript_segments,
    write_full_transcript_with_speakers,
)
from .stage_selector import find_episode_for_workflow
from .status_manager import save_status_json

WORKFLOW_NAME = "preaudit-alignment"
STAGE_NAME = "alignment"


def _find_source_audio(intake_dir: Path) -> Path:
    candidates = sorted(p for p in intake_dir.iterdir() if p.is_file() and p.name.startswith("source_audio."))
    if not candidates:
        raise FileNotFoundError("No se encontró source_audio en 00_intake/")
    return candidates[0]


def run_preaudit_alignment() -> int:
    should_run, runtime_info = preaudit_runtime_gate("preaudit_alignment")
    if not should_run:
        print("Runtime control: skip alignment.")
        return 0

    ctx = find_episode_for_workflow(WORKFLOW_NAME)
    if ctx is None:
        print("No hay episodio elegible para alignment.")
        return 0

    layout, status_payload, report_payload = init_episode_context(ctx, WORKFLOW_NAME, STAGE_NAME, "align_text_speakers")

    try:
        append_report_note(report_payload, f"Runtime control workflow=preaudit_alignment decision={runtime_info.get('decision')}")

        alignment_meta = assign_speakers_to_transcript_segments(
            layout["transcription"] / "whisperx_aligned_segments.json",
            layout["diarization"] / "speaker_segments.json",
            layout["alignment"] / "aligned_transcript_segments.json",
            layout["alignment"] / "doubtful_segments.json",
        )
        append_report_note(report_payload, f"Cruce transcripción+diarización generado ({alignment_meta['num_aligned_segments']} segmentos alineados)")
        append_report_note(report_payload, f"Segmentos dudosos detectados: {alignment_meta['num_doubtful_segments']}")

        status_payload["current_step"] = "redecode_doubtful"
        save_status_json(ctx.work_dir, status_payload)

        audio_path = _find_source_audio(layout["intake"])
        redecoded_meta = redecode_doubtful_segments(
            audio_path,
            layout["alignment"] / "doubtful_segments.json",
            layout["temp"],
            layout["alignment"] / "redecoded_segments.json",
        )
        append_report_note(report_payload, f"Segunda pasada completada ({redecoded_meta['num_redecoded_segments']} segmentos reprocesados)")

        status_payload["current_step"] = "build_global_srt"
        save_status_json(ctx.work_dir, status_payload)

        num_segments = write_full_transcript_with_speakers(
            layout["alignment"] / "aligned_transcript_segments.json",
            layout["alignment"] / "full_transcript_speakers.srt",
        )
        append_report_note(report_payload, f"SRT con hablantes generado ({num_segments} segmentos)")

        finalize_workflow_success(ctx, WORKFLOW_NAME, STAGE_NAME, layout, status_payload, report_payload)
        print(f"Alignment completado para {ctx.episode.id}")
        return 0

    except Exception as exc:
        finalize_workflow_failure(
            ctx, WORKFLOW_NAME, STAGE_NAME, status_payload.get("current_step", "align_text_speakers"),
            layout, status_payload, report_payload, exc
        )
        raise
