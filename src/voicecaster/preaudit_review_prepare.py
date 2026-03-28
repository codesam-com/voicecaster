from __future__ import annotations

from pathlib import Path

from .episode_queue import update_episode_status
from .preaudit_common import (
    append_report_note,
    finalize_workflow_failure,
    finalize_workflow_success,
    init_episode_context,
    init_speakers_reviewed,
    preaudit_runtime_gate,
    safe_unlink,
)
from .reporting import write_json
from .speaker_alignment import write_per_speaker_srts
from .stage_selector import find_episode_for_workflow
from .status_manager import save_status_json

WORKFLOW_NAME = "preaudit-review-prepare"
STAGE_NAME = "review_prepare"


def _find_source_audio(intake_dir: Path) -> Path | None:
    candidates = sorted(p for p in intake_dir.iterdir() if p.is_file() and p.name.startswith("source_audio."))
    return candidates[0] if candidates else None


def run_preaudit_review_prepare() -> int:
    should_run, runtime_info = preaudit_runtime_gate("preaudit_review_prepare")
    if not should_run:
        print("Runtime control: skip review prepare.")
        return 0

    ctx = find_episode_for_workflow(WORKFLOW_NAME)
    if ctx is None:
        print("No hay episodio elegible para review prepare.")
        return 0

    layout, status_payload, report_payload = init_episode_context(ctx, WORKFLOW_NAME, STAGE_NAME, "speaker_srts")

    try:
        append_report_note(report_payload, f"Runtime control workflow=preaudit_review_prepare decision={runtime_info.get('decision')}")

        speakers_auto_dir = layout["review"] / "speakers_auto"
        speakers_reviewed_dir = layout["review"] / "speakers_reviewed"

        per_speaker_outputs = write_per_speaker_srts(
            layout["alignment"] / "aligned_transcript_segments.json",
            speakers_auto_dir,
        )
        append_report_note(report_payload, f"SRT automáticos por hablante generados ({len(per_speaker_outputs)} archivos)")

        reviewed_initialized = init_speakers_reviewed(speakers_auto_dir, speakers_reviewed_dir)
        if reviewed_initialized:
            append_report_note(report_payload, "speakers_reviewed inicializado desde speakers_auto/")
        else:
            append_report_note(report_payload, "speakers_reviewed ya existía y no se sobrescribió")

        status_payload["current_step"] = "episode_outputs"
        save_status_json(ctx.work_dir, status_payload)

        # mínimo viable: outputs finales de episodio
        (layout["episode_outputs"] / "summary.md").write_text(
            "# Summary\n\nPREAUDITORÍA completada y lista para revisión humana.\n",
            encoding="utf-8",
        )
        (layout["episode_outputs"] / "outline.md").write_text(
            "# Outline\n\n1. Ingesta\n2. Transcripción\n3. Diarización\n4. Alineado\n5. Revisión humana preparada\n",
            encoding="utf-8",
        )

        episode_payload = {
            "episode_id": ctx.episode.id,
            "podcast_title": ctx.episode.podcast_title,
            "episode_title": ctx.episode.episode_title,
            "status_after_preaudit": "pending_review",
            "review_structure": {
                "speakers_auto_dir": "04_review/speakers_auto",
                "speakers_reviewed_dir": "04_review/speakers_reviewed",
                "reviewed_initialized_from_auto": reviewed_initialized,
            },
        }
        write_json(layout["episode_outputs"] / "episode.json", episode_payload)

        # limpieza final de audio temporal
        safe_unlink(layout["temp"] / "audio_mono_16k.wav")
        safe_unlink(_find_source_audio(layout["intake"]))

        report_payload["result"] = "pending_review"
        report_payload["finished_at"] = report_payload.get("finished_at")
        write_json(layout["logs"] / "report.json", report_payload)

        update_episode_status(ctx.episode.id, "pending_review")

        finalize_workflow_success(ctx, WORKFLOW_NAME, STAGE_NAME, layout, status_payload, report_payload)
        print(f"Review prepare completado para {ctx.episode.id} -> pending_review")
        return 0

    except Exception as exc:
        finalize_workflow_failure(
            ctx, WORKFLOW_NAME, STAGE_NAME, status_payload.get("current_step", "speaker_srts"),
            layout, status_payload, report_payload, exc
        )
        raise
