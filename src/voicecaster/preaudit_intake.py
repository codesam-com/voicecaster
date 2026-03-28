from __future__ import annotations

from .audio_probe import AudioProbeError, probe_audio_file
from .downloader import DownloadError, IncompatibleSourceError, download_audio_to_workdir
from .episode_queue import update_episode_status
from .preaudit_common import (
    append_report_note,
    ensure_audit_yaml,
    finalize_workflow_failure,
    finalize_workflow_success,
    init_episode_context,
    preaudit_runtime_gate,
)
from .reporting import write_json
from .stage_selector import find_episode_for_workflow
from .url_resolver import normalize_download_url

WORKFLOW_NAME = "preaudit-intake"
STAGE_NAME = "intake"


def run_preaudit_intake() -> int:
    should_run, runtime_info = preaudit_runtime_gate("preaudit_intake")
    if not should_run:
        print("Runtime control: skip intake.")
        return 0

    ctx = find_episode_for_workflow(WORKFLOW_NAME)
    if ctx is None:
        print("No hay episodio elegible para intake.")
        return 0

    layout, status_payload, report_payload = init_episode_context(ctx, WORKFLOW_NAME, STAGE_NAME, "download")

    try:
        update_episode_status(ctx.episode.id, "processing")

        normalized_url = normalize_download_url(str(ctx.episode.url))
        report_payload["source_url_normalized"] = normalized_url
        append_report_note(report_payload, f"Runtime control workflow=preaudit_intake decision={runtime_info.get('decision')}")

        audio_path = download_audio_to_workdir(normalized_url, layout["intake"], "source_audio")
        append_report_note(report_payload, f"Audio descargado en: {audio_path.name}")

        status_payload["current_step"] = "probe_audio"
        from .status_manager import save_status_json
        save_status_json(ctx.work_dir, status_payload)

        audio_probe = probe_audio_file(audio_path)
        append_report_note(report_payload, "Audio validado con ffprobe.")

        write_json(
            layout["intake"] / "source_metadata.json",
            {
                "episode_id": ctx.episode.id,
                "podcast_title": ctx.episode.podcast_title,
                "episode_title": ctx.episode.episode_title,
                "source_url_original": str(ctx.episode.url),
                "source_url_normalized": normalized_url,
                "audio_probe": audio_probe,
            },
        )

        audit_path = ensure_audit_yaml(ctx.review_dir)
        append_report_note(report_payload, f"audit.yaml listo en: {audit_path}")

        finalize_workflow_success(ctx, WORKFLOW_NAME, STAGE_NAME, layout, status_payload, report_payload)
        print(f"Intake completado para {ctx.episode.id}")
        return 0

    except (IncompatibleSourceError, DownloadError, AudioProbeError, Exception) as exc:
        finalize_workflow_failure(
            ctx, WORKFLOW_NAME, STAGE_NAME, status_payload.get("current_step", "download"),
            layout, status_payload, report_payload, exc
        )
        raise
