# src/voicecaster/intake/runner.py
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import requests

from .detect_duplicates import detect_and_mark_duplicate_intake_ids
from .download_audio import DownloadContentError, download_audio
from .error_classification import build_error_payload, classify_exception
from .fs_paths import (
    get_episode_cleanup_path,
    get_episode_error_path,
    get_episode_events_log_path,
    get_episode_intake_result_path,
    get_episode_normalized_source_path,
    get_episode_report_path,
    get_episode_request_path,
    get_episode_source_metadata_path,
    get_episode_status_path,
    get_episode_temp_audio_path,
)
from .fs_workdir import cleanup_temp_dir, create_episode_workdir_structure, reset_episode_workdir, write_workdir_readme
from .handle_result import apply_failure_result, apply_success_result
from .inputs_load_and_save import load_inputs, save_inputs
from .json_utils import write_json_atomic, write_json_file
from .logging_events import append_event
from .logging_report import build_initial_report, finalize_report, save_report
from .normalize_source import normalize_source
from .select_next_episode import select_next_intake_episode
from .time_utils import utc_now_iso
from .validate_audio import AudioValidationError, validate_audio


def write_status(
    status_path: Path,
    *,
    episode_id: str,
    status: str,
    current_step: str,
    started_at: str,
    result: str | None = None,
    finished_at: str | None = None,
) -> None:
    payload = {
        "episode_id": episode_id,
        "workflow": "intake",
        "status": status,
        "current_step": current_step,
        "started_at": started_at,
        "finished_at": finished_at,
        "result": result,
    }
    write_json_atomic(status_path, payload, indent=2)


def run() -> int:
    episodes = load_inputs()

    duplicate_changes = detect_and_mark_duplicate_intake_ids(episodes)
    if duplicate_changes:
        save_inputs(episodes)

    selected_index, episode = select_next_intake_episode(episodes)
    if episode is None or selected_index is None:
        return 0

    episode_id = str(episode["id"])
    started_at = utc_now_iso()

    reset_episode_workdir(episode_id)
    create_episode_workdir_structure(episode_id)
    write_workdir_readme(episode_id)

    status_path = get_episode_status_path(episode_id)
    request_path = get_episode_request_path(episode_id)
    normalized_source_path = get_episode_normalized_source_path(episode_id)
    source_metadata_path = get_episode_source_metadata_path(episode_id)
    intake_result_path = get_episode_intake_result_path(episode_id)
    cleanup_path = get_episode_cleanup_path(episode_id)
    events_log_path = get_episode_events_log_path(episode_id)
    report_path = get_episode_report_path(episode_id)
    error_path = get_episode_error_path(episode_id)
    temp_audio_path = get_episode_temp_audio_path(episode_id)

    write_status(
        status_path,
        episode_id=episode_id,
        status="intake",
        current_step="init",
        started_at=started_at,
    )

    append_event(
        events_log_path,
        "intake_started",
        episode_id=episode_id,
        selected_index=selected_index,
        duplicate_changes=duplicate_changes,
    )

    write_json_file(request_path, episode, indent=2)

    report = build_initial_report(episode_id, episode)
    save_report(report_path, report)

    cleanup_payload: dict[str, Any] = {
        "audio_deleted": False,
        "temp_dir_deleted": False,
        "finished_at": None,
    }

    try:
        write_status(
            status_path,
            episode_id=episode_id,
            status="intake",
            current_step="normalize_source",
            started_at=started_at,
        )

        normalized = normalize_source(str(episode["url"]))
        write_json_file(normalized_source_path, normalized, indent=2)

        report["source"]["url_normalized"] = normalized["url_normalized"]
        report["source"]["source_type"] = normalized["source_type"]
        save_report(report_path, report)

        append_event(
            events_log_path,
            "source_normalized",
            episode_id=episode_id,
            source_type=normalized["source_type"],
            url_normalized=normalized["url_normalized"],
        )

        write_status(
            status_path,
            episode_id=episode_id,
            status="intake",
            current_step="download_audio",
            started_at=started_at,
        )

        download_info = download_audio(normalized["url_normalized"], temp_audio_path)

        append_event(
            events_log_path,
            "audio_downloaded",
            episode_id=episode_id,
            final_url=download_info["final_url"],
            bytes_written=download_info["bytes_written"],
            content_type=download_info.get("content_type"),
        )

        write_status(
            status_path,
            episode_id=episode_id,
            status="intake",
            current_step="validate_audio",
            started_at=started_at,
        )

        validation = validate_audio(temp_audio_path)
        source_metadata = validation["source_metadata"]

        write_json_file(source_metadata_path, source_metadata, indent=2)

        # 🟡 NUEVO: guardar metadata en report
        report["audio"] = source_metadata
        save_report(report_path, report)

        append_event(
            events_log_path,
            "audio_validated",
            episode_id=episode_id,
            duration_seconds=source_metadata.get("duration_seconds"),
            audio_codec=source_metadata.get("audio_codec"),
            sample_rate=source_metadata.get("sample_rate"),
            channels=source_metadata.get("channels"),
        )

        result_info = apply_success_result(episode)
        save_inputs(episodes)

        write_json_file(
            intake_result_path,
            {
                **result_info,
                "message": "Intake completado correctamente.",
                "finished_at": utc_now_iso(),
            },
            indent=2,
        )

        finalize_report(report, result="success", note="Audio descargado, validado y limpiado.")
        save_report(report_path, report)

        # 🔴 CORRECCIÓN: status real del episodio
        write_status(
            status_path,
            episode_id=episode_id,
            status=episode["status"],  # ahora "transcript"
            current_step="done",
            started_at=started_at,
            result="success",
            finished_at=utc_now_iso(),
        )

        append_event(
            events_log_path,
            "intake_completed",
            episode_id=episode_id,
            status_after=episode.get("status"),
            retries_after=episode.get("retries"),
        )

        return 0

    except DownloadContentError as exc:
        payload = build_error_payload(
            message="La fuente respondió, pero el contenido no es utilizable para intake.",
            error_type="content",
            exception=exc,
        )
        return _handle_failure(
            episodes, episode, episode_id, started_at,
            status_path, error_path, intake_result_path,
            report, report_path, events_log_path, payload
        )

    except AudioValidationError as exc:
        payload = build_error_payload(
            message="El archivo descargado no es un audio trabajable.",
            error_type="content",
            exception=exc,
        )
        return _handle_failure(
            episodes, episode, episode_id, started_at,
            status_path, error_path, intake_result_path,
            report, report_path, events_log_path, payload
        )

    except requests.exceptions.RequestException as exc:
        payload = build_error_payload(
            message="Fallo de red durante intake.",
            error_type="network",
            exception=exc,
        )
        return _handle_failure(
            episodes, episode, episode_id, started_at,
            status_path, error_path, intake_result_path,
            report, report_path, events_log_path, payload
        )

    except Exception as exc:
        payload = build_error_payload(
            message="Excepción no controlada durante intake.",
            error_type=classify_exception(exc),
            exception=exc,
            extra={"traceback": traceback.format_exc()},
        )
        return _handle_failure(
            episodes, episode, episode_id, started_at,
            status_path, error_path, intake_result_path,
            report, report_path, events_log_path, payload
        )

    finally:
        audio_deleted = False
        if temp_audio_path.exists():
            temp_audio_path.unlink(missing_ok=True)
            audio_deleted = True

        cleanup_temp_dir(episode_id)

        cleanup_payload["audio_deleted"] = audio_deleted
        cleanup_payload["temp_dir_deleted"] = True
        cleanup_payload["finished_at"] = utc_now_iso()
        write_json_file(cleanup_path, cleanup_payload, indent=2)


def _handle_failure(
    episodes,
    episode,
    episode_id,
    started_at,
    status_path,
    error_path,
    intake_result_path,
    report,
    report_path,
    events_log_path,
    payload,
) -> int:
    error_type = payload["error_type"]
    result_info = apply_failure_result(episode, error_type)
    save_inputs(episodes)

    write_json_file(error_path, payload, indent=2)

    write_json_file(
        intake_result_path,
        {
            **result_info,
            "message": payload["message"],
            "finished_at": utc_now_iso(),
        },
        indent=2,
    )

    finalize_report(report, result="failure")
    save_report(report_path, report)

    write_status(
        status_path,
        episode_id=episode_id,
        status=episode.get("status"),
        current_step="done",
        started_at=started_at,
        result="failure",
        finished_at=utc_now_iso(),
    )

    append_event(
        events_log_path,
        "intake_failed",
        episode_id=episode_id,
        error_type=error_type,
        status_after=result_info["status_after"],
        retries_after=result_info["retries_after"],
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(run())
