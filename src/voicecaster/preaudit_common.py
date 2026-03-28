from __future__ import annotations

import shutil
import traceback
from datetime import UTC, datetime
from pathlib import Path

from .config import RUNTIME_CONTROL_PATH
from .reporting import utc_now_iso, write_json
from .runtime_control import should_run_now, update_runtime_control
from .status_manager import (
    load_status_json,
    mark_workflow_completed,
    mark_workflow_failed,
    mark_workflow_started,
    save_status_json,
)
from .work_layout import ensure_work_layout
from .yaml_io import write_yaml


def preaudit_runtime_gate(workflow_name: str) -> tuple[bool, dict]:
    return should_run_now(RUNTIME_CONTROL_PATH, workflow_name)


def _report_partial_result_from_stage(stage_name: str) -> str:
    mapping = {
        "intake": "intake_completed",
        "transcription": "transcription_completed",
        "diarization": "diarization_completed",
        "alignment": "alignment_completed",
        "review_prepare": "pending_review",
    }
    return mapping.get(stage_name, "in_progress")


def _find_source_audio_anywhere(intake_dir: Path) -> Path | None:
    if not intake_dir.exists():
        return None

    candidates = sorted(
        p for p in intake_dir.iterdir()
        if p.is_file() and p.name.startswith("source_audio")
    )
    return candidates[0] if candidates else None


def _sync_artifacts_into_status(
    work_dir: Path,
    review_dir: Path,
    status_payload: dict,
) -> dict:
    intake_dir = work_dir / "00_intake"
    transcription_dir = work_dir / "01_transcription"
    diarization_dir = work_dir / "02_diarization"
    alignment_dir = work_dir / "03_alignment"
    review_phase_dir = work_dir / "04_review"
    outputs_dir = work_dir / "05_episode_outputs"
    logs_dir = work_dir / "06_logs"

    source_audio = _find_source_audio_anywhere(intake_dir)
    source_metadata = intake_dir / "source_metadata.json"

    artifacts = status_payload.setdefault("artifacts", {})

    def rel(path: Path | None) -> str | None:
        if path is None or not path.exists():
            return None
        return str(path.relative_to(work_dir))

    artifacts["source_audio"] = rel(source_audio)
    artifacts["source_metadata"] = rel(source_metadata)

    artifacts["vad_segments"] = rel(transcription_dir / "vad_segments.json")
    artifacts["whisper_raw_segments"] = rel(transcription_dir / "whisper_raw_segments.json")
    artifacts["whisperx_aligned_segments"] = rel(transcription_dir / "whisperx_aligned_segments.json")
    artifacts["transcript_segments"] = rel(transcription_dir / "transcript_segments.json")

    artifacts["speaker_segments"] = rel(diarization_dir / "speaker_segments.json")
    artifacts["speaker_timeline"] = rel(diarization_dir / "speaker_timeline.rttm")

    artifacts["aligned_transcript_segments"] = rel(alignment_dir / "aligned_transcript_segments.json")
    artifacts["doubtful_segments"] = rel(alignment_dir / "doubtful_segments.json")
    artifacts["redecoded_segments"] = rel(alignment_dir / "redecoded_segments.json")
    artifacts["full_transcript_speakers"] = rel(alignment_dir / "full_transcript_speakers.srt")

    artifacts["speakers_auto_dir"] = (
        rel(review_phase_dir / "speakers_auto")
        if (review_phase_dir / "speakers_auto").exists()
        else None
    )
    artifacts["speakers_reviewed_dir"] = (
        rel(review_phase_dir / "speakers_reviewed")
        if (review_phase_dir / "speakers_reviewed").exists()
        else None
    )

    artifacts["episode_json"] = rel(outputs_dir / "episode.json")
    artifacts["report_json"] = rel(logs_dir / "report.json")

    return status_payload


def init_episode_context(ctx, workflow_name: str, stage_name: str, step_name: str) -> tuple[dict, dict, dict]:
    work_dir = ctx.work_dir
    review_dir = ctx.review_dir
    review_dir.mkdir(parents=True, exist_ok=True)

    layout = ensure_work_layout(work_dir, ctx.episode.id)
    status_payload = load_status_json(work_dir, ctx.episode.id)
    status_payload = mark_workflow_started(status_payload, workflow_name, stage_name, step_name)
    status_payload = _sync_artifacts_into_status(work_dir, review_dir, status_payload)
    save_status_json(work_dir, status_payload)

    report_path = layout["logs"] / "report.json"
    if report_path.exists():
        import json
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report_payload = {
            "episode_id": ctx.episode.id,
            "phase": "preaudit",
            "started_at": status_payload.get("started_at") or utc_now_iso(),
            "finished_at": None,
            "result": "in_progress",
            "notes": [],
            "workflow_runs": [],
            "source_url_original": str(ctx.episode.url),
            "source_url_normalized": None,
        }

    report_payload.setdefault("workflow_runs", [])
    report_payload["result"] = "in_progress"
    report_payload["finished_at"] = utc_now_iso()

    report_payload["workflow_runs"].append(
        {
            "workflow": workflow_name,
            "started_at": utc_now_iso(),
            "finished_at": None,
            "result": None,
        }
    )

    return layout, status_payload, report_payload


def append_report_note(report_payload: dict, note: str) -> None:
    report_payload.setdefault("notes", []).append(note)


def finalize_workflow_success(
    ctx,
    workflow_name: str,
    stage_name: str,
    layout: dict,
    status_payload: dict,
    report_payload: dict,
) -> None:
    finished_at = utc_now_iso()

    if report_payload.get("workflow_runs"):
        report_payload["workflow_runs"][-1]["finished_at"] = finished_at
        report_payload["workflow_runs"][-1]["result"] = "ok"

    report_payload["finished_at"] = finished_at
    report_payload["result"] = _report_partial_result_from_stage(stage_name)

    write_json(layout["logs"] / "report.json", report_payload)

    status_payload = mark_workflow_completed(status_payload, workflow_name, stage_name)
    status_payload = _sync_artifacts_into_status(ctx.work_dir, ctx.review_dir, status_payload)
    save_status_json(ctx.work_dir, status_payload)

    started_at = status_payload.get("started_at")
    if started_at:
        started_dt = datetime.fromisoformat(started_at)
        finished_dt = datetime.now(UTC)
        duration_run = (finished_dt - started_dt).total_seconds()
        update_runtime_control(
            RUNTIME_CONTROL_PATH,
            workflow_name.replace("-", "_"),
            duration_run,
            finished_dt,
        )


def finalize_workflow_failure(
    ctx,
    workflow_name: str,
    stage_name: str,
    step_name: str,
    layout: dict,
    status_payload: dict,
    report_payload: dict,
    exc: Exception,
) -> None:
    append_report_note(report_payload, f"Excepción no controlada: {exc}")
    append_report_note(report_payload, traceback.format_exc())

    finished_at = utc_now_iso()

    if report_payload.get("workflow_runs"):
        report_payload["workflow_runs"][-1]["finished_at"] = finished_at
        report_payload["workflow_runs"][-1]["result"] = "failed"

    report_payload["result"] = "failed"
    report_payload["finished_at"] = finished_at
    write_json(layout["logs"] / "report.json", report_payload)

    status_payload = mark_workflow_failed(status_payload, workflow_name, stage_name, step_name, str(exc))
    status_payload = _sync_artifacts_into_status(ctx.work_dir, ctx.review_dir, status_payload)
    save_status_json(ctx.work_dir, status_payload)


def ensure_audit_yaml(review_dir: Path) -> Path:
    audit_path = review_dir / "audit.yaml"
    if not audit_path.exists():
        write_yaml(
            audit_path,
            {
                "identity_review_done": False,
                "srt_audit_done": False,
                "approved_as_source_of_truth": False,
                "speaker_mapping_final": {},
            },
        )
    return audit_path


def safe_unlink(path: Path | None) -> None:
    try:
        if path is not None and path.exists():
            path.unlink()
    except Exception:
        pass


def init_speakers_reviewed(speakers_auto_dir: Path, speakers_reviewed_dir: Path) -> bool:
    if not speakers_reviewed_dir.exists():
        shutil.copytree(speakers_auto_dir, speakers_reviewed_dir)
        return True
    return False
