from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import REVIEWS_DIR, WORK_DIR
from .episode_queue import load_pending_queue
from .models import EpisodeEntry
from .status_manager import load_status_json, save_status_json


@dataclass
class EpisodeContext:
    episode: EpisodeEntry
    work_dir: Path
    review_dir: Path
    status_payload: dict


def _first_matching_audio_file(intake_dir: Path) -> Path | None:
    if not intake_dir.exists():
        return None

    candidates = sorted(
        p
        for p in intake_dir.iterdir()
        if p.is_file() and p.name.startswith("source_audio.")
    )
    return candidates[0] if candidates else None


def is_intake_done(work_dir: Path, review_dir: Path) -> tuple[bool, dict]:
    intake_dir = work_dir / "00_intake"
    source_audio = _first_matching_audio_file(intake_dir)
    source_metadata = intake_dir / "source_metadata.json"
    audit_yaml = review_dir / "audit.yaml"

    artifacts = {
        "source_audio": str(source_audio.relative_to(work_dir))
        if source_audio
        else None,
        "source_metadata": str(source_metadata.relative_to(work_dir))
        if source_metadata.exists()
        else None,
    }

    done = source_audio is not None and source_metadata.exists() and audit_yaml.exists()
    return done, artifacts


def is_transcription_done(work_dir: Path) -> tuple[bool, dict]:
    tdir = work_dir / "01_transcription"
    required = {
        "vad_segments": tdir / "vad_segments.json",
        "whisper_raw_segments": tdir / "whisper_raw_segments.json",
        "whisperx_aligned_segments": tdir / "whisperx_aligned_segments.json",
        "transcript_segments": tdir / "transcript_segments.json",
    }
    optional_summaries = [
        tdir / "full_transcript.txt",
        tdir / "full_transcript.srt",
    ]

    artifacts = {
        key: str(path.relative_to(work_dir)) if path.exists() else None
        for key, path in required.items()
    }

    done = all(path.exists() for path in required.values()) and all(
        path.exists() for path in optional_summaries
    )
    return done, artifacts


def is_diarization_done(work_dir: Path) -> tuple[bool, dict]:
    ddir = work_dir / "02_diarization"
    speaker_segments = ddir / "speaker_segments.json"
    speaker_timeline = ddir / "speaker_timeline.rttm"

    artifacts = {
        "speaker_segments": str(speaker_segments.relative_to(work_dir))
        if speaker_segments.exists()
        else None,
        "speaker_timeline": str(speaker_timeline.relative_to(work_dir))
        if speaker_timeline.exists()
        else None,
    }

    done = speaker_segments.exists() and speaker_timeline.exists()
    return done, artifacts


def is_alignment_done(work_dir: Path) -> tuple[bool, dict]:
    adir = work_dir / "03_alignment"
    aligned = adir / "aligned_transcript_segments.json"
    doubtful = adir / "doubtful_segments.json"
    redecoded = adir / "redecoded_segments.json"
    speakers_srt = adir / "full_transcript_speakers.srt"

    artifacts = {
        "aligned_transcript_segments": str(aligned.relative_to(work_dir))
        if aligned.exists()
        else None,
        "doubtful_segments": str(doubtful.relative_to(work_dir))
        if doubtful.exists()
        else None,
        "redecoded_segments": str(redecoded.relative_to(work_dir))
        if redecoded.exists()
        else None,
        "full_transcript_speakers": str(speakers_srt.relative_to(work_dir))
        if speakers_srt.exists()
        else None,
    }

    done = (
        aligned.exists()
        and doubtful.exists()
        and redecoded.exists()
        and speakers_srt.exists()
    )
    return done, artifacts


def is_review_prepare_done(work_dir: Path) -> tuple[bool, dict]:
    review_dir = work_dir / "04_review"
    speakers_auto = review_dir / "speakers_auto"
    speakers_reviewed = review_dir / "speakers_reviewed"

    outputs_dir = work_dir / "05_episode_outputs"
    summary = outputs_dir / "summary.md"
    outline = outputs_dir / "outline.md"
    episode_json = outputs_dir / "episode.json"

    logs_dir = work_dir / "06_logs"
    report_json = logs_dir / "report.json"

    artifacts = {
        "speakers_auto_dir": str(speakers_auto.relative_to(work_dir))
        if speakers_auto.exists()
        else None,
        "speakers_reviewed_dir": str(speakers_reviewed.relative_to(work_dir))
        if speakers_reviewed.exists()
        else None,
        "episode_json": str(episode_json.relative_to(work_dir))
        if episode_json.exists()
        else None,
        "report_json": str(report_json.relative_to(work_dir))
        if report_json.exists()
        else None,
    }

    done = (
        speakers_auto.exists()
        and speakers_reviewed.exists()
        and summary.exists()
        and outline.exists()
        and episode_json.exists()
        and report_json.exists()
    )
    return done, artifacts


def reconcile_status_with_filesystem(episode: EpisodeEntry) -> EpisodeContext:
    work_dir = WORK_DIR / episode.id
    review_dir = REVIEWS_DIR / episode.id

    payload = load_status_json(work_dir, episode.id)

    intake_done, intake_artifacts = is_intake_done(work_dir, review_dir)
    transcription_done, transcription_artifacts = is_transcription_done(work_dir)
    diarization_done, diarization_artifacts = is_diarization_done(work_dir)
    alignment_done, alignment_artifacts = is_alignment_done(work_dir)
    review_prepare_done, review_artifacts = is_review_prepare_done(work_dir)

    payload["stage_outputs"]["intake_done"] = intake_done
    payload["stage_outputs"]["transcription_done"] = transcription_done
    payload["stage_outputs"]["diarization_done"] = diarization_done
    payload["stage_outputs"]["alignment_done"] = alignment_done
    payload["stage_outputs"]["review_prepare_done"] = review_prepare_done

    payload["artifacts"].update(intake_artifacts)
    payload["artifacts"].update(transcription_artifacts)
    payload["artifacts"].update(diarization_artifacts)
    payload["artifacts"].update(alignment_artifacts)
    payload["artifacts"].update(review_artifacts)

    if review_prepare_done:
        payload["preaudit_stage"] = "review_prepare"
    elif alignment_done:
        payload["preaudit_stage"] = "alignment"
    elif diarization_done:
        payload["preaudit_stage"] = "diarization"
    elif transcription_done:
        payload["preaudit_stage"] = "transcription"
    elif intake_done:
        payload["preaudit_stage"] = "intake"
    else:
        payload["preaudit_stage"] = "not_started"

    save_status_json(work_dir, payload)

    return EpisodeContext(
        episode=episode,
        work_dir=work_dir,
        review_dir=review_dir,
        status_payload=payload,
    )


def _processing_contexts() -> list[EpisodeContext]:
    items = load_pending_queue()
    contexts: list[EpisodeContext] = []
    for item in items:
        if item.status == "processing":
            contexts.append(reconcile_status_with_filesystem(item))
    return contexts


def find_episode_for_intake() -> EpisodeContext | None:
    items = load_pending_queue()

    # No abrir episodios nuevos si ya hay uno en curso.
    if any(item.status == "processing" for item in items):
        return None

    for item in items:
        if item.status == "pending":
            return reconcile_status_with_filesystem(item)
    return None


def find_episode_for_transcription() -> EpisodeContext | None:
    for ctx in _processing_contexts():
        s = ctx.status_payload["stage_outputs"]
        if s["intake_done"] and not s["transcription_done"]:
            return ctx
    return None


def find_episode_for_diarization() -> EpisodeContext | None:
    for ctx in _processing_contexts():
        s = ctx.status_payload["stage_outputs"]
        if s["transcription_done"] and not s["diarization_done"]:
            return ctx
    return None


def find_episode_for_alignment() -> EpisodeContext | None:
    for ctx in _processing_contexts():
        s = ctx.status_payload["stage_outputs"]
        if (
            s["transcription_done"]
            and s["diarization_done"]
            and not s["alignment_done"]
        ):
            return ctx
    return None


def find_episode_for_review_prepare() -> EpisodeContext | None:
    for ctx in _processing_contexts():
        s = ctx.status_payload["stage_outputs"]
        if s["alignment_done"] and not s["review_prepare_done"]:
            return ctx
    return None


def find_episode_for_workflow(workflow_name: str) -> EpisodeContext | None:
    dispatch: dict[str, Callable[[], EpisodeContext | None]] = {
        "preaudit-intake": find_episode_for_intake,
        "preaudit-transcription": find_episode_for_transcription,
        "preaudit-diarization": find_episode_for_diarization,
        "preaudit-alignment": find_episode_for_alignment,
        "preaudit-review-prepare": find_episode_for_review_prepare,
    }

    finder = dispatch.get(workflow_name)
    if finder is None:
        raise ValueError(f"Workflow no soportado para selección: {workflow_name}")
    return finder()
