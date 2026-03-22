from __future__ import annotations

from pathlib import Path

from .archive_utils import create_zip_archive
from .audio_probe import AudioProbeError, probe_audio_file
from .config import MAX_PROCESSING_RETRIES, REVIEWS_DIR, WORK_DIR
from .downloader import DownloadError, IncompatibleSourceError, download_audio_to_workdir
from .episode_queue import reserve_next_pending_episode, update_episode_status
from .reporting import utc_now_iso, write_json
from .transcriber import transcribe_to_srt
from .url_resolver import normalize_download_url
from .yaml_io import write_yaml


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_").lower()


def _write_basic_summary(
    work_dir: Path,
    podcast_title: str,
    episode_title: str,
    duration_seconds: float | None,
    detected_language: str | None,
) -> None:
    duration_text = (
        f"{duration_seconds:.2f} segundos" if isinstance(duration_seconds, float) else "desconocida"
    )
    language_text = detected_language or "desconocido"

    (work_dir / "summary.md").write_text(
        (
            "# Summary\n\n"
            "PREAUDITORÍA completada.\n\n"
            f"- Podcast: {podcast_title}\n"
            f"- Episodio: {episode_title}\n"
            f"- Duración detectada: {duration_text}\n"
            f"- Idioma detectado: {language_text}\n"
            "- Estado: pendiente de diarización e identificación real de hablantes\n"
        ),
        encoding="utf-8",
    )


def _write_basic_outline(work_dir: Path) -> None:
    (work_dir / "outline.md").write_text(
        (
            "# Outline\n\n"
            "1. Descarga validada\n"
            "2. Audio validado técnicamente con ffprobe\n"
            "3. Transcripción global generada\n"
            "4. Pendiente diarización\n"
            "5. Pendiente propuesta real de identidades por hablante\n"
            "6. Pendiente generación de `speaker_<id>.srt`\n"
        ),
        encoding="utf-8",
    )


def run_preaudit() -> int:
    print("Iniciando PREAUDITORÍA...")

    episode = reserve_next_pending_episode()
    if episode is None:
        print("No hay episodios pendientes.")
        return 0

    print(f"Episodio reservado: {episode.id}")

    work_dir = WORK_DIR / episode.id
    review_dir = REVIEWS_DIR / episode.id
    work_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now_iso()
    normalized_url = normalize_download_url(str(episode.url))
    audit_path = review_dir / "audit.yaml"

    report_payload = {
        "episode_id": episode.id,
        "phase": "preaudit",
        "started_at": started_at,
        "finished_at": None,
        "result": None,
        "notes": [],
        "source_url_original": str(episode.url),
        "source_url_normalized": normalized_url,
    }

    try:
        audio_path = download_audio_to_workdir(normalized_url, work_dir, episode.id)
        report_payload["notes"].append(f"Audio descargado en: {audio_path.name}")

        audio_probe = probe_audio_file(audio_path)
        report_payload["notes"].append("Audio validado con ffprobe.")

        transcription_meta = transcribe_to_srt(
            audio_path=audio_path,
            output_srt=work_dir / "full_transcript.srt",
        )
        report_payload["notes"].append(
            f"Transcripción generada ({transcription_meta['num_segments']} segmentos)."
        )

        write_yaml(
            audit_path,
            {
                "identity_review_done": False,
                "srt_audit_done": False,
                "approved_as_source_of_truth": False,
                "speaker_mapping_final": {},
            },
        )

        speakers_dir = work_dir / "speakers"
        speakers_dir.mkdir(exist_ok=True)

        detected_language = transcription_meta.get("language")
        duration_seconds = audio_probe.get("duration_seconds")

        _write_basic_summary(
            work_dir=work_dir,
            podcast_title=episode.podcast_title,
            episode_title=episode.episode_title,
            duration_seconds=duration_seconds,
            detected_language=detected_language,
        )
        _write_basic_outline(work_dir)

        episode_payload = {
            "episode_id": episode.id,
            "podcast_title": episode.podcast_title,
            "episode_title": episode.episode_title,
            "source_url_original": str(episode.url),
            "source_url_normalized": normalized_url,
            "status_after_preaudit": "pending_review",
            "downloaded_audio_filename": audio_path.name,
            "audio_probe": audio_probe,
            "language_detected": detected_language,
            "transcription": {
                "num_segments": transcription_meta.get("num_segments"),
                "speaker_diarization_applied": False,
                "speaker_level_outputs_generated": False,
            },
            "speaker_candidates": [],
            "duration_seconds": duration_seconds,
            "topics_detected": [],
            "participants_declared": episode.participants or [],
        }
        write_json(work_dir / "episode.json", episode_payload)

        archive_path = create_zip_archive(work_dir, work_dir / f"{episode.id}_preaudit.zip")
        report_payload["notes"].append(f"Archivo comprimido generado: {archive_path.name}")
        report_payload["notes"].append(
            "Aún no se generaron salidas por hablante: diarización pendiente."
        )

        report_payload["result"] = "pending_review"
        report_payload["finished_at"] = utc_now_iso()
        write_json(work_dir / "report.json", report_payload)

        update_episode_status(episode.id, "pending_review")

        print(f"Archivo de auditoría: {audit_path}")
        print(f"Directorio de trabajo: {work_dir}")
        print(f"Report: {work_dir / 'report.json'}")
        print(f"PREAUDITORÍA completada para {episode.id} -> pending_review")
        return 0

    except IncompatibleSourceError as exc:
        report_payload["result"] = "incompatible"
        report_payload["finished_at"] = utc_now_iso()
        report_payload["notes"].append(str(exc))
        write_json(work_dir / "report.json", report_payload)

        update_episode_status(episode.id, "incompatible")
        print(f"{episode.id}: fuente incompatible -> incompatible")
        return 0

    except (DownloadError, AudioProbeError) as exc:
        report_payload["result"] = "failed"
        report_payload["finished_at"] = utc_now_iso()
        report_payload["notes"].append(str(exc))
        write_json(work_dir / "report.json", report_payload)

        update_episode_status(episode.id, "failed", increment_retries=True)
        print(f"{episode.id}: fallo técnico -> failed")
        return 1

    except Exception as exc:
        report_payload["result"] = "failed"
        report_payload["finished_at"] = utc_now_iso()
        report_payload["notes"].append(f"Excepción no controlada: {exc}")
        write_json(work_dir / "report.json", report_payload)

        update_episode_status(episode.id, "failed", increment_retries=True)
        print(f"{episode.id}: excepción no controlada -> failed")
        return 1
