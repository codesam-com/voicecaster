from __future__ import annotations

from .archive_utils import create_zip_archive
from .audio_probe import AudioProbeError, probe_audio_file
from .config import MAX_PROCESSING_RETRIES, REVIEWS_DIR, WORK_DIR
from .downloader import DownloadError, IncompatibleSourceError, download_audio_to_workdir
from .episode_queue import reserve_next_pending_episode, update_episode_status
from .reporting import utc_now_iso, write_json
from .url_resolver import normalize_download_url
from .yaml_io import write_yaml


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

        audit_path = review_dir / "audit.yaml"
        write_yaml(
            audit_path,
            {
                "identity_review_done": False,
                "srt_audit_done": False,
                "approved_as_source_of_truth": False,
                "speaker_mapping_final": {},
            },
        )

        duration_seconds = audio_probe.get("duration_seconds")
        duration_text = (
            f"{duration_seconds:.2f} segundos"
            if isinstance(duration_seconds, (int, float))
            else "desconocida"
        )

        (work_dir / "summary.md").write_text(
            (
                "# Summary\n\n"
                "PREAUDITORÍA técnica completada.\n\n"
                f"- Episodio: {episode.episode_title}\n"
                f"- Podcast: {episode.podcast_title}\n"
                f"- Duración detectada: {duration_text}\n"
                "- Estado: pendiente de integración ASR/diarización real\n"
            ),
            encoding="utf-8",
        )

        (work_dir / "outline.md").write_text(
            (
                "# Outline\n\n"
                "1. Descarga verificada\n"
                "2. Validación técnica del audio\n"
                "3. Pendiente transcripción completa\n"
                "4. Pendiente diarización\n"
                "5. Pendiente propuesta real de identidades\n"
            ),
            encoding="utf-8",
        )

        (work_dir / "full_transcript.srt").write_text(
            "1\n"
            "00:00:00,000 --> 00:00:02,000\n"
            "[speaker_01] Placeholder de transcripción.\n",
            encoding="utf-8",
        )

        speaker_dir = work_dir / "speakers"
        speaker_dir.mkdir(parents=True, exist_ok=True)

        (speaker_dir / "speaker_01.srt").write_text(
            "1\n"
            "00:00:00,000 --> 00:00:02,000\n"
            "Placeholder de intervención.\n",
            encoding="utf-8",
        )

        episode_payload = {
            "episode_id": episode.id,
            "podcast_title": episode.podcast_title,
            "episode_title": episode.episode_title,
            "source_url_original": str(episode.url),
            "source_url_normalized": normalized_url,
            "status_after_preaudit": "pending_review",
            "downloaded_audio_filename": audio_path.name,
            "audio_probe": audio_probe,
            "speaker_candidates": [
                {
                    "speaker_id": "speaker_01",
                    "inferred_name": "pendiente_de_validar",
                    "confidence": 0.0,
                    "alternatives": [],
                }
            ],
            "duration_seconds": audio_probe.get("duration_seconds"),
            "topics_detected": [],
            "participants_declared": episode.participants or [],
        }
        write_json(work_dir / "episode.json", episode_payload)

        archive_path = create_zip_archive(work_dir, work_dir / f"{episode.id}_preaudit.zip")
        report_payload["notes"].append(f"Archivo comprimido generado: {archive_path.name}")

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
        next_retry_count = episode.retries + 1

        report_payload["result"] = "failed"
        report_payload["finished_at"] = utc_now_iso()
        report_payload["notes"].append(str(exc))
        write_json(work_dir / "report.json", report_payload)

        should_retry = next_retry_count < MAX_PROCESSING_RETRIES
        update_episode_status(
            episode.id,
            "failed",
            increment_retries=True,
        )

        if should_retry:
            print(f"{episode.id}: fallo técnico recuperable -> failed (retry pendiente)")
        else:
            print(f"{episode.id}: fallo técnico -> failed (sin más reintentos)")
        return 1

    except Exception as exc:
        report_payload["result"] = "failed"
        report_payload["finished_at"] = utc_now_iso()
        report_payload["notes"].append(f"Excepción no controlada: {exc}")
        write_json(work_dir / "report.json", report_payload)

        update_episode_status(episode.id, "failed", increment_retries=True)
        print(f"{episode.id}: excepción no controlada -> failed")
        return 1
