from __future__ import annotations

import shutil
import traceback
from pathlib import Path

from .audio_normalize import normalize_audio_for_diarization
from .audio_probe import AudioProbeError, probe_audio_file
from .audio_publish import build_published_audio_filename, rewrite_audio_metadata
from .config import (
    DRIVE_PODCASTS_PENDING_FOLDER_ID,
    GOOGLE_SERVICE_ACCOUNT_JSON,
    HF_TOKEN,
    MAX_PROCESSING_RETRIES,
    REVIEWS_DIR,
    WORK_DIR,
)
from .diarizer import diarize_audio
from .downloader import DownloadError, IncompatibleSourceError, download_audio_to_workdir
from .drive_uploader import publish_audio_to_podcasts_pending
from .episode_queue import (
    reserve_next_pending_episode,
    update_episode_operational_audio_url,
    update_episode_status,
)
from .reporting import utc_now_iso, write_json
from .speaker_alignment import (
    assign_speakers_to_transcript_segments,
    write_full_transcript_with_speakers,
    write_per_speaker_srts,
)
from .transcriber import transcribe_bundle
from .url_resolver import normalize_download_url
from .yaml_io import write_yaml


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _initialize_reviewed_speakers(work_dir: Path, report_payload: dict) -> dict:
    speakers_auto_dir = work_dir / "speakers_auto"
    speakers_reviewed_dir = work_dir / "speakers_reviewed"

    initialized = False
    if not speakers_reviewed_dir.exists():
        shutil.copytree(speakers_auto_dir, speakers_reviewed_dir)
        initialized = True
        print("speakers_reviewed inicializado desde speakers_auto/")
        report_payload["notes"].append("speakers_reviewed inicializado desde speakers_auto/")
    else:
        print("speakers_reviewed ya existe, no se sobrescribe")
        report_payload["notes"].append("speakers_reviewed ya existía y no se sobrescribió")

    return {
        "speakers_auto_dir": speakers_auto_dir.name,
        "speakers_reviewed_dir": speakers_reviewed_dir.name,
        "reviewed_initialized_from_auto": initialized,
    }


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
        "source_url_original": str(episode.source_url_original) if episode.source_url_original else str(episode.url),
        "source_url_normalized": normalized_url,
    }

    audio_path: Path | None = None
    normalized_for_diarization: Path | None = None
    published_audio_path: Path | None = None
    drive_publication_meta: dict | None = None

    try:
        audio_path = download_audio_to_workdir(normalized_url, work_dir, episode.id)
        report_payload["notes"].append(f"Audio descargado en: {audio_path.name}")

        audio_probe = probe_audio_file(audio_path)
        report_payload["notes"].append("Audio validado con ffprobe.")

        if episode.source_drive_url:
            drive_publication_meta = {
                "file_id": None,
                "operational_url": str(episode.source_drive_url),
                "web_view_url": None,
                "published_filename": None,
                "reused_existing_drive_source": True,
            }
            report_payload["notes"].append(
                "El episodio ya operaba desde Drive; no se republicó el audio en Podcasts/Pending."
            )
        else:
            if not DRIVE_PODCASTS_PENDING_FOLDER_ID:
                raise RuntimeError("DRIVE_PODCASTS_PENDING_FOLDER_ID no configurado.")
            if not GOOGLE_SERVICE_ACCOUNT_JSON:
                raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON no configurado.")

            published_filename = build_published_audio_filename(
                episode.podcast_title,
                episode.episode_title,
                audio_path.suffix,
            )
            published_audio_path = work_dir / published_filename

            rewrite_audio_metadata(
                input_audio=audio_path,
                output_audio=published_audio_path,
                podcast_title=episode.podcast_title,
                episode_title=episode.episode_title,
            )
            report_payload["notes"].append(
                f"Audio reescrito con metadatos limpios: {published_audio_path.name}"
            )

            drive_publication_meta = publish_audio_to_podcasts_pending(
                service_account_json=GOOGLE_SERVICE_ACCOUNT_JSON,
                local_path=published_audio_path,
                filename=published_filename,
                pending_folder_id=DRIVE_PODCASTS_PENDING_FOLDER_ID,
            )
            drive_publication_meta["published_filename"] = published_filename
            drive_publication_meta["reused_existing_drive_source"] = False

            report_payload["notes"].append(
                f"Audio subido a Drive/Podcasts/Pending: {drive_publication_meta['file_id']}"
            )

            update_episode_operational_audio_url(
                episode_id=episode.id,
                new_url=drive_publication_meta["operational_url"],
            )
            report_payload["notes"].append(
                "inputs/episodes.yaml actualizado para usar la URL operativa de Drive."
            )

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

        transcription_meta = transcribe_bundle(
            audio_path,
            work_dir,
            model_name="base",
        )
        report_payload["notes"].append(
            f"Transcripción generada ({transcription_meta['num_segments_srt']} segmentos)"
        )
        report_payload["notes"].append(
            f"Idioma detectado: {transcription_meta.get('language') or 'desconocido'}"
        )
        report_payload["notes"].append(
            f"Texto transcrito: {transcription_meta.get('text_characters', 0)} caracteres"
        )
        print(f"Transcripción completada: {transcription_meta['num_segments_srt']} segmentos")

        print("Normalizando audio para diarización...")
        normalized_for_diarization = normalize_audio_for_diarization(
            audio_path,
            work_dir / "audio_mono_16k.wav",
        )
        print(f"Audio normalizado: {normalized_for_diarization}")

        print("Iniciando diarización...")
        diarization_meta = diarize_audio(
            normalized_for_diarization,
            work_dir / "speaker_timeline.rttm",
            work_dir / "speaker_segments.json",
            HF_TOKEN,
        )
        report_payload["notes"].append(
            f"Diarización generada ({diarization_meta['num_speakers_detected']} hablantes, {diarization_meta['num_segments']} segmentos)"
        )
        report_payload["notes"].append(
            f"Tiempos diarización: {diarization_meta['timing_seconds']}"
        )
        print(
            f"Diarización completada: {diarization_meta['num_speakers_detected']} hablantes, {diarization_meta['num_segments']} segmentos"
        )

        print("Cruzando transcripción + diarización...")
        alignment_meta = assign_speakers_to_transcript_segments(
            work_dir / "transcript_segments.json",
            work_dir / "speaker_segments.json",
            work_dir / "aligned_transcript_segments.json",
        )

        num_full_srt_segments = write_full_transcript_with_speakers(
            work_dir / "aligned_transcript_segments.json",
            work_dir / "full_transcript_speakers.srt",
        )

        per_speaker_outputs = write_per_speaker_srts(
            work_dir / "aligned_transcript_segments.json",
            work_dir / "speakers_auto",
        )

        report_payload["notes"].append(
            f"Cruce transcripción+diarización generado ({alignment_meta['num_aligned_segments']} segmentos alineados)"
        )
        report_payload["notes"].append(
            f"SRT con hablantes generado ({num_full_srt_segments} segmentos)"
        )
        report_payload["notes"].append(
            f"SRT automáticos por hablante generados ({len(per_speaker_outputs)} archivos)"
        )
        print(
            f"Cruce completado: {alignment_meta['num_aligned_segments']} segmentos, {len(per_speaker_outputs)} speakers"
        )

        review_structure = _initialize_reviewed_speakers(work_dir, report_payload)

        duration_seconds = audio_probe.get("duration_seconds")
        duration_text = (
            f"{duration_seconds:.2f} segundos"
            if isinstance(duration_seconds, float)
            else "desconocida"
        )

        language_detected = transcription_meta.get("language") or "desconocido"
        transcript_preview = transcription_meta.get("text_preview") or ""

        (work_dir / "summary.md").write_text(
            (
                "# Summary\n\n"
                "PREAUDITORÍA técnica completada.\n\n"
                f"- Episodio: {episode.episode_title}\n"
                f"- Podcast: {episode.podcast_title}\n"
                f"- Duración detectada: {duration_text}\n"
                f"- Idioma detectado: {language_detected}\n"
                "- Estado: transcripción global y diarización generadas; auditoría humana pendiente\n"
                f"- Segmentos SRT: {transcription_meta.get('num_segments_srt')}\n"
                f"- Caracteres transcritos: {transcription_meta.get('text_characters')}\n"
                f"- Hablantes detectados: {diarization_meta.get('num_speakers_detected')}\n"
                f"- Segmentos de diarización: {diarization_meta.get('num_segments')}\n"
                f"- Segmentos alineados: {alignment_meta.get('num_aligned_segments')}\n"
                f"- Hablantes distintos en alineado: {alignment_meta.get('num_distinct_speakers')}\n\n"
                "## Vista previa\n\n"
                f"{transcript_preview}\n"
            ),
            encoding="utf-8",
        )

        (work_dir / "outline.md").write_text(
            (
                "# Outline\n\n"
                "1. Descarga verificada\n"
                "2. Validación técnica del audio\n"
                "3. Publicación del audio fuente en Drive/Podcasts/Pending\n"
                "4. Transcripción global generada\n"
                "5. Detección de idioma estimada\n"
                "6. Diarización generada\n"
                "7. Cruce transcripción + diarización\n"
                "8. Generación de SRT automáticos por hablante\n"
                "9. Inicialización de revisión humana\n"
                "10. Pendiente propuesta real de identidades\n"
                "11. Pendiente auditoría humana\n"
            ),
            encoding="utf-8",
        )

        episode_payload = {
            "episode_id": episode.id,
            "podcast_title": episode.podcast_title,
            "episode_title": episode.episode_title,
            "source_url_original": str(episode.source_url_original) if episode.source_url_original else str(episode.url),
            "source_url_normalized": normalized_url,
            "source_drive_url": drive_publication_meta.get("operational_url") if drive_publication_meta else str(episode.source_drive_url) if episode.source_drive_url else None,
            "status_after_preaudit": "pending_review",
            "downloaded_audio_filename": audio_path.name if audio_path else None,
            "audio_probe": audio_probe,
            "source_drive_publication": {
                "file_id": drive_publication_meta.get("file_id") if drive_publication_meta else None,
                "operational_url": drive_publication_meta.get("operational_url") if drive_publication_meta else None,
                "web_view_url": drive_publication_meta.get("web_view_url") if drive_publication_meta else None,
                "published_filename": drive_publication_meta.get("published_filename") if drive_publication_meta else None,
                "reused_existing_drive_source": drive_publication_meta.get("reused_existing_drive_source") if drive_publication_meta else False,
            },
            "language_detected": transcription_meta.get("language"),
            "transcription": {
                "model_name": transcription_meta.get("model_name"),
                "num_segments_srt": transcription_meta.get("num_segments_srt"),
                "num_segments_json": transcription_meta.get("num_segments_json"),
                "text_characters": transcription_meta.get("text_characters"),
            },
            "diarization": {
                "num_speakers_detected": diarization_meta.get("num_speakers_detected"),
                "num_segments": diarization_meta.get("num_segments"),
                "speaker_ids": diarization_meta.get("speaker_ids"),
                "timing_seconds": diarization_meta.get("timing_seconds"),
            },
            "alignment": {
                "num_aligned_segments": alignment_meta.get("num_aligned_segments"),
                "num_distinct_speakers": alignment_meta.get("num_distinct_speakers"),
                "total_aligned_time_seconds": alignment_meta.get("total_aligned_time_seconds"),
            },
            "speaker_metrics": alignment_meta.get("speaker_metrics"),
            "speaker_candidates": [
                {
                    "speaker_id": metric["speaker_id"],
                    "inferred_name": "pendiente_de_validar",
                    "confidence": 0.0,
                    "alternatives": [],
                }
                for metric in alignment_meta.get("speaker_metrics", [])
                if metric["speaker_id"] != "speaker_unknown"
            ],
            "review_structure": review_structure,
            "duration_seconds": audio_probe.get("duration_seconds"),
            "topics_detected": [],
            "participants_declared": episode.participants or [],
        }
        write_json(work_dir / "episode.json", episode_payload)

        if normalized_for_diarization is not None:
            _safe_unlink(normalized_for_diarization)
        if published_audio_path is not None:
            _safe_unlink(published_audio_path)
        if audio_path is not None:
            _safe_unlink(audio_path)

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

        update_episode_status(
            episode.id,
            "failed",
            increment_retries=True,
        )
        print(f"{episode.id}: fallo técnico -> failed")
        return 1

    except Exception as exc:
        report_payload["result"] = "failed"
        report_payload["finished_at"] = utc_now_iso()
        report_payload["notes"].append(f"Excepción no controlada: {exc}")
        report_payload["notes"].append(traceback.format_exc())
        write_json(work_dir / "report.json", report_payload)

        update_episode_status(episode.id, "failed", increment_retries=True)
        print(f"{episode.id}: excepción no controlada -> failed")
        raise
