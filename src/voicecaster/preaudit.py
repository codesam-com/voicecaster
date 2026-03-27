from __future__ import annotations

import shutil
import traceback
from datetime import datetime, UTC
from pathlib import Path

from .audio_normalize import normalize_audio_for_diarization
from .audio_probe import AudioProbeError, probe_audio_file
from .config import HF_TOKEN, RUNTIME_CONTROL_PATH, REVIEWS_DIR, WORK_DIR
from .diarizer import diarize_audio
from .downloader import DownloadError, IncompatibleSourceError, download_audio_to_workdir
from .episode_queue import reserve_next_pending_episode, update_episode_status
from .redecode import redecode_doubtful_segments
from .reporting import utc_now_iso, write_json
from .runtime_control import should_run_now, update_runtime_control
from .speaker_alignment import (
    assign_speakers_to_transcript_segments,
    write_full_transcript_with_speakers,
    write_per_speaker_srts,
)
from .transcriber import transcribe_audio_large_v3, write_whisper_outputs
from .url_resolver import normalize_download_url
from .vad import detect_vad_segments
from .whisperx_aligner import align_with_whisperx
from .work_layout import build_work_layout, write_status_json, write_work_readme
from .yaml_io import write_yaml

WORKFLOW_NAME = "preaudit"


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _init_speakers_reviewed(speakers_auto_dir: Path, speakers_reviewed_dir: Path) -> bool:
    if not speakers_reviewed_dir.exists():
        shutil.copytree(speakers_auto_dir, speakers_reviewed_dir)
        return True
    return False


def run_preaudit() -> int:
    print("Iniciando PREAUDITORÍA...")

    should_run, runtime_info = should_run_now(RUNTIME_CONTROL_PATH, WORKFLOW_NAME)
    if not should_run:
        print("Runtime control: todavía no toca ejecutar. Exit limpio.")
        print(runtime_info)
        return 0

    episode = reserve_next_pending_episode()
    if episode is None:
        print("No hay episodios pendientes.")
        return 0

    print(f"Episodio reservado: {episode.id}")

    started_dt = datetime.now(UTC)
    started_at = started_dt.isoformat()

    work_dir = WORK_DIR / episode.id
    review_dir = REVIEWS_DIR / episode.id
    work_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    layout = build_work_layout(work_dir)
    write_work_readme(work_dir, episode.id)

    normalized_url = normalize_download_url(str(episode.url))

    report_payload = {
        "episode_id": episode.id,
        "phase": "preaudit",
        "started_at": started_at,
        "finished_at": None,
        "result": None,
        "notes": [
            f"Runtime control workflow={WORKFLOW_NAME} decision={runtime_info.get('decision')}"
        ],
        "source_url_original": str(episode.url),
        "source_url_normalized": normalized_url,
    }
    if runtime_info.get("next_allowed_run_at"):
        report_payload["notes"].append(f"Next allowed run at: {runtime_info['next_allowed_run_at']}")

    audio_path: Path | None = None
    normalized_for_diarization: Path | None = None

    status_payload = {
        "episode_id": episode.id,
        "phase": "preaudit",
        "status": "running",
        "started_at": started_at,
        "current_step": "init",
    }
    write_status_json(work_dir, status_payload)

    try:
        status_payload["current_step"] = "download"
        write_status_json(work_dir, status_payload)

        audio_path = download_audio_to_workdir(normalized_url, layout["intake"], "source_audio")
        report_payload["notes"].append(f"Audio descargado en: {audio_path.name}")

        status_payload["current_step"] = "probe_audio"
        write_status_json(work_dir, status_payload)
        audio_probe = probe_audio_file(audio_path)
        report_payload["notes"].append("Audio validado con ffprobe.")

        source_metadata = {
            "episode_id": episode.id,
            "source_url_original": str(episode.url),
            "source_url_normalized": normalized_url,
            "audio_probe": audio_probe,
        }
        write_json(layout["intake"] / "source_metadata.json", source_metadata)

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

        status_payload["current_step"] = "vad"
        write_status_json(work_dir, status_payload)
        vad_meta = detect_vad_segments(audio_path, layout["transcription"] / "vad_segments.json")
        report_payload["notes"].append(f"VAD generado ({vad_meta['num_vad_segments']} segmentos)")
        report_payload["notes"].append(f"Voz detectada: {vad_meta['total_speech_seconds']} segundos")

        status_payload["current_step"] = "transcription"
        write_status_json(work_dir, status_payload)
        whisper_result = transcribe_audio_large_v3(audio_path)
        transcription_meta = write_whisper_outputs(whisper_result, layout["transcription"])
        report_payload["notes"].append(f"Transcripción generada ({transcription_meta['num_segments_srt']} segmentos)")
        report_payload["notes"].append(f"Idioma detectado: {transcription_meta.get('language') or 'desconocido'}")
        report_payload["notes"].append(f"Texto transcrito: {transcription_meta.get('text_characters', 0)} caracteres")

        status_payload["current_step"] = "forced_alignment"
        write_status_json(work_dir, status_payload)
        whisperx_meta = align_with_whisperx(
            audio_path,
            layout["transcription"] / "transcript_segments.json",
            layout["transcription"] / "whisperx_aligned_segments.json",
            transcription_meta.get("language"),
        )
        report_payload["notes"].append(f"WhisperX alignment generado ({whisperx_meta['num_aligned_segments']} segmentos)")

        status_payload["current_step"] = "normalize_for_diarization"
        write_status_json(work_dir, status_payload)
        normalized_for_diarization = normalize_audio_for_diarization(
            audio_path,
            layout["temp"] / "audio_mono_16k.wav",
        )

        status_payload["current_step"] = "diarization"
        write_status_json(work_dir, status_payload)
        diarization_meta = diarize_audio(
            normalized_for_diarization,
            layout["diarization"] / "speaker_timeline.rttm",
            layout["diarization"] / "speaker_segments.json",
            HF_TOKEN,
        )
        report_payload["notes"].append(
            f"Diarización generada ({diarization_meta['num_speakers_detected']} hablantes, {diarization_meta['num_segments']} segmentos)"
        )
        report_payload["notes"].append(f"Tiempos diarización: {diarization_meta['timing_seconds']}")

        status_payload["current_step"] = "alignment"
        write_status_json(work_dir, status_payload)
        alignment_meta = assign_speakers_to_transcript_segments(
            layout["transcription"] / "whisperx_aligned_segments.json",
            layout["diarization"] / "speaker_segments.json",
            layout["alignment"] / "aligned_transcript_segments.json",
            layout["alignment"] / "doubtful_segments.json",
        )
        report_payload["notes"].append(
            f"Cruce transcripción+diarización generado ({alignment_meta['num_aligned_segments']} segmentos alineados)"
        )
        report_payload["notes"].append(
            f"Segmentos dudosos detectados: {alignment_meta['num_doubtful_segments']}"
        )

        status_payload["current_step"] = "redecode_doubtful"
        write_status_json(work_dir, status_payload)
        redecoded_meta = redecode_doubtful_segments(
            audio_path,
            layout["alignment"] / "doubtful_segments.json",
            layout["temp"],
            layout["alignment"] / "redecoded_segments.json",
        )
        report_payload["notes"].append(
            f"Segunda pasada completada ({redecoded_meta['num_redecoded_segments']} segmentos reprocesados)"
        )

        num_full_srt_segments = write_full_transcript_with_speakers(
            layout["alignment"] / "aligned_transcript_segments.json",
            layout["alignment"] / "full_transcript_speakers.srt",
        )
        report_payload["notes"].append(f"SRT con hablantes generado ({num_full_srt_segments} segmentos)")

        status_payload["current_step"] = "speaker_srts"
        write_status_json(work_dir, status_payload)

        speakers_auto_dir = layout["review"] / "speakers_auto"
        speakers_reviewed_dir = layout["review"] / "speakers_reviewed"

        per_speaker_outputs = write_per_speaker_srts(
            layout["alignment"] / "aligned_transcript_segments.json",
            speakers_auto_dir,
        )
        reviewed_initialized = _init_speakers_reviewed(speakers_auto_dir, speakers_reviewed_dir)

        report_payload["notes"].append(
            f"SRT automáticos por hablante generados ({len(per_speaker_outputs)} archivos)"
        )
        if reviewed_initialized:
            report_payload["notes"].append("speakers_reviewed inicializado desde speakers_auto/")
        else:
            report_payload["notes"].append("speakers_reviewed ya existía y no se sobrescribió")

        status_payload["current_step"] = "episode_outputs"
        write_status_json(work_dir, status_payload)

        duration_seconds = audio_probe.get("duration_seconds")
        duration_text = f"{duration_seconds:.2f} segundos" if isinstance(duration_seconds, float) else "desconocida"
        language_detected = transcription_meta.get("language") or "desconocido"
        transcript_preview = transcription_meta.get("text_preview") or ""

        (layout["episode_outputs"] / "summary.md").write_text(
            (
                "# Summary\n\n"
                "PREAUDITORÍA quality_max completada.\n\n"
                f"- Episodio: {episode.episode_title}\n"
                f"- Podcast: {episode.podcast_title}\n"
                f"- Duración detectada: {duration_text}\n"
                f"- Idioma detectado: {language_detected}\n"
                "- Perfil de calidad: quality_max\n"
                f"- Segmentos VAD: {vad_meta.get('num_vad_segments')}\n"
                f"- Segmentos SRT: {transcription_meta.get('num_segments_srt')}\n"
                f"- Caracteres transcritos: {transcription_meta.get('text_characters')}\n"
                f"- Hablantes detectados: {diarization_meta.get('num_speakers_detected')}\n"
                f"- Segmentos de diarización: {diarization_meta.get('num_segments')}\n"
                f"- Segmentos alineados: {alignment_meta.get('num_aligned_segments')}\n"
                f"- Segmentos dudosos: {alignment_meta.get('num_doubtful_segments')}\n"
                f"- Segmentos reprocesados: {redecoded_meta.get('num_redecoded_segments')}\n\n"
                "## Vista previa\n\n"
                f"{transcript_preview}\n"
            ),
            encoding="utf-8",
        )

        (layout["episode_outputs"] / "outline.md").write_text(
            (
                "# Outline\n\n"
                "1. Descarga verificada\n"
                "2. Validación técnica del audio\n"
                "3. VAD generado\n"
                "4. Transcripción large-v3 generada\n"
                "5. WhisperX forced alignment generado\n"
                "6. Diarización pyannote generada\n"
                "7. Cruce transcripción + diarización generado\n"
                "8. Segmentos dudosos detectados y reprocesados\n"
                "9. speakers_auto generado\n"
                "10. speakers_reviewed preparado para auditoría humana\n"
            ),
            encoding="utf-8",
        )

        episode_payload = {
            "episode_id": episode.id,
            "podcast_title": episode.podcast_title,
            "episode_title": episode.episode_title,
            "source_url_original": str(episode.url),
            "source_url_normalized": normalized_url,
            "status_after_preaudit": "pending_review",
            "downloaded_audio_filename": audio_path.name if audio_path else None,
            "audio_probe": audio_probe,
            "quality_profile": "quality_max",
            "vad": vad_meta,
            "language_detected": transcription_meta.get("language"),
            "transcription": {
                "model_name": "openai/whisper-large-v3",
                "num_segments_srt": transcription_meta.get("num_segments_srt"),
                "num_segments_json": transcription_meta.get("num_segments_json"),
                "text_characters": transcription_meta.get("text_characters"),
            },
            "whisperx_alignment": whisperx_meta,
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
                "num_doubtful_segments": alignment_meta.get("num_doubtful_segments"),
            },
            "quality_metrics": {
                "redecoded_segments": redecoded_meta.get("num_redecoded_segments"),
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
            "review_structure": {
                "speakers_auto_dir": "04_review/speakers_auto",
                "speakers_reviewed_dir": "04_review/speakers_reviewed",
                "reviewed_initialized_from_auto": reviewed_initialized,
            },
            "duration_seconds": audio_probe.get("duration_seconds"),
            "topics_detected": [],
            "participants_declared": episode.participants or [],
        }
        write_json(layout["episode_outputs"] / "episode.json", episode_payload)

        if normalized_for_diarization is not None:
            _safe_unlink(normalized_for_diarization)
        if audio_path is not None:
            _safe_unlink(audio_path)

        report_payload["result"] = "pending_review"
        report_payload["finished_at"] = utc_now_iso()
        write_json(layout["logs"] / "report.json", report_payload)

        update_episode_status(episode.id, "pending_review")

        finished_dt = datetime.now(UTC)
        duration_run = (finished_dt - started_dt).total_seconds()
        update_runtime_control(RUNTIME_CONTROL_PATH, WORKFLOW_NAME, duration_run, finished_dt)

        status_payload["status"] = "completed"
        status_payload["current_step"] = "done"
        status_payload["finished_at"] = finished_dt.isoformat()
        status_payload["result"] = "pending_review"
        write_status_json(work_dir, status_payload)

        print(f"Archivo de auditoría: {audit_path}")
        print(f"Directorio de trabajo: {work_dir}")
        print(f"Report: {layout['logs'] / 'report.json'}")
        print(f"PREAUDITORÍA completada para {episode.id} -> pending_review")
        return 0

    except IncompatibleSourceError as exc:
        report_payload["result"] = "incompatible"
        report_payload["finished_at"] = utc_now_iso()
        report_payload["notes"].append(str(exc))
        write_json(layout["logs"] / "report.json", report_payload)
        update_episode_status(episode.id, "incompatible")
        status_payload["status"] = "failed"
        status_payload["current_step"] = "error"
        status_payload["result"] = "incompatible"
        write_status_json(work_dir, status_payload)
        return 0

    except (DownloadError, AudioProbeError) as exc:
        report_payload["result"] = "failed"
        report_payload["finished_at"] = utc_now_iso()
        report_payload["notes"].append(str(exc))
        write_json(layout["logs"] / "report.json", report_payload)
        update_episode_status(episode.id, "failed", increment_retries=True)
        status_payload["status"] = "failed"
        status_payload["current_step"] = "error"
        status_payload["result"] = "failed"
        write_status_json(work_dir, status_payload)
        return 1

    except Exception as exc:
        report_payload["result"] = "failed"
        report_payload["finished_at"] = utc_now_iso()
        report_payload["notes"].append(f"Excepción no controlada: {exc}")
        report_payload["notes"].append(traceback.format_exc())
        write_json(layout["logs"] / "report.json", report_payload)
        update_episode_status(episode.id, "failed", increment_retries=True)
        status_payload["status"] = "failed"
        status_payload["current_step"] = "error"
        status_payload["result"] = "failed"
        write_status_json(work_dir, status_payload)
        raise
