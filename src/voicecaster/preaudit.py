from __future__ import annotations

from pathlib import Path

from .config import REVIEWS_DIR, WORK_DIR
from .episode_queue import reserve_next_pending_episode, update_episode_status
from .reporting import write_json, utc_now_iso
from .url_resolver import normalize_download_url
from .yaml_io import write_yaml


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_").lower()


def run_preaudit() -> int:
    episode = reserve_next_pending_episode()
    if episode is None:
        print("No hay episodios pendientes.")
        return 0

    work_dir = WORK_DIR / episode.id
    review_dir = REVIEWS_DIR / episode.id
    work_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    normalized_url = normalize_download_url(str(episode.url))

    # TODO v1.1:
    # - verificar accesibilidad real del recurso
    # - descargar audio temporalmente
    # - transcribir
    # - diarizar
    # - generar srt global + speaker_xx.srt
    # - generar summary.md, outline.md, episode.json

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

    (work_dir / "summary.md").write_text(
        "# Summary\n\nPendiente de integración real de transcripción.\n",
        encoding="utf-8",
    )
    (work_dir / "outline.md").write_text(
        "# Outline\n\nPendiente de integración real de transcripción.\n",
        encoding="utf-8",
    )
    (work_dir / "full_transcript.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n[speaker_01] Placeholder de transcripción.\n",
        encoding="utf-8",
    )
    speaker_dir = work_dir / "speakers"
    speaker_dir.mkdir(exist_ok=True)
    (speaker_dir / "speaker_01.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nPlaceholder de intervención.\n",
        encoding="utf-8",
    )

    episode_payload = {
        "episode_id": episode.id,
        "podcast_title": episode.podcast_title,
        "episode_title": episode.episode_title,
        "source_url_original": str(episode.url),
        "source_url_normalized": normalized_url,
        "status_after_preaudit": "pending_review",
        "speaker_candidates": [
            {
                "speaker_id": "speaker_01",
                "inferred_name": "pendiente_de_validar",
                "confidence": 0.0,
                "alternatives": [],
            }
        ],
        "duration_seconds": None,
        "topics_detected": [],
        "participants_declared": episode.participants or [],
    }
    write_json(work_dir / "episode.json", episode_payload)

    report_payload = {
        "episode_id": episode.id,
        "phase": "preaudit",
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "result": "pending_review",
        "notes": [
            "Estructura base generada.",
            "Falta cableado real de ASR, diarización y extracción de métricas.",
        ],
    }
    write_json(work_dir / "report.json", report_payload)

    update_episode_status(episode.id, "pending_review")
    print(f"PREAUDITORÍA completada para {episode.id} -> pending_review")
    return 0
