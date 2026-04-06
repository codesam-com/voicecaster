from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schemas import IdentityCandidate, IdentityDecision, IdentityQAResult


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_speaker_identity_request_json(identity_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(identity_dir / "speaker_identity_request.json", payload)


def write_speaker_voice_profiles_json(
    identity_dir: Path,
    episode_id: str,
    speakers: list[Any],
) -> None:
    payload = {
        "episode_id": episode_id,
        "speakers": [speaker.to_dict() for speaker in speakers],
    }
    _write_json(identity_dir / "speaker_voice_profiles.json", payload)


def write_identity_evidence_json(
    identity_dir: Path,
    episode_id: str,
    selected_segments_by_speaker: dict[str, list[dict[str, Any]]],
) -> None:
    payload = {
        "episode_id": episode_id,
        "speakers": [
            {
                "speaker": speaker,
                "selected_segments": segments,
            }
            for speaker, segments in selected_segments_by_speaker.items()
        ],
    }
    _write_json(identity_dir / "identity_evidence.json", payload)


def write_text_support_json(
    identity_dir: Path,
    episode_id: str,
    text_evidence: list[Any],
) -> None:
    payload = {
        "episode_id": episode_id,
        "speakers": [item.to_dict() for item in text_evidence],
    }
    _write_json(identity_dir / "text_support.json", payload)


def write_identity_candidates_json(
    identity_dir: Path,
    episode_id: str,
    candidates_by_speaker: dict[str, list[IdentityCandidate]],
) -> None:
    payload = {
        "episode_id": episode_id,
        "speakers": [
            {
                "speaker": speaker,
                "candidates": [candidate.to_dict() for candidate in candidates],
            }
            for speaker, candidates in candidates_by_speaker.items()
        ],
    }
    _write_json(identity_dir / "identity_candidates.json", payload)


def write_speaker_identity_json(
    identity_dir: Path,
    episode_id: str,
    decisions: list[IdentityDecision],
) -> None:
    payload = {
        "episode_id": episode_id,
        "status": "unverified",
        "speakers": [decision.to_dict() for decision in decisions],
    }
    _write_json(identity_dir / "speaker_identity.json", payload)


def write_speaker_identity_metadata_json(
    identity_dir: Path,
    payload: dict[str, Any],
) -> None:
    _write_json(identity_dir / "speaker_identity_metadata.json", payload)


def write_qa_summary_json(
    identity_dir: Path,
    qa_result: IdentityQAResult,
) -> None:
    _write_json(identity_dir / "qa_summary.json", qa_result.to_dict())


def write_identity_result_json(
    identity_dir: Path,
    payload: dict[str, Any],
) -> None:
    _write_json(identity_dir / "identity_result.json", payload)
