from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .profile_quality import classify_profile_quality

TESTING_REGISTRY_ROOT = Path("data/testing_speakers")
TESTING_REGISTRY_PROFILES = TESTING_REGISTRY_ROOT / "profiles"
TESTING_REGISTRY_INDEX = TESTING_REGISTRY_ROOT / "index.json"
TESTING_REGISTRY_METADATA = TESTING_REGISTRY_ROOT / "registry_metadata.json"


@dataclass(slots=True)
class TestingRegistryProfile:
    testing_speaker_id: str
    display_label: str
    profile_quality: str
    num_episodes: int
    total_speech_seconds: float
    canonical_embedding: dict[str, Any]
    path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_placeholder() -> str:
    return "1970-01-01T00:00:00Z"


def _ensure_registry_structure() -> None:
    TESTING_REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
    TESTING_REGISTRY_PROFILES.mkdir(parents=True, exist_ok=True)

    if not TESTING_REGISTRY_METADATA.exists():
        TESTING_REGISTRY_METADATA.write_text(
            json.dumps(
                {
                    "registry_type": "testing_speakers",
                    "version": "v1",
                    "source_of_truth": False,
                    "description": "Temporary biometric registry for cross-episode speaker matching tests.",
                    "updated_at": _utc_placeholder(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if not TESTING_REGISTRY_INDEX.exists():
        TESTING_REGISTRY_INDEX.write_text(
            json.dumps({"version": "v1", "profiles": []}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _next_testing_speaker_id() -> str:
    _ensure_registry_structure()
    index = _read_json(TESTING_REGISTRY_INDEX, {"version": "v1", "profiles": []})
    profiles = index.get("profiles", [])

    max_id = 0
    for item in profiles:
        if not isinstance(item, dict):
            continue
        raw_id = str(item.get("testing_speaker_id") or "")
        if raw_id.startswith("test_spk_"):
            try:
                max_id = max(max_id, int(raw_id.split("_")[-1]))
            except ValueError:
                continue

    return f"test_spk_{max_id + 1:06d}"


def load_testing_registry_profiles() -> list[dict[str, Any]]:
    _ensure_registry_structure()
    index = _read_json(TESTING_REGISTRY_INDEX, {"version": "v1", "profiles": []})
    results: list[dict[str, Any]] = []

    for item in index.get("profiles", []):
        if not isinstance(item, dict):
            continue

        profile_dir = Path(str(item.get("path") or ""))
        if not profile_dir.exists():
            continue

        profile_json = _read_json(profile_dir / "profile.json", {})
        canonical_embedding = _read_json(profile_dir / "canonical_embedding.json", {})
        episodes_json = _read_json(profile_dir / "episodes.json", {"episodes": []})

        results.append(
            {
                "testing_speaker_id": profile_json.get("testing_speaker_id"),
                "display_label": profile_json.get("display_label"),
                "profile_quality": canonical_embedding.get("profile_quality"),
                "num_episodes": len(episodes_json.get("episodes", [])),
                "total_speech_seconds": canonical_embedding.get("total_speech_seconds_used", 0.0),
                "canonical_embedding": canonical_embedding,
                "path": str(profile_dir),
            }
        )

    return results


def _build_canonical_embedding_from_episode_profile(profile: dict[str, Any]) -> dict[str, Any]:
    embedding_primary = profile.get("embedding_primary") or {}
    dispersion = profile.get("intra_speaker_dispersion") or {}

    return {
        "model": embedding_primary.get("model"),
        "vector": embedding_primary.get("vector"),
        "aggregation_method": embedding_primary.get("method"),
        "num_episode_profiles_used": 1,
        "total_segments_used": profile.get("num_segments_selected"),
        "total_speech_seconds_used": profile.get("speech_seconds_selected"),
        "intra_registry_dispersion_mean": dispersion.get("mean"),
        "intra_registry_dispersion_p95": dispersion.get("p95"),
        "profile_quality": classify_profile_quality(profile),
        "updated_at": _utc_placeholder(),
    }


def _merge_vectors(existing: list[float], new: list[float], weight_existing: float, weight_new: float) -> list[float]:
    total = weight_existing + weight_new
    if total <= 0:
        return existing

    return [
        ((a * weight_existing) + (b * weight_new)) / total
        for a, b in zip(existing, new)
    ]


def _append_match_history(profile_dir: Path, payload: dict[str, Any]) -> None:
    path = profile_dir / "match_history.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def create_testing_profile_from_episode_speaker(
    *,
    episode_id: str,
    speaker: str,
    profile: dict[str, Any],
) -> str:
    _ensure_registry_structure()

    testing_speaker_id = _next_testing_speaker_id()
    profile_dir = TESTING_REGISTRY_PROFILES / testing_speaker_id
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_payload = {
        "testing_speaker_id": testing_speaker_id,
        "display_label": f"Test Speaker {testing_speaker_id.split('_')[-1]}",
        "status": "active",
        "created_at": _utc_placeholder(),
        "updated_at": _utc_placeholder(),
        "source_of_truth": False,
        "notes": "Temporary cross-episode speaker bucket for testing.",
    }

    canonical_embedding = _build_canonical_embedding_from_episode_profile(profile)

    episodes_payload = {
        "testing_speaker_id": testing_speaker_id,
        "episodes": [
            {
                "episode_id": episode_id,
                "speaker": speaker,
                "speech_seconds_selected": profile.get("speech_seconds_selected"),
                "num_segments_used": profile.get("num_segments_selected"),
                "profile_quality": classify_profile_quality(profile),
                "embedding_status": (profile.get("embedding_primary") or {}).get("status"),
            }
        ],
    }

    _write_json(profile_dir / "profile.json", profile_payload)
    _write_json(profile_dir / "canonical_embedding.json", canonical_embedding)
    _write_json(profile_dir / "episodes.json", episodes_payload)

    _append_match_history(
        profile_dir,
        {
            "episode_id": episode_id,
            "speaker": speaker,
            "action": "created_new_testing_profile",
            "testing_speaker_id": testing_speaker_id,
            "score": None,
            "timestamp": _utc_placeholder(),
        },
    )

    index = _read_json(TESTING_REGISTRY_INDEX, {"version": "v1", "profiles": []})
    index["profiles"].append(
        {
            "testing_speaker_id": testing_speaker_id,
            "status": "active",
            "num_episodes": 1,
            "total_speech_seconds": float(profile.get("speech_seconds_selected") or 0.0),
            "path": str(profile_dir),
        }
    )
    _write_json(TESTING_REGISTRY_INDEX, index)

    return testing_speaker_id


def update_testing_profile_with_episode_speaker(
    *,
    testing_speaker_id: str,
    episode_id: str,
    speaker: str,
    profile: dict[str, Any],
    score: float,
) -> None:
    _ensure_registry_structure()

    profile_dir = TESTING_REGISTRY_PROFILES / testing_speaker_id
    profile_json = _read_json(profile_dir / "profile.json", {})
    canonical_embedding = _read_json(profile_dir / "canonical_embedding.json", {})
    episodes_json = _read_json(profile_dir / "episodes.json", {"episodes": []})

    existing_vector = canonical_embedding.get("vector")
    new_vector = (profile.get("embedding_primary") or {}).get("vector")

    if not isinstance(existing_vector, list) or not isinstance(new_vector, list):
        raise RuntimeError("Cannot update testing registry profile without valid vectors.")

    existing_seconds = float(canonical_embedding.get("total_speech_seconds_used") or 0.0)
    new_seconds = float(profile.get("speech_seconds_selected") or 0.0)

    merged_vector = _merge_vectors(existing_vector, new_vector, existing_seconds, new_seconds)

    episodes_json["episodes"].append(
        {
            "episode_id": episode_id,
            "speaker": speaker,
            "speech_seconds_selected": profile.get("speech_seconds_selected"),
            "num_segments_used": profile.get("num_segments_selected"),
            "profile_quality": classify_profile_quality(profile),
            "embedding_status": (profile.get("embedding_primary") or {}).get("status"),
        }
    )

    canonical_embedding.update(
        {
            "vector": merged_vector,
            "num_episode_profiles_used": len(episodes_json["episodes"]),
            "total_segments_used": int(canonical_embedding.get("total_segments_used") or 0)
            + int(profile.get("num_segments_selected") or 0),
            "total_speech_seconds_used": existing_seconds + new_seconds,
            "profile_quality": classify_profile_quality(profile),
            "updated_at": _utc_placeholder(),
        }
    )

    profile_json["updated_at"] = _utc_placeholder()

    _write_json(profile_dir / "profile.json", profile_json)
    _write_json(profile_dir / "canonical_embedding.json", canonical_embedding)
    _write_json(profile_dir / "episodes.json", episodes_json)

    _append_match_history(
        profile_dir,
        {
            "episode_id": episode_id,
            "speaker": speaker,
            "action": "matched_existing_testing_profile",
            "testing_speaker_id": testing_speaker_id,
            "score": score,
            "timestamp": _utc_placeholder(),
        },
    )

    index = _read_json(TESTING_REGISTRY_INDEX, {"version": "v1", "profiles": []})
    for item in index.get("profiles", []):
        if item.get("testing_speaker_id") == testing_speaker_id:
            item["num_episodes"] = len(episodes_json["episodes"])
            item["total_speech_seconds"] = canonical_embedding.get("total_speech_seconds_used", 0.0)
            break
    _write_json(TESTING_REGISTRY_INDEX, index)
