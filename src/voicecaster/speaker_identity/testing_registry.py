from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .profile_quality import classify_profile_quality

TESTING_REGISTRY_ROOT = Path("data/testing_speakers")
TESTING_REGISTRY_PROFILES = TESTING_REGISTRY_ROOT / "profiles"
TESTING_REGISTRY_INDEX = TESTING_REGISTRY_ROOT / "index.json"
TESTING_REGISTRY_METADATA = TESTING_REGISTRY_ROOT / "registry_metadata.json"


def _utc_placeholder() -> str:
    return "1970-01-01T00:00:00Z"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _profile_dir_for(testing_speaker_id: str) -> Path:
    return TESTING_REGISTRY_PROFILES / testing_speaker_id


def _scan_existing_profile_ids_on_disk() -> list[str]:
    if not TESTING_REGISTRY_PROFILES.exists():
        return []

    ids: list[str] = []
    for child in TESTING_REGISTRY_PROFILES.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("test_spk_"):
            ids.append(child.name)

    ids.sort()
    return ids


def _extract_numeric_suffix(testing_speaker_id: str) -> int | None:
    if not testing_speaker_id.startswith("test_spk_"):
        return None

    suffix = testing_speaker_id.split("_")[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _ensure_registry_structure() -> None:
    TESTING_REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
    TESTING_REGISTRY_PROFILES.mkdir(parents=True, exist_ok=True)

    if not TESTING_REGISTRY_METADATA.exists():
        _write_json(
            TESTING_REGISTRY_METADATA,
            {
                "registry_type": "testing_speakers",
                "version": "v1",
                "source_of_truth": False,
                "description": "Temporary biometric registry for cross-episode speaker matching tests.",
                "updated_at": _utc_placeholder(),
            },
        )

    if not TESTING_REGISTRY_INDEX.exists():
        _write_json(TESTING_REGISTRY_INDEX, {"version": "v1", "profiles": []})

    _reconcile_index_with_disk()


def _load_index() -> dict[str, Any]:
    _ensure_registry_structure()
    index = _read_json(TESTING_REGISTRY_INDEX, {"version": "v1", "profiles": []})

    if not isinstance(index, dict):
        raise RuntimeError("Testing registry index.json must be a JSON object.")

    profiles = index.get("profiles")
    if not isinstance(profiles, list):
        raise RuntimeError("Testing registry index.json field 'profiles' must be a list.")

    return index


def _save_index(index: dict[str, Any]) -> None:
    _write_json(TESTING_REGISTRY_INDEX, index)


def _reconcile_index_with_disk() -> None:
    """
    Garantiza que index.json refleje al menos todos los perfiles existentes en disco.
    Nunca borra perfiles del índice automáticamente.
    """
    index = _read_json(TESTING_REGISTRY_INDEX, {"version": "v1", "profiles": []})
    if not isinstance(index, dict):
        index = {"version": "v1", "profiles": []}

    profiles = index.get("profiles")
    if not isinstance(profiles, list):
        profiles = []

    by_id: dict[str, dict[str, Any]] = {}
    for item in profiles:
        if not isinstance(item, dict):
            continue
        testing_speaker_id = str(item.get("testing_speaker_id") or "").strip()
        if not testing_speaker_id:
            continue
        by_id[testing_speaker_id] = item

    changed = False

    for testing_speaker_id in _scan_existing_profile_ids_on_disk():
        if testing_speaker_id in by_id:
            continue

        profile_dir = _profile_dir_for(testing_speaker_id)
        canonical_embedding = _read_json(profile_dir / "canonical_embedding.json", {})
        episodes_json = _read_json(profile_dir / "episodes.json", {"episodes": []})

        total_speech_seconds = float(
            canonical_embedding.get("total_speech_seconds_used") or 0.0
        )
        num_episodes = len(episodes_json.get("episodes", [])) if isinstance(
            episodes_json.get("episodes"), list
        ) else 0

        by_id[testing_speaker_id] = {
            "testing_speaker_id": testing_speaker_id,
            "status": "active",
            "num_episodes": num_episodes,
            "total_speech_seconds": total_speech_seconds,
            "path": str(profile_dir),
        }
        changed = True

    if changed:
        merged_profiles = sorted(
            by_id.values(),
            key=lambda item: _extract_numeric_suffix(str(item.get("testing_speaker_id") or "")) or 0,
        )
        _save_index({"version": "v1", "profiles": merged_profiles})


def _next_testing_speaker_id() -> str:
    """
    Calcula el siguiente ID usando el máximo observado en disco y en index.json.
    Así evitamos reciclar IDs incluso si el índice quedó desfasado.
    """
    _ensure_registry_structure()
    index = _load_index()

    max_id = 0

    for testing_speaker_id in _scan_existing_profile_ids_on_disk():
        numeric = _extract_numeric_suffix(testing_speaker_id)
        if numeric is not None:
            max_id = max(max_id, numeric)

    for item in index.get("profiles", []):
        if not isinstance(item, dict):
            continue
        testing_speaker_id = str(item.get("testing_speaker_id") or "").strip()
        numeric = _extract_numeric_suffix(testing_speaker_id)
        if numeric is not None:
            max_id = max(max_id, numeric)

    return f"test_spk_{max_id + 1:06d}"


def _append_match_history(profile_dir: Path, payload: dict[str, Any]) -> None:
    path = profile_dir / "match_history.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_canonical_embedding_from_episode_profile(profile: dict[str, Any]) -> dict[str, Any]:
    embedding_primary = profile.get("embedding_primary") or {}
    dispersion = profile.get("intra_speaker_dispersion") or {}

    return {
        "model": embedding_primary.get("model"),
        "vector": embedding_primary.get("vector"),
        "aggregation_method": embedding_primary.get("method"),
        "num_episode_profiles_used": 1,
        "total_segments_used": int(profile.get("num_segments_selected") or 0),
        "total_speech_seconds_used": float(profile.get("speech_seconds_selected") or 0.0),
        "intra_registry_dispersion_mean": dispersion.get("mean"),
        "intra_registry_dispersion_p95": dispersion.get("p95"),
        "profile_quality": classify_profile_quality(profile),
        "updated_at": _utc_placeholder(),
    }


def _merge_vectors(
    existing: list[float],
    new: list[float],
    weight_existing: float,
    weight_new: float,
) -> list[float]:
    total = weight_existing + weight_new
    if total <= 0.0:
        return existing

    if len(existing) != len(new):
        raise RuntimeError("Cannot merge registry vectors with different dimensions.")

    return [
        ((a * weight_existing) + (b * weight_new)) / total
        for a, b in zip(existing, new)
    ]


def load_testing_registry_profiles() -> list[dict[str, Any]]:
    _ensure_registry_structure()
    index = _load_index()
    results: list[dict[str, Any]] = []

    for item in index.get("profiles", []):
        if not isinstance(item, dict):
            continue

        testing_speaker_id = str(item.get("testing_speaker_id") or "").strip()
        if not testing_speaker_id:
            continue

        profile_dir = _profile_dir_for(testing_speaker_id)
        if not profile_dir.exists():
            continue

        profile_json = _read_json(profile_dir / "profile.json", {})
        canonical_embedding = _read_json(profile_dir / "canonical_embedding.json", {})
        episodes_json = _read_json(profile_dir / "episodes.json", {"episodes": []})

        results.append(
            {
                "testing_speaker_id": profile_json.get("testing_speaker_id", testing_speaker_id),
                "display_label": profile_json.get("display_label", testing_speaker_id),
                "profile_quality": canonical_embedding.get("profile_quality"),
                "num_episodes": len(episodes_json.get("episodes", []))
                if isinstance(episodes_json.get("episodes"), list)
                else 0,
                "total_speech_seconds": float(
                    canonical_embedding.get("total_speech_seconds_used") or 0.0
                ),
                "canonical_embedding": canonical_embedding,
                "path": str(profile_dir),
            }
        )

    return results


def create_testing_profile_from_episode_speaker(
    *,
    episode_id: str,
    speaker: str,
    profile: dict[str, Any],
) -> str:
    _ensure_registry_structure()

    testing_speaker_id = _next_testing_speaker_id()
    profile_dir = _profile_dir_for(testing_speaker_id)

    if profile_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing testing registry profile: {testing_speaker_id}"
        )

    profile_dir.mkdir(parents=True, exist_ok=False)

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
                "speech_seconds_selected": float(profile.get("speech_seconds_selected") or 0.0),
                "num_segments_used": int(profile.get("num_segments_selected") or 0),
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

    index = _load_index()
    profiles = index.get("profiles", [])
    if not isinstance(profiles, list):
        raise RuntimeError("Testing registry index 'profiles' must be a list.")

    # seguridad extra: no duplicar entrada en índice
    for item in profiles:
        if isinstance(item, dict) and item.get("testing_speaker_id") == testing_speaker_id:
            raise RuntimeError(
                f"Refusing to duplicate testing registry index entry: {testing_speaker_id}"
            )

    profiles.append(
        {
            "testing_speaker_id": testing_speaker_id,
            "status": "active",
            "num_episodes": 1,
            "total_speech_seconds": float(profile.get("speech_seconds_selected") or 0.0),
            "path": str(profile_dir),
        }
    )

    profiles.sort(
        key=lambda item: _extract_numeric_suffix(str(item.get("testing_speaker_id") or "")) or 0
    )
    index["profiles"] = profiles
    _save_index(index)

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

    profile_dir = _profile_dir_for(testing_speaker_id)
    if not profile_dir.exists():
        raise RuntimeError(
            f"Testing registry profile directory does not exist: {testing_speaker_id}"
        )

    profile_json = _read_json(profile_dir / "profile.json", {})
    canonical_embedding = _read_json(profile_dir / "canonical_embedding.json", {})
    episodes_json = _read_json(profile_dir / "episodes.json", {"episodes": []})

    existing_vector = canonical_embedding.get("vector")
    new_vector = (profile.get("embedding_primary") or {}).get("vector")

    if not isinstance(existing_vector, list) or not existing_vector:
        raise RuntimeError("Existing testing registry vector is missing or invalid.")
    if not isinstance(new_vector, list) or not new_vector:
        raise RuntimeError("New episode profile vector is missing or invalid.")

    existing_seconds = float(canonical_embedding.get("total_speech_seconds_used") or 0.0)
    new_seconds = float(profile.get("speech_seconds_selected") or 0.0)

    merged_vector = _merge_vectors(existing_vector, new_vector, existing_seconds, new_seconds)

    episodes = episodes_json.get("episodes")
    if not isinstance(episodes, list):
        raise RuntimeError("Testing registry episodes.json field 'episodes' must be a list.")

    # seguridad: no duplicar el mismo episodio+speaker dentro del mismo bucket
    for item in episodes:
        if not isinstance(item, dict):
            continue
        if item.get("episode_id") == episode_id and item.get("speaker") == speaker:
            raise RuntimeError(
                f"Duplicate episode speaker detected in testing registry: "
                f"{testing_speaker_id} / {episode_id} / {speaker}"
            )

    episodes.append(
        {
            "episode_id": episode_id,
            "speaker": speaker,
            "speech_seconds_selected": float(profile.get("speech_seconds_selected") or 0.0),
            "num_segments_used": int(profile.get("num_segments_selected") or 0),
            "profile_quality": classify_profile_quality(profile),
            "embedding_status": (profile.get("embedding_primary") or {}).get("status"),
        }
    )

    canonical_embedding.update(
        {
            "vector": merged_vector,
            "num_episode_profiles_used": len(episodes),
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

    index = _load_index()
    profiles = index.get("profiles", [])
    if not isinstance(profiles, list):
        raise RuntimeError("Testing registry index 'profiles' must be a list.")

    found = False
    for item in profiles:
        if not isinstance(item, dict):
            continue
        if item.get("testing_speaker_id") == testing_speaker_id:
            item["num_episodes"] = len(episodes)
            item["total_speech_seconds"] = canonical_embedding.get(
                "total_speech_seconds_used", 0.0
            )
            item["path"] = str(profile_dir)
            found = True
            break

    if not found:
        profiles.append(
            {
                "testing_speaker_id": testing_speaker_id,
                "status": "active",
                "num_episodes": len(episodes),
                "total_speech_seconds": canonical_embedding.get(
                    "total_speech_seconds_used", 0.0
                ),
                "path": str(profile_dir),
            }
        )

    profiles.sort(
        key=lambda item: _extract_numeric_suffix(str(item.get("testing_speaker_id") or "")) or 0
    )
    index["profiles"] = profiles
    _save_index(index)
