from __future__ import annotations

from typing import Any


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def classify_match_band(score: float | None) -> str:
    if score is None:
        return "no_match"

    if score >= 0.85:
        return "high_match"

    if score >= 0.78:
        return "candidate_match"

    return "no_match"


def rank_testing_registry_matches(
    profile: dict[str, Any],
    registry_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    embedding_primary = profile.get("embedding_primary") or {}
    source_vector = embedding_primary.get("vector")

    if not isinstance(source_vector, list) or not source_vector:
        return []

    results: list[dict[str, Any]] = []

    for registry_profile in registry_profiles:
        canonical_embedding = registry_profile.get("canonical_embedding") or {}
        target_vector = canonical_embedding.get("vector")

        if not isinstance(target_vector, list) or not target_vector:
            continue

        score = round(cosine_similarity(source_vector, target_vector), 6)

        results.append(
            {
                "testing_speaker_id": registry_profile.get("testing_speaker_id"),
                "display_label": registry_profile.get("display_label"),
                "score": score,
                "match_band": classify_match_band(score),
                "registry_num_episodes": registry_profile.get("num_episodes"),
                "registry_total_speech_seconds": registry_profile.get("total_speech_seconds"),
                "registry_profile_quality": registry_profile.get("profile_quality"),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results
