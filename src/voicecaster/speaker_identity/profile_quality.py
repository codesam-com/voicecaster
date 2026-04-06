from __future__ import annotations

from typing import Any


def classify_profile_quality(profile: dict[str, Any]) -> str:
    speech_seconds_selected = float(profile.get("speech_seconds_selected") or 0.0)
    num_segments_selected = int(profile.get("num_segments_selected") or 0)

    dispersion = profile.get("intra_speaker_dispersion") or {}
    dispersion_mean = dispersion.get("mean")

    if dispersion_mean is None:
        return "insufficient_profile"

    dispersion_mean = float(dispersion_mean)

    if (
        speech_seconds_selected >= 45.0
        and num_segments_selected >= 5
        and dispersion_mean <= 0.10
    ):
        return "strong_profile"

    if (
        speech_seconds_selected >= 20.0
        and num_segments_selected >= 3
        and dispersion_mean <= 0.16
    ):
        return "usable_profile"

    if (
        speech_seconds_selected >= 8.0
        and num_segments_selected >= 2
        and dispersion_mean <= 0.25
    ):
        return "weak_profile"

    return "insufficient_profile"


def is_profile_eligible_for_testing_registry(profile: dict[str, Any]) -> bool:
    embedding_primary = profile.get("embedding_primary") or {}
    status = embedding_primary.get("status")
    vector = embedding_primary.get("vector")

    if status != "ok":
        return False
    if not isinstance(vector, list) or not vector:
        return False

    quality = classify_profile_quality(profile)
    return quality in {"strong_profile", "usable_profile"}
