from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .biometric_extractor import AggregatedBiometricProfile, SegmentEmbedding
from .segment_selector import SelectedSegment


@dataclass(slots=True)
class EpisodeSpeakerVoiceProfile:
    speaker: str
    speech_seconds_total: float
    speech_seconds_selected: float
    num_segments_selected: int
    num_segments_rejected: int
    num_words_selected: int
    speech_rate_words_per_second: float
    first_selected_start: float | None
    last_selected_end: float | None
    temporal_coverage_seconds: float | None
    usable_for_identity: bool
    embedding_primary: dict[str, Any] | None
    embedding_secondary: dict[str, Any] | None
    intra_speaker_dispersion: dict[str, Any] | None
    quality_summary: dict[str, Any]
    auxiliary_voice_traits: dict[str, Any]
    segment_embeddings: list[dict[str, Any]]
    biometric_profile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_episode_speaker_voice_profile(
    *,
    speaker: str,
    speech_seconds_total: float,
    selected_segments: list[SelectedSegment],
    total_candidate_segments: int,
    segment_embeddings: list[SegmentEmbedding],
    biometric_profile: AggregatedBiometricProfile,
    language_hint: str | None = None,
) -> EpisodeSpeakerVoiceProfile:
    num_segments_selected = len(selected_segments)
    num_segments_rejected = max(0, total_candidate_segments - num_segments_selected)

    speech_seconds_selected = round(sum(seg.duration for seg in selected_segments), 3)
    num_words_selected = sum(seg.num_words for seg in selected_segments)

    speech_rate_words_per_second = (
        round(num_words_selected / speech_seconds_selected, 4)
        if speech_seconds_selected > 0
        else 0.0
    )

    first_selected_start = selected_segments[0].start if selected_segments else None
    last_selected_end = selected_segments[-1].end if selected_segments else None

    temporal_coverage_seconds = None
    if first_selected_start is not None and last_selected_end is not None:
        temporal_coverage_seconds = round(last_selected_end - first_selected_start, 3)

    usable_for_identity = speech_seconds_selected > 0 and num_words_selected > 0

    embedding_primary = {
        "model": biometric_profile.primary_model,
        "vector": biometric_profile.primary_embedding,
        "method": biometric_profile.aggregation_method,
        "status": biometric_profile.status,
    }

    embedding_secondary = {
        "model": biometric_profile.secondary_model,
        "vector": biometric_profile.secondary_embedding,
        "method": biometric_profile.aggregation_method,
        "status": biometric_profile.status,
    }

    intra_speaker_dispersion = {
        "mean": biometric_profile.intra_speaker_dispersion_mean,
        "p95": biometric_profile.intra_speaker_dispersion_p95,
        "status": biometric_profile.status,
    }

    quality_summary = {
        "selected_segments": num_segments_selected,
        "rejected_segments": num_segments_rejected,
        "selection_strategy": "temporal_coverage_plus_duration_fill",
        "selected_speech_seconds": speech_seconds_selected,
        "embedding_segments": len(segment_embeddings),
        "embedding_backend_status": biometric_profile.status,
    }

    auxiliary_voice_traits = {
        "language_hint": language_hint,
        "speech_rate_words_per_second": speech_rate_words_per_second,
        "pitch_mean": None,
        "pitch_range": None,
        "pause_ratio": None,
        "status": "partial_profile_v2",
    }

    return EpisodeSpeakerVoiceProfile(
        speaker=speaker,
        speech_seconds_total=round(speech_seconds_total, 3),
        speech_seconds_selected=speech_seconds_selected,
        num_segments_selected=num_segments_selected,
        num_segments_rejected=num_segments_rejected,
        num_words_selected=num_words_selected,
        speech_rate_words_per_second=speech_rate_words_per_second,
        first_selected_start=first_selected_start,
        last_selected_end=last_selected_end,
        temporal_coverage_seconds=temporal_coverage_seconds,
        usable_for_identity=usable_for_identity,
        embedding_primary=embedding_primary,
        embedding_secondary=embedding_secondary,
        intra_speaker_dispersion=intra_speaker_dispersion,
        quality_summary=quality_summary,
        auxiliary_voice_traits=auxiliary_voice_traits,
        segment_embeddings=[item.to_dict() for item in segment_embeddings],
        biometric_profile=biometric_profile.to_dict(),
    )
