from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

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
    quality_summary: dict[str, Any]
    auxiliary_voice_traits: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_episode_speaker_voice_profile(
    *,
    speaker: str,
    speech_seconds_total: float,
    selected_segments: list[SelectedSegment],
    total_candidate_segments: int,
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

    quality_summary = {
        "selected_segments": num_segments_selected,
        "rejected_segments": num_segments_rejected,
        "selection_strategy": "temporal_coverage_plus_duration_fill",
        "selected_speech_seconds": speech_seconds_selected,
    }

    auxiliary_voice_traits = {
        "language_hint": language_hint,
        "speech_rate_words_per_second": speech_rate_words_per_second,
        "pitch_mean": None,
        "pitch_range": None,
        "pause_ratio": None,
        "status": "partial_profile_v1",
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
        quality_summary=quality_summary,
        auxiliary_voice_traits=auxiliary_voice_traits,
    )
