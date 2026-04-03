# =========================================
# FILE: src/voicecaster/alignment/run.py
# =========================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .assign_segments import assign_speakers_to_segments
from .assign_words import assign_speakers_to_words
from .export_srt import write_speaker_srt
from .loader import (
    AlignmentInputError,
    build_alignment_paths,
    ensure_stage_dir,
    load_speaker_segments,
    load_transcript_preview,
    validate_required_inputs,
)
from .metrics import compute_alignment_metrics
from .normalizer import (
    AlignmentNormalizationError,
    normalize_speaker_segments,
    normalize_transcript_segments,
)
from .schemas import AlignmentMetrics, dataclass_to_dict
from .split_merge import (
    merge_adjacent_same_speaker_utterances,
    split_segments_into_utterances,
)

ALIGNMENT_ALGORITHM_VERSION = "04_alignment_v1"

NEAREST_SPEAKER_GAP_TOLERANCE = 0.35
MIN_WORDS_PER_SPLIT_CHUNK = 2
MIN_CHUNK_DURATION = 0.35
NOISE_BRIDGE_MAX_DURATION = 0.30
MERGE_SAME_SPEAKER_GAP_MAX = 0.60

HIGH_UNKNOWN_WORD_RATIO_WARNING = 0.01
HIGH_UNKNOWN_UTTERANCE_RATIO_WARNING = 0.01
HIGH_MULTI_SPEAKER_SEGMENT_RATIO_WARNING = 0.15


def run_alignment() -> int:
    """
    Entry point for 04_alignment.
    Current version expects an explicit episode id to be wired by the caller.
    In repository integration, this should be connected to the same selector pattern
    used by the rest of the workflows.
    """
    raise NotImplementedError(
        "run_alignment() must be integrated with the repo workflow selector "
        "that picks the first episode with status='alignment'."
    )


def process_episode(episode_id: str, work_root: Path) -> dict[str, Any]:
    """
    Pure alignment orchestration for one already-selected episode.
    """
    paths = build_alignment_paths(work_root=work_root, episode_id=episode_id)
    ensure_stage_dir(paths)

    transcript_raw = load_transcript_preview(paths.transcript_preview_json)
    speakers_raw = load_speaker_segments(paths.speaker_segments_json)
    validate_required_inputs(transcript_raw, speakers_raw)

    transcript_segments = normalize_transcript_segments(transcript_raw)
    speaker_segments = normalize_speaker_segments(speakers_raw)

    aligned_words = assign_speakers_to_words(
        transcript_segments=transcript_segments,
        speaker_segments=speaker_segments,
        nearest_gap_tolerance=NEAREST_SPEAKER_GAP_TOLERANCE,
    )

    segment_assignments = assign_speakers_to_segments(
        transcript_segments=transcript_segments,
        aligned_words=aligned_words,
        speaker_segments=speaker_segments,
    )

    utterances = split_segments_into_utterances(
        transcript_segments=transcript_segments,
        aligned_words=aligned_words,
        segment_assignments=segment_assignments,
        min_words_per_chunk=MIN_WORDS_PER_SPLIT_CHUNK,
        min_chunk_duration=MIN_CHUNK_DURATION,
        noise_bridge_max_duration=NOISE_BRIDGE_MAX_DURATION,
    )

    utterances = merge_adjacent_same_speaker_utterances(
        utterances=utterances,
        max_gap=MERGE_SAME_SPEAKER_GAP_MAX,
    )

    metrics = compute_alignment_metrics(
        transcript_segments=transcript_segments,
        speaker_segments=speaker_segments,
        aligned_words=aligned_words,
        utterances=utterances,
        algorithm_version=ALIGNMENT_ALGORITHM_VERSION,
        high_unknown_word_ratio_warning=HIGH_UNKNOWN_WORD_RATIO_WARNING,
        high_unknown_utterance_ratio_warning=HIGH_UNKNOWN_UTTERANCE_RATIO_WARNING,
        high_multi_speaker_segment_ratio_warning=HIGH_MULTI_SPEAKER_SEGMENT_RATIO_WARNING,
    )

    _write_json(paths.aligned_words_json, {
        "episode_id": episode_id,
        "algorithm_version": ALIGNMENT_ALGORITHM_VERSION,
        "words": dataclass_to_dict(aligned_words),
    })

    _write_json(paths.aligned_utterances_json, {
        "episode_id": episode_id,
        "algorithm_version": ALIGNMENT_ALGORITHM_VERSION,
        "utterances": dataclass_to_dict(utterances),
    })

    write_speaker_srt(paths.subtitles_speakers_srt, utterances)

    _write_json(paths.alignment_metadata_json, dataclass_to_dict(metrics))

    _write_json(paths.alignment_preview_json, _build_alignment_preview(episode_id, utterances, metrics))

    _write_json(paths.alignment_result_json, {
        "result": "success",
        "episode_id": episode_id,
        "algorithm_version": ALIGNMENT_ALGORITHM_VERSION,
        "files_generated": [
            str(paths.aligned_words_json.name),
            str(paths.aligned_utterances_json.name),
            str(paths.subtitles_speakers_srt.name),
            str(paths.alignment_metadata_json.name),
            str(paths.alignment_result_json.name),
            str(paths.alignment_preview_json.name),
        ],
        "warnings": metrics.warnings,
    })

    return {
        "episode_id": episode_id,
        "algorithm_version": ALIGNMENT_ALGORITHM_VERSION,
        "metrics": dataclass_to_dict(metrics),
    }


def _build_alignment_preview(
    episode_id: str,
    utterances: list[Any],
    metrics: AlignmentMetrics,
    preview_size: int = 50,
) -> dict[str, Any]:
    """
    Lightweight preview for quick inspection.
    """
    speaker_counts: dict[str, int] = {}
    for utt in utterances:
        speaker_counts[utt.speaker] = speaker_counts.get(utt.speaker, 0) + 1

    return {
        "episode_id": episode_id,
        "algorithm_version": metrics.algorithm_version,
        "summary": {
            "aligned_utterances": len(utterances),
            "warnings": metrics.warnings,
            "speakers": speaker_counts,
        },
        "utterances_preview": dataclass_to_dict(utterances[:preview_size]),
    }


def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        raise ValueError("Output path cannot be None")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def process_episode_safe(episode_id: str, work_root: Path) -> tuple[bool, dict[str, Any]]:
    """
    Safe wrapper useful for workflow integration.
    """
    try:
        result = process_episode(episode_id=episode_id, work_root=work_root)
        return True, result
    except (AlignmentInputError, AlignmentNormalizationError) as exc:
        return False, {
            "result": "failed",
            "episode_id": episode_id,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
