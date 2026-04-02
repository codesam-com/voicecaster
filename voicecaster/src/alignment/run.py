from __future__ import annotations

from .assign_segments import assign_speakers_to_segments
from .assign_words import assign_speakers_to_words
from .export_srt import write_speaker_srt
from .loader import (
    build_alignment_paths,
    load_speaker_segments_json,
    load_transcript_preview_json,
    validate_raw_speaker_segments,
    validate_raw_transcript_preview,
    validate_required_files,
)
from .metrics import build_alignment_preview, compute_alignment_metadata
from .normalizer import (
    normalize_speaker_segments,
    normalize_transcript_segments,
    validate_monotonic_timeline,
)
from .schemas import AlignmentConfig
from .split_merge import build_aligned_utterances


def process_episode_alignment(episode_dir):
    paths = build_alignment_paths(episode_dir)
    validate_required_files(paths)

    raw_transcript = load_transcript_preview_json(paths.transcript_preview_path)
    raw_speakers = load_speaker_segments_json(paths.speaker_segments_path)

    validate_raw_transcript_preview(raw_transcript)
    validate_raw_speaker_segments(raw_speakers)

    transcript_segments = normalize_transcript_segments(raw_transcript)
    speaker_segments = normalize_speaker_segments(raw_speakers)
    validate_monotonic_timeline(transcript_segments, speaker_segments)

    config = AlignmentConfig()

    aligned_words = assign_speakers_to_words(
        transcript_segments=transcript_segments,
        speaker_segments=speaker_segments,
        config=config,
    )

    segment_assignments = assign_speakers_to_segments(
        transcript_segments=transcript_segments,
        aligned_words=aligned_words,
        speaker_segments=speaker_segments,
    )

    utterances = build_aligned_utterances(
        transcript_segments=transcript_segments,
        aligned_words=aligned_words,
        segment_assignments=segment_assignments,
        config=config,
    )

    metadata = compute_alignment_metadata(
        transcript_segments=transcript_segments,
        speaker_segments=speaker_segments,
        aligned_words=aligned_words,
        utterances=utterances,
    )

    preview = build_alignment_preview(
        utterances=utterances,
        metadata=metadata,
        preview_limit=config.preview_limit,
    )

    paths.alignment_dir.mkdir(parents=True, exist_ok=True)

    import json

    paths.aligned_words_path.write_text(
        json.dumps(
            {
                "algorithm_version": "04_alignment_v1",
                "words": [word.to_dict() for word in aligned_words],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    paths.aligned_utterances_path.write_text(
        json.dumps(
            {
                "algorithm_version": "04_alignment_v1",
                "utterances": [utterance.to_dict() for utterance in utterances],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    paths.alignment_metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    paths.alignment_preview_path.write_text(
        json.dumps(preview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    paths.alignment_result_path.write_text(
        json.dumps(
            {
                "result": "success",
                "status_before": "alignment",
                "status_after": "completed",
                "message": "Alignment completed successfully.",
                "files_generated": [
                    "aligned_words.json",
                    "aligned_utterances.json",
                    "alignment_metadata.json",
                    "alignment_preview.json",
                    "alignment_result.json",
                    "subtitles_speakers.srt",
                ],
                "warnings": metadata.get("warnings", []),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_speaker_srt(paths.subtitles_speakers_path, utterances)


def run_alignment() -> None:
    raise NotImplementedError(
        "run_alignment() orchestration with episode selection/update is pending integration "
        "with the existing project runtime/status manager."
    )
