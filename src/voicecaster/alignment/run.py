from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import (
    ALIGNMENT_MAX_UTTERANCE_CHARS,
    ALIGNMENT_MAX_UTTERANCE_SECONDS,
    ALIGNMENT_MERGE_GAP_SECONDS,
    MAX_RETRIES,
    TURN_MAX_CHARS,
    TURN_MAX_DURATION_SECONDS,
    TURN_MERGE_GAP_SECONDS,
)
from .io import (
    ensure_required_paths,
    increment_retries,
    load_diarization_metadata,
    load_speaker_metrics,
    load_speaker_segments,
    load_transcript_segments,
    load_transcript_with_speakers,
    mark_ruined,
    select_episode,
    update_episode_status,
)
from .normalize import normalize_utterances
from .qa import run_alignment_qa
from .turns import build_turns
from .words import flatten_aligned_words
from .write_outputs import (
    ensure_alignment_dir,
    write_aligned_turns_json,
    write_aligned_utterances_json,
    write_aligned_words_json,
    write_alignment_metadata_json,
    write_alignment_result_json,
    write_qa_summary_json,
    write_speakers_index_json,
    write_subtitles_final_srt,
    write_transcript_final_txt,
)


WORK_DIR = Path("work")


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_speakers_index(
    utterances: list[Any],
    turns: list[Any],
    words: list[Any],
) -> dict[str, Any]:
    utterance_counts: dict[str, int] = {}
    turn_counts: dict[str, int] = {}
    word_counts: dict[str, int] = {}
    speech_seconds: dict[str, float] = {}
    first_seen: dict[str, float] = {}
    last_seen: dict[str, float] = {}

    total_speech = 0.0

    for utt in utterances:
        speaker = utt.speaker or "unknown"
        utterance_counts[speaker] = utterance_counts.get(speaker, 0) + 1
        speech_seconds[speaker] = speech_seconds.get(speaker, 0.0) + float(utt.duration)
        total_speech += float(utt.duration)

        if speaker not in first_seen or utt.start < first_seen[speaker]:
            first_seen[speaker] = utt.start
        if speaker not in last_seen or utt.end > last_seen[speaker]:
            last_seen[speaker] = utt.end

    for turn in turns:
        speaker = turn.speaker or "unknown"
        turn_counts[speaker] = turn_counts.get(speaker, 0) + 1

    for word in words:
        speaker = word.speaker or "unknown"
        word_counts[speaker] = word_counts.get(speaker, 0) + 1

    speakers_payload = []
    all_speakers = sorted(set(utterance_counts) | set(turn_counts) | set(word_counts))

    for speaker in all_speakers:
        seconds = round(speech_seconds.get(speaker, 0.0), 3)
        share = round((seconds / total_speech), 4) if total_speech > 0 else 0.0
        speakers_payload.append(
            {
                "speaker": speaker,
                "num_turns": turn_counts.get(speaker, 0),
                "num_utterances": utterance_counts.get(speaker, 0),
                "num_words": word_counts.get(speaker, 0),
                "speech_seconds": seconds,
                "first_seen": first_seen.get(speaker),
                "last_seen": last_seen.get(speaker),
                "share_of_speech": share,
            }
        )

    return {"speakers": speakers_payload}


def main() -> int:
    try:
        episode = select_episode()
    except Exception:
        print("[alignment] No work to do.")
        return 0

    episode_id = episode["id"]
    work_episode_dir = WORK_DIR / episode_id
    alignment_dir = ensure_alignment_dir(work_episode_dir)

    print(f"[alignment] Processing episode: {episode_id}")

    try:
        ensure_required_paths(work_episode_dir)

        transcript_segments = load_transcript_segments(work_episode_dir)
        transcript_with_speakers = load_transcript_with_speakers(work_episode_dir)
        speaker_segments = load_speaker_segments(work_episode_dir)
        speaker_metrics = load_speaker_metrics(work_episode_dir)
        diarization_metadata = load_diarization_metadata(work_episode_dir)

        utterances, normalization_report = normalize_utterances(
            transcript_with_speakers,
            merge_gap_seconds=ALIGNMENT_MERGE_GAP_SECONDS,
            max_utterance_seconds=ALIGNMENT_MAX_UTTERANCE_SECONDS,
            max_utterance_chars=ALIGNMENT_MAX_UTTERANCE_CHARS,
        )

        turns, turns_report = build_turns(
            utterances,
            merge_gap_seconds=TURN_MERGE_GAP_SECONDS,
            max_duration_seconds=TURN_MAX_DURATION_SECONDS,
            max_chars=TURN_MAX_CHARS,
        )

        words = flatten_aligned_words(utterances, turns)

        qa_result = run_alignment_qa(utterances, turns, words)
        if not qa_result.passed:
            raise RuntimeError(
                f"Alignment QA failed: {[issue.to_dict() for issue in qa_result.issues]}"
            )

        speakers_index = build_speakers_index(utterances, turns, words)

        metadata_payload = {
            "workflow": "alignment",
            "version": "v1",
            "inputs": {
                "transcript_segments_found": True,
                "transcript_with_speakers_found": True,
                "speaker_segments_found": True,
            },
            "counts": {
                "input_transcript_segments": len(transcript_segments),
                "input_transcript_with_speakers": len(transcript_with_speakers),
                "input_speaker_segments": len(speaker_segments),
                "output_utterances": len(utterances),
                "output_turns": len(turns),
                "output_words": len(words),
                "speakers_detected": len(speakers_index["speakers"]),
            },
            "normalization": normalization_report,
            "turns": turns_report,
            "qa": qa_result.to_dict(),
            "upstream": {
                "speaker_metrics": speaker_metrics,
                "diarization_metadata_summary": {
                    "assignment_stats": diarization_metadata.get("assignment_stats"),
                    "qa": diarization_metadata.get("qa"),
                },
            },
        }

        write_aligned_utterances_json(alignment_dir, utterances)
        write_aligned_turns_json(alignment_dir, turns)
        write_aligned_words_json(alignment_dir, words)
        write_subtitles_final_srt(alignment_dir, utterances)
        write_transcript_final_txt(alignment_dir, turns)
        write_speakers_index_json(alignment_dir, speakers_index)
        write_alignment_metadata_json(alignment_dir, metadata_payload)
        write_qa_summary_json(alignment_dir, qa_result)

        write_alignment_result_json(
            alignment_dir,
            {
                "result": "success",
                "status_before": "alignment",
                "status_after": "completed",
                "retries_before": int(episode.get("retries", 0)),
                "retries_after": 0,
                "message": "Alignment completed successfully.",
                "finished_at": utc_now_iso(),
            },
        )

        update_episode_status(episode_id, "completed")

        print("[alignment] SUCCESS")
        return 0

    except Exception as exc:
        print(f"[alignment] ERROR: {repr(exc)}")

        retries_after = increment_retries(episode_id)
        if retries_after > MAX_RETRIES:
            mark_ruined(episode_id)

        try:
            write_alignment_result_json(
                alignment_dir,
                {
                    "result": "error",
                    "status_before": "alignment",
                    "status_after": "ruined" if retries_after > MAX_RETRIES else "alignment",
                    "retries_before": int(episode.get("retries", 0)),
                    "retries_after": retries_after,
                    "error": repr(exc),
                    "retry_consumed": True,
                    "finished_at": utc_now_iso(),
                },
            )
        except Exception as write_exc:
            print(f"[alignment] Failed to write alignment_result.json: {repr(write_exc)}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
