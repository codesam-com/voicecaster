from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

from voicecaster.transcription.run import ContentError, download_file, ffprobe_audio

from .biometric_extractor import aggregate_biometric_profile, extract_segment_embeddings
from .config import (
    MAX_RETRIES,
    MAX_SELECTED_SEGMENTS_PER_SPEAKER,
    MIN_NUM_WORDS_FOR_ANALYSIS,
    MIN_SELECTED_SEGMENT_SECONDS,
    MIN_SELECTED_SEGMENT_WORDS,
    MIN_SPEECH_SECONDS_FOR_ANALYSIS,
    SUCCESS_STATUS,
    TARGET_STATUS,
    UNKNOWN_DISPLAY_NAME,
)
from .loader import (
    ensure_identity_dir,
    ensure_required_paths,
    get_work_episode_dir,
    increment_retries,
    load_aligned_utterances,
    load_alignment_metadata,
    load_episode_inputs_record,
    load_speaker_metrics,
    load_speakers_index,
    load_transcript_preview_if_exists,
    load_transcript_with_speakers,
    mark_ruined,
    select_episode,
    update_episode_status,
)
from .qa import run_identity_qa
from .schemas import IdentityCandidate, IdentityDecision, SpeakerEpisodeSummary
from .segment_selector import select_segments_for_speaker
from .text_support import SpeakerTextEvidence, build_text_evidence
from .voice_profile import EpisodeSpeakerVoiceProfile, build_episode_speaker_voice_profile
from .write_outputs import (
    utc_now_iso,
    write_biometric_summary_json,
    write_identity_candidates_json,
    write_identity_evidence_json,
    write_identity_result_json,
    write_qa_summary_json,
    write_speaker_identity_json,
    write_speaker_identity_metadata_json,
    write_speaker_identity_request_json,
    write_speaker_voice_profiles_json,
    write_text_support_json,
)


def _download_audio_for_speaker_identity(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    download_file(url, target)
    probe = ffprobe_audio(target)
    if not probe.get("ffprobe_ok", False):
        raise ContentError("Downloaded resource is not a valid audio file for speaker identity.")
    return target


def _build_speaker_summaries(
    speakers_index: dict[str, Any],
) -> list[SpeakerEpisodeSummary]:
    result: list[SpeakerEpisodeSummary] = []

    for item in speakers_index.get("speakers", []):
        speaker = str(item.get("speaker") or "").strip()
        if not speaker:
            continue

        speech_seconds = float(item.get("speech_seconds") or 0.0)
        num_words = int(item.get("num_words") or 0)

        usable_for_identity = (
            speech_seconds >= MIN_SPEECH_SECONDS_FOR_ANALYSIS
            and num_words >= MIN_NUM_WORDS_FOR_ANALYSIS
        )

        identity_state = (
            "hypothesis_new_person" if usable_for_identity else "insufficient_voice_evidence"
        )

        result.append(
            SpeakerEpisodeSummary(
                speaker=speaker,
                speech_seconds=round(speech_seconds, 3),
                num_turns=int(item.get("num_turns") or 0),
                num_utterances=int(item.get("num_utterances") or 0),
                num_words=num_words,
                first_seen=item.get("first_seen"),
                last_seen=item.get("last_seen"),
                share_of_speech=float(item.get("share_of_speech") or 0.0),
                identity_state=identity_state,
                review_required=True,
                usable_for_identity=usable_for_identity,
            )
        )

    return result


def _build_voice_profiles(
    summaries: list[SpeakerEpisodeSummary],
    aligned_utterances: list[dict[str, Any]],
    *,
    language_hint: str | None,
    audio_path: Path,
) -> tuple[list[EpisodeSpeakerVoiceProfile], dict[str, list[dict[str, Any]]]]:
    profiles: list[EpisodeSpeakerVoiceProfile] = []
    selected_segments_by_speaker: dict[str, list[dict[str, Any]]] = {}

    for summary in summaries:
        selected_segments = select_segments_for_speaker(
            summary.speaker,
            aligned_utterances,
            min_segment_seconds=MIN_SELECTED_SEGMENT_SECONDS,
            min_num_words=MIN_SELECTED_SEGMENT_WORDS,
            max_segments=MAX_SELECTED_SEGMENTS_PER_SPEAKER,
        )

        selected_segments_by_speaker[summary.speaker] = [
            seg.to_dict() for seg in selected_segments
        ]

        segment_embeddings = extract_segment_embeddings(
            summary.speaker,
            selected_segments,
            audio_path,
        )

        biometric_profile = aggregate_biometric_profile(
            summary.speaker,
            segment_embeddings,
        )

        profile = build_episode_speaker_voice_profile(
            speaker=summary.speaker,
            speech_seconds_total=summary.speech_seconds,
            selected_segments=selected_segments,
            total_candidate_segments=len(selected_segments),
            segment_embeddings=segment_embeddings,
            biometric_profile=biometric_profile,
            language_hint=language_hint,
        )
        profiles.append(profile)

    return profiles, selected_segments_by_speaker


def _index_text_evidence(
    text_evidence: list[SpeakerTextEvidence],
) -> dict[str, SpeakerTextEvidence]:
    return {item.speaker: item for item in text_evidence}


def _build_candidates_and_decisions_from_profiles(
    profiles: list[EpisodeSpeakerVoiceProfile],
    text_evidence_by_speaker: dict[str, SpeakerTextEvidence],
) -> tuple[dict[str, list[IdentityCandidate]], list[IdentityDecision]]:
    candidates_by_speaker: dict[str, list[IdentityCandidate]] = {}
    decisions: list[IdentityDecision] = []

    for profile in profiles:
        text_ev = text_evidence_by_speaker.get(profile.speaker)
        text_score = text_ev.text_score_hint if text_ev else 0.0

        voice_status = None
        if profile.embedding_primary:
            voice_status = profile.embedding_primary.get("status")

        voice_score = 1.0 if voice_status == "ok" else 0.0

        if profile.usable_for_identity:
            candidate = IdentityCandidate(
                candidate_type="new_hypothetical_identity",
                speaker_id=None,
                display_name=UNKNOWN_DISPLAY_NAME,
                voice_score=voice_score,
                text_score=text_score,
                context_score=0.0,
                final_score=round(0.9 * voice_score + 0.1 * text_score, 4),
                decision_band="review_required",
            )
            decision = IdentityDecision(
                speaker=profile.speaker,
                proposed_identity=None,
                proposed_display_name=UNKNOWN_DISPLAY_NAME,
                confidence=0.0,
                identity_state="hypothesis_new_person",
                review_required=True,
            )
        else:
            candidate = IdentityCandidate(
                candidate_type="unknown",
                speaker_id=None,
                display_name=UNKNOWN_DISPLAY_NAME,
                voice_score=0.0,
                text_score=text_score,
                context_score=0.0,
                final_score=text_score,
                decision_band="insufficient_evidence",
            )
            decision = IdentityDecision(
                speaker=profile.speaker,
                proposed_identity=None,
                proposed_display_name=UNKNOWN_DISPLAY_NAME,
                confidence=0.0,
                identity_state="insufficient_voice_evidence",
                review_required=True,
            )

        candidates_by_speaker[profile.speaker] = [candidate]
        decisions.append(decision)

    return candidates_by_speaker, decisions


def main() -> int:
    temp_audio_path: Path | None = None

    try:
        episode = select_episode()
    except Exception:
        print("[speaker_identity] No work to do.")
        return 0

    episode_id = str(episode["id"])
    work_episode_dir = get_work_episode_dir(episode_id)
    identity_dir = ensure_identity_dir(work_episode_dir)

    print(f"[speaker_identity] Processing episode: {episode_id}")

    try:
        ensure_required_paths(work_episode_dir)

        episode_record = load_episode_inputs_record(episode_id)

        source_url = episode_record.get("url")
        if not isinstance(source_url, str) or not source_url.strip():
            raise RuntimeError("No usable source URL found for speaker_identity.")

        temp_audio_path = work_episode_dir / "99_temp" / "audio_for_speaker_identity"
        audio_path = _download_audio_for_speaker_identity(source_url, temp_audio_path)

        aligned_utterances = load_aligned_utterances(work_episode_dir)
        speakers_index = load_speakers_index(work_episode_dir)
        speaker_metrics = load_speaker_metrics(work_episode_dir)
        alignment_metadata = load_alignment_metadata(work_episode_dir)
        transcript_with_speakers = load_transcript_with_speakers(work_episode_dir)
        transcript_preview = load_transcript_preview_if_exists(work_episode_dir)

        language_hint = None
        if transcript_preview is not None:
            language_hint = transcript_preview.get("language")

        text_evidence = build_text_evidence(
            transcript_with_speakers,
            episode_record.get("participants"),
        )
        text_evidence_by_speaker = _index_text_evidence(text_evidence)

        request_payload = {
            "episode_id": episode_id,
            "status_before": TARGET_STATUS,
            "requested_at": utc_now_iso(),
            "podcast_title": episode_record.get("podcast_title"),
            "episode_title": episode_record.get("episode_title"),
            "participants": episode_record.get("participants"),
            "url": episode_record.get("url"),
        }
        write_speaker_identity_request_json(identity_dir, request_payload)

        summaries = _build_speaker_summaries(speakers_index)
        voice_profiles, selected_segments_by_speaker = _build_voice_profiles(
            summaries,
            aligned_utterances,
            language_hint=language_hint,
            audio_path=audio_path,
        )

        candidates_by_speaker, decisions = _build_candidates_and_decisions_from_profiles(
            voice_profiles,
            text_evidence_by_speaker,
        )

        expected_speakers = {item.speaker for item in summaries}
        qa_result = run_identity_qa(expected_speakers, decisions)
        if not qa_result.passed:
            raise RuntimeError(f"Speaker identity QA failed: {qa_result.to_dict()}")

        metadata_payload = {
            "workflow": "speaker_identity",
            "version": "v2_biometric",
            "inputs": {
                "aligned_utterances_found": True,
                "speakers_index_found": True,
                "speaker_metrics_found": True,
                "alignment_metadata_found": True,
                "transcript_with_speakers_found": True,
                "transcript_preview_found": transcript_preview is not None,
                "biometric_backend": "speechbrain_ecapa_tdnn",
            },
            "counts": {
                "speakers_detected": len(summaries),
                "speakers_usable_for_identity": sum(
                    1 for item in voice_profiles if item.usable_for_identity
                ),
                "speakers_insufficient_voice_evidence": sum(
                    1 for item in voice_profiles if not item.usable_for_identity
                ),
                "selected_segments_total": sum(
                    len(segments) for segments in selected_segments_by_speaker.values()
                ),
                "embedding_segments_total": sum(
                    len(profile.segment_embeddings) for profile in voice_profiles
                ),
                "speakers_with_biometric_profile_ok": sum(
                    1 for profile in voice_profiles
                    if profile.embedding_primary
                    and profile.embedding_primary.get("status") == "ok"
                ),
                "text_self_identifications_detected": sum(
                    1 for item in text_evidence if item.self_identification_detected
                ),
                "text_participant_matches_detected": sum(
                    1 for item in text_evidence if len(item.participant_matches) > 0
                ),
            },
            "thresholds": {
                "min_speech_seconds_for_analysis": MIN_SPEECH_SECONDS_FOR_ANALYSIS,
                "min_num_words_for_analysis": MIN_NUM_WORDS_FOR_ANALYSIS,
                "min_selected_segment_seconds": MIN_SELECTED_SEGMENT_SECONDS,
                "min_selected_segment_words": MIN_SELECTED_SEGMENT_WORDS,
                "max_selected_segments_per_speaker": MAX_SELECTED_SEGMENTS_PER_SPEAKER,
            },
            "upstream": {
                "speaker_metrics": speaker_metrics,
                "alignment_counts": alignment_metadata.get("counts"),
            },
            "qa": qa_result.to_dict(),
        }

        write_speaker_voice_profiles_json(identity_dir, episode_id, voice_profiles)
        write_biometric_summary_json(identity_dir, episode_id, voice_profiles)
        write_identity_evidence_json(identity_dir, episode_id, selected_segments_by_speaker)
        write_text_support_json(identity_dir, episode_id, text_evidence)
        write_identity_candidates_json(identity_dir, episode_id, candidates_by_speaker)
        write_speaker_identity_json(identity_dir, episode_id, decisions)
        write_speaker_identity_metadata_json(identity_dir, metadata_payload)
        write_qa_summary_json(identity_dir, qa_result)

        write_identity_result_json(
            identity_dir,
            {
                "result": "success",
                "status_before": TARGET_STATUS,
                "status_after": SUCCESS_STATUS,
                "retries_before": int(episode.get("retries", 0)),
                "retries_after": 0,
                "message": "Speaker identity completed successfully.",
                "finished_at": utc_now_iso(),
            },
        )

        update_episode_status(episode_id, SUCCESS_STATUS)

        print("[speaker_identity] SUCCESS")
        return 0

    except Exception as exc:
        print(f"[speaker_identity] ERROR: {repr(exc)}")
        traceback.print_exc()

        retries_after = increment_retries(episode_id)
        if retries_after > MAX_RETRIES:
            mark_ruined(episode_id)

        try:
            write_identity_result_json(
                identity_dir,
                {
                    "result": "error",
                    "status_before": TARGET_STATUS,
                    "status_after": "ruined" if retries_after > MAX_RETRIES else TARGET_STATUS,
                    "retries_before": int(episode.get("retries", 0)),
                    "retries_after": retries_after,
                    "error": repr(exc),
                    "retry_consumed": True,
                    "finished_at": utc_now_iso(),
                },
            )
        except Exception as write_exc:
            print(f"[speaker_identity] Failed to write identity_result.json: {repr(write_exc)}")

        return 1

    finally:
        try:
            if temp_audio_path is not None:
                temp_audio_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
