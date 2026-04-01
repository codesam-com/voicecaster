# src/voicecaster/diarization/metrics.py

from __future__ import annotations

from statistics import median

from .models import SpeakerSegment, TranscriptUtterance


def compute_speaker_metrics(
    speaker_segments: list[SpeakerSegment],
    utterances: list[TranscriptUtterance],
) -> dict:
    total_speech_seconds = round(sum(seg.duration for seg in speaker_segments), 3)

    by_speaker_segments: dict[str, list[SpeakerSegment]] = {}
    by_speaker_utterances: dict[str, list[TranscriptUtterance]] = {}

    for seg in speaker_segments:
        by_speaker_segments.setdefault(seg.speaker, []).append(seg)

    for utt in utterances:
        if utt.speaker:
            by_speaker_utterances.setdefault(utt.speaker, []).append(utt)

    speakers_payload = []
    for speaker in sorted(by_speaker_segments):
        segs = by_speaker_segments[speaker]
        utts = by_speaker_utterances.get(speaker, [])

        durations = [seg.duration for seg in segs]
        speech_seconds = round(sum(durations), 3)

        confidences = [
            utt.speaker_confidence
            for utt in utts
            if utt.speaker_confidence is not None
        ]

        low_confidence_segments = sum(
            1 for utt in utts if "low_confidence_assignment" in utt.flags
        )

        speakers_payload.append(
            {
                "speaker": speaker,
                "speech_seconds": speech_seconds,
                "speech_ratio": round(speech_seconds / total_speech_seconds, 4)
                if total_speech_seconds else 0.0,
                "num_turns": len(segs),
                "avg_turn_seconds": round(speech_seconds / len(segs), 3) if segs else 0.0,
                "median_turn_seconds": round(median(durations), 3) if durations else 0.0,
                "longest_turn_seconds": round(max(durations), 3) if durations else 0.0,
                "first_seen": min((seg.start for seg in segs), default=None),
                "last_seen": max((seg.end for seg in segs), default=None),
                "assignment_confidence_mean": round(sum(confidences) / len(confidences), 4)
                if confidences else None,
                "low_confidence_segments": low_confidence_segments,
            }
        )

    return {
        "num_speakers_detected": len(speakers_payload),
        "total_speech_seconds": total_speech_seconds,
        "speakers": speakers_payload,
    }
