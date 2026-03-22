from __future__ import annotations

import json
from pathlib import Path

from pyannote.audio import Pipeline


def diarize_audio(
    audio_path: Path,
    output_rttm: Path,
    output_segments_json: Path,
    hf_token: str,
) -> dict:
    if not hf_token.strip():
        raise ValueError("HF_TOKEN no configurado.")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )

    diarization = pipeline(str(audio_path))

    output_rttm.parent.mkdir(parents=True, exist_ok=True)
    with output_rttm.open("w", encoding="utf-8") as f:
        diarization.write_rttm(f)

    segments = []
    speakers = set()

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
        segments.append(
            {
                "speaker_id": speaker,
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "duration": round(float(turn.end - turn.start), 3),
            }
        )

    output_segments_json.parent.mkdir(parents=True, exist_ok=True)
    output_segments_json.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "num_speakers_detected": len(speakers),
        "num_segments": len(segments),
        "speaker_ids": sorted(speakers),
    }
