from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from pyannote.audio import Pipeline


def _write_rttm_like(segments: list[dict], output_rttm: Path, uri: str) -> None:
    output_rttm.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    for seg in segments:
        start = float(seg["start"])
        duration = float(seg["duration"])
        speaker = str(seg["speaker_id"])

        lines.append(
            f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        )

    output_rttm.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def diarize_audio(
    audio_path: Path,
    output_rttm: Path,
    output_segments_json: Path,
    hf_token: str,
) -> dict:
    if not hf_token.strip():
        raise ValueError("HF_TOKEN no configurado.")

    t0 = time.time()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )
    pipeline.to(torch.device("cpu"))

    t1 = time.time()

    diarization = pipeline(str(audio_path))

    t2 = time.time()

    segments: list[dict] = []
    speakers: set[str] = set()

    # Compatibilidad con el objeto devuelto por pyannote 4.x
    speaker_tracks = getattr(diarization, "speaker_diarization", diarization)

    for turn, _, speaker in speaker_tracks.itertracks(yield_label=True):
        speaker_id = str(speaker)
        start = round(float(turn.start), 3)
        end = round(float(turn.end), 3)
        duration = round(float(turn.end - turn.start), 3)

        speakers.add(speaker_id)
        segments.append(
            {
                "speaker_id": speaker_id,
                "start": start,
                "end": end,
                "duration": duration,
            }
        )

    output_segments_json.parent.mkdir(parents=True, exist_ok=True)
    output_segments_json.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_rttm_like(segments, output_rttm, uri=audio_path.stem)

    t3 = time.time()

    return {
        "num_speakers_detected": len(speakers),
        "num_segments": len(segments),
        "speaker_ids": sorted(speakers),
        "timing_seconds": {
            "load_pipeline": round(t1 - t0, 3),
            "run_pipeline": round(t2 - t1, 3),
            "serialize_outputs": round(t3 - t2, 3),
            "total": round(t3 - t0, 3),
        },
    }
