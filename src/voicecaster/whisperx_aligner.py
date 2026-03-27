from __future__ import annotations

import json
from pathlib import Path

import whisperx


def align_with_whisperx(
    audio_path: Path,
    transcript_segments_path: Path,
    output_json: Path,
    language_code: str | None,
) -> dict:
    transcript_segments = json.loads(transcript_segments_path.read_text(encoding="utf-8"))

    whisperx_segments = [
        {
            "id": seg.get("id"),
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": str(seg["text"]).strip(),
        }
        for seg in transcript_segments
        if str(seg.get("text", "")).strip()
    ]

    # Cargar audio en memoria para evitar rutas de decodificación frágiles
    audio = whisperx.load_audio(str(audio_path))

    align_model, metadata = whisperx.load_align_model(
        language_code=language_code or "es",
        device="cpu",
    )

    aligned = whisperx.align(
        whisperx_segments,
        align_model,
        metadata,
        audio,
        "cpu",
        return_char_alignments=False,
    )

    raw_segments = aligned.get("segments", []) or []

    normalized_segments = []
    for idx, seg in enumerate(raw_segments):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        start = seg.get("start")
        end = seg.get("end")

        if start is None or end is None:
            continue

        normalized_segments.append(
            {
                "id": seg.get("id", idx),
                "start": float(start),
                "end": float(end),
                "text": text,
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(normalized_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "num_aligned_segments": len(normalized_segments),
        "language_code": language_code or "es",
    }
