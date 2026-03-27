from __future__ import annotations

import json
from pathlib import Path

from .transcriber import transcribe_audio_large_v3
from .vad import cut_audio_with_ffmpeg


def redecode_doubtful_segments(
    source_audio_path: Path,
    doubtful_segments_path: Path,
    temp_dir: Path,
    output_json: Path,
) -> dict:
    doubtful_segments = json.loads(doubtful_segments_path.read_text(encoding="utf-8"))
    redecoded = []

    for seg in doubtful_segments:
        start = float(seg["start"])
        end = float(seg["end"])
        seg_id = seg.get("id")

        cut_path = temp_dir / f"doubtful_{seg_id}.wav"
        cut_audio_with_ffmpeg(source_audio_path, start, end, cut_path)

        result = transcribe_audio_large_v3(cut_path)
        text = (result.get("text") or "").strip()

        redecoded.append(
            {
                "id": seg_id,
                "start": start,
                "end": end,
                "speaker_id": seg.get("speaker_id"),
                "original_text": seg.get("text"),
                "redecoded_text": text,
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(redecoded, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "num_redecoded_segments": len(redecoded),
    }
