from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


_MODEL_CACHE: dict[str, object] = {}


def _get_asr_pipeline(model_id: str = "openai/whisper-large-v3"):
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    torch_dtype = torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=4,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=-1,
    )
    _MODEL_CACHE[model_id] = asr
    return asr


def _format_timestamp(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000))
    hours = total_milliseconds // 3_600_000
    minutes = (total_milliseconds % 3_600_000) // 60_000
    secs = (total_milliseconds % 60_000) // 1000
    millis = total_milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def transcribe_audio_large_v3(audio_path: Path) -> dict:
    asr = _get_asr_pipeline("openai/whisper-large-v3")
    result = asr(str(audio_path))
    return result


def write_whisper_outputs(result: dict, work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)

    chunks = result.get("chunks", []) or []
    text = (result.get("text") or "").strip()

    whisper_raw_segments = []
    srt_lines: list[str] = []
    kept = 0

    for idx, chunk in enumerate(chunks):
        ts = chunk.get("timestamp")
        txt = str(chunk.get("text", "")).strip()
        if not txt or not ts or ts[0] is None or ts[1] is None:
            continue

        start = float(ts[0])
        end = float(ts[1])

        whisper_raw_segments.append(
            {
                "id": idx,
                "start": start,
                "end": end,
                "text": txt,
            }
        )

        kept += 1
        srt_lines.append(str(kept))
        srt_lines.append(f"{_format_timestamp(start)} --> {_format_timestamp(end)}")
        srt_lines.append(txt)
        srt_lines.append("")

    (work_dir / "whisper_raw_segments.json").write_text(
        json.dumps(whisper_raw_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (work_dir / "transcript_segments.json").write_text(
        json.dumps(whisper_raw_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (work_dir / "full_transcript.txt").write_text(text + "\n", encoding="utf-8")
    (work_dir / "full_transcript.srt").write_text("\n".join(srt_lines).strip() + "\n", encoding="utf-8")

    return {
        "language": result.get("language"),
        "model_name": "openai/whisper-large-v3",
        "num_segments_srt": kept,
        "num_segments_json": len(whisper_raw_segments),
        "text_characters": len(text),
        "text_preview": text[:500],
    }
