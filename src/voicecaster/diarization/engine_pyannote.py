# src/voicecaster/diarization/engine_pyannote.py

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    PYANNOTE_FALLBACK_PIPELINE,
    PYANNOTE_PRIMARY_PIPELINE,
)
from .models import RawSpeakerSegment


class PyannoteEngineError(RuntimeError):
    """Raised when pyannote diarization cannot be executed."""


def run_pyannote_diarization(
    audio_path: Path,
    hf_token: str,
    use_gpu_if_available: bool = False,
) -> tuple[list[RawSpeakerSegment], dict[str, Any]]:
    """
    Run speaker diarization with pyannote and return a neutral internal format.

    Returns:
        (raw_segments, metadata)
    """
    if not audio_path.exists():
        raise PyannoteEngineError(f"Audio file does not exist: {audio_path}")

    if not hf_token:
        raise PyannoteEngineError("HF_TOKEN is required to load pyannote pipelines.")

    try:
        import torch
        from pyannote.audio import Pipeline
    except Exception as exc:
        raise PyannoteEngineError(
            "Unable to import pyannote.audio and dependencies."
        ) from exc

    load_errors: list[dict[str, str]] = []
    pipeline = None
    selected_pipeline_name: str | None = None

    pipeline_attempts = [
        (PYANNOTE_PRIMARY_PIPELINE, {"token": hf_token}),
        (PYANNOTE_FALLBACK_PIPELINE, {"use_auth_token": hf_token}),
    ]

    for pipeline_name, kwargs in pipeline_attempts:
        try:
            pipeline = Pipeline.from_pretrained(pipeline_name, **kwargs)
            selected_pipeline_name = pipeline_name
            break
        except Exception as exc:
            load_errors.append(
                {
                    "pipeline": pipeline_name,
                    "error": repr(exc),
                }
            )

    if pipeline is None or selected_pipeline_name is None:
        raise PyannoteEngineError(
            f"Unable to load any pyannote pipeline. Errors: {load_errors}"
        )

    device = "cpu"
    if use_gpu_if_available and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        device = "cuda"

    try:
        diarization = pipeline(str(audio_path))
    except Exception as exc:
        raise PyannoteEngineError(
            f"pyannote diarization failed for audio: {audio_path}"
        ) from exc

    raw_segments: list[RawSpeakerSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        raw_segments.append(
            RawSpeakerSegment(
                start=float(turn.start),
                end=float(turn.end),
                speaker_raw=str(speaker),
                confidence=None,
                engine="pyannote",
                extra={},
            )
        )

    metadata: dict[str, Any] = {
        "engine": "pyannote",
        "pipeline": selected_pipeline_name,
        "device": device,
        "audio_path": str(audio_path),
        "num_raw_segments": len(raw_segments),
        "load_errors": load_errors,
    }

    return raw_segments, metadata
