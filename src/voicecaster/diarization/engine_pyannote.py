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

    This implementation loads the waveform in memory and passes:
        {"waveform": tensor, "sample_rate": int}
    to the pipeline, to avoid depending on pyannote's path-based audio decoding.

    It supports both:
    - modern community-1 outputs returning a DiarizeOutput object
    - older outputs returning an Annotation-like object
    """
    if not audio_path.exists():
        raise PyannoteEngineError(f"Audio file does not exist: {audio_path}")

    if not hf_token:
        raise PyannoteEngineError("HF_TOKEN is required to load pyannote pipelines.")

    try:
        import soundfile as sf
        import torch
        from pyannote.audio import Pipeline
    except Exception as exc:
        raise PyannoteEngineError(
            "Unable to import pyannote.audio and diarization dependencies."
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
        waveform_np, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=True,
        )
    except Exception as exc:
        raise PyannoteEngineError(
            f"Unable to read audio into memory: {audio_path}"
        ) from exc

    try:
        # soundfile returns [time, channels]
        # pyannote expects [channels, time]
        waveform = torch.from_numpy(waveform_np.T)
    except Exception as exc:
        raise PyannoteEngineError(
            "Failed to convert waveform to torch tensor."
        ) from exc

    try:
        output = pipeline(
            {
                "waveform": waveform,
                "sample_rate": int(sample_rate),
            }
        )
    except Exception as exc:
        raise PyannoteEngineError(
            f"pyannote diarization failed for in-memory audio: {audio_path}"
        ) from exc

    try:
        diarization_obj, diarization_source = _extract_diarization_object(output)
    except Exception as exc:
        raise PyannoteEngineError(
            f"Unable to extract diarization object from pipeline output of type {type(output).__name__}"
        ) from exc

    raw_segments = _serialize_diarization_output(diarization_obj)

    metadata: dict[str, Any] = {
        "engine": "pyannote",
        "pipeline": selected_pipeline_name,
        "device": device,
        "audio_path": str(audio_path),
        "audio_loaded_in_memory": True,
        "sample_rate": int(sample_rate),
        "num_channels": int(waveform.shape[0]),
        "output_type": type(output).__name__,
        "diarization_source": diarization_source,
        "num_raw_segments": len(raw_segments),
        "load_errors": load_errors,
    }

    return raw_segments, metadata


def _extract_diarization_object(output: Any) -> tuple[Any, str]:
    """
    Extract the diarization-bearing object from pyannote output.

    community-1 returns a DiarizeOutput with:
    - output.speaker_diarization
    - output.exclusive_speaker_diarization

    For this project, exclusive diarization is preferred because it simplifies
    reconciliation with ASR timestamps.
    """
    if hasattr(output, "exclusive_speaker_diarization"):
        diarization_obj = getattr(output, "exclusive_speaker_diarization")
        if diarization_obj is not None:
            return diarization_obj, "exclusive_speaker_diarization"

    if hasattr(output, "speaker_diarization"):
        diarization_obj = getattr(output, "speaker_diarization")
        if diarization_obj is not None:
            return diarization_obj, "speaker_diarization"

    # Backward compatibility: older pyannote may return the diarization object directly
    if hasattr(output, "itertracks"):
        return output, "direct_annotation"

    raise PyannoteEngineError(
        f"Unsupported pyannote output structure: {type(output).__name__}"
    )


def _serialize_diarization_output(diarization_obj: Any) -> list[RawSpeakerSegment]:
    """
    Serialize pyannote diarization output into internal RawSpeakerSegment objects.

    Supports:
    - Annotation-like objects with itertracks(yield_label=True)
    - iterables of (turn, speaker)
    """
    raw_segments: list[RawSpeakerSegment] = []

    if hasattr(diarization_obj, "itertracks"):
        for turn, _, speaker in diarization_obj.itertracks(yield_label=True):
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
        return raw_segments

    try:
        for item in diarization_obj:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(
                    f"Unexpected diarization tuple structure: {item!r}"
                )
            turn, speaker = item
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
        return raw_segments
    except TypeError as exc:
        raise PyannoteEngineError(
            "Diarization object is not iterable and does not expose itertracks()."
        ) from exc
