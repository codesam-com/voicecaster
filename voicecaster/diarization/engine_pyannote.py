# voicecaster/diarization/engine_pyannote.py

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    PYANNOTE_FALLBACK_PIPELINE,
    PYANNOTE_PRIMARY_PIPELINE,
)
from .models import RawSpeakerSegment


class PyannotePipelineLoadError(RuntimeError):
    pass


class PyannoteDiarizationError(RuntimeError):
    pass


def _validate_audio_path(audio_path: Path) -> None:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio path is not a file: {audio_path}")


def _load_pyannote_pipeline(
    hf_token: str,
) -> tuple[Any, str, list[dict[str, str]]]:
    """
    Intenta cargar el pipeline principal y, si falla, el fallback.
    Devuelve:
      - pipeline cargado
      - nombre del pipeline seleccionado
      - lista de errores acumulados de intentos previos
    """
    errors: list[dict[str, str]] = []

    from pyannote.audio import Pipeline  # import tardío

    load_attempts: list[tuple[str, dict[str, Any]]] = [
        (PYANNOTE_PRIMARY_PIPELINE, {"token": hf_token}),
        (PYANNOTE_FALLBACK_PIPELINE, {"use_auth_token": hf_token}),
    ]

    for pipeline_name, kwargs in load_attempts:
        try:
            pipeline = Pipeline.from_pretrained(pipeline_name, **kwargs)
            return pipeline, pipeline_name, errors
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "pipeline": pipeline_name,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
            )

    raise PyannotePipelineLoadError(
        f"Unable to load any pyannote pipeline. Attempts={errors}"
    )


def _maybe_move_pipeline_to_gpu(
    pipeline: Any,
    use_gpu_if_available: bool,
) -> str:
    """
    Mueve el pipeline a CUDA si se permite y está disponible.
    Devuelve el nombre del dispositivo efectivo.
    """
    if not use_gpu_if_available:
        return "cpu"

    try:
        import torch
    except Exception:  # noqa: BLE001
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    try:
        pipeline.to(torch.device("cuda"))
        return "cuda"
    except Exception:  # noqa: BLE001
        return "cpu"


def run_pyannote_diarization(
    audio_path: Path,
    hf_token: str,
    use_gpu_if_available: bool = False,
) -> tuple[list[RawSpeakerSegment], dict[str, Any]]:
    """
    Ejecuta diarización con pyannote y devuelve:
      1. segmentos crudos en formato neutral del proyecto
      2. metadata técnica del motor realmente usado
    """
    _validate_audio_path(audio_path)

    pipeline, selected_pipeline_name, load_errors = _load_pyannote_pipeline(hf_token)
    device = _maybe_move_pipeline_to_gpu(
        pipeline=pipeline,
        use_gpu_if_available=use_gpu_if_available,
    )

    try:
        diarization = pipeline(str(audio_path))
    except Exception as exc:  # noqa: BLE001
        raise PyannoteDiarizationError(
            f"pyannote diarization failed for audio={audio_path}: {exc}"
        ) from exc

    raw_segments: list[RawSpeakerSegment] = []

    try:
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            start = float(turn.start)
            end = float(turn.end)

            raw_segments.append(
                RawSpeakerSegment(
                    start=start,
                    end=end,
                    speaker_raw=str(speaker_label),
                    confidence=None,
                    engine="pyannote",
                    extra={},
                )
            )
    except Exception as exc:  # noqa: BLE001
        raise PyannoteDiarizationError(
            f"Failed while converting pyannote output to internal segments: {exc}"
        ) from exc

    raw_segments.sort(key=lambda s: (s.start, s.end, s.speaker_raw))

    metadata: dict[str, Any] = {
        "engine": "pyannote",
        "pipeline": selected_pipeline_name,
        "device": device,
        "audio_path": str(audio_path),
        "num_raw_segments": len(raw_segments),
        "load_errors": load_errors,
    }

    return raw_segments, metadata
