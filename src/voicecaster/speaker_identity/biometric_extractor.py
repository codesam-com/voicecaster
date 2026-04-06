from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

from .segment_selector import SelectedSegment


@dataclass(slots=True)
class SegmentEmbedding:
    speaker: str
    start: float
    end: float
    duration: float
    num_words: int
    primary_model: str | None
    secondary_model: str | None
    primary_vector: list[float] | None
    secondary_vector: list[float] | None
    embedding_status: str
    quality_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AggregatedBiometricProfile:
    speaker: str
    primary_model: str | None
    secondary_model: str | None
    primary_embedding: list[float] | None
    secondary_embedding: list[float] | None
    aggregation_method: str
    num_segments_used: int
    total_duration_used: float
    intra_speaker_dispersion_mean: float | None
    intra_speaker_dispersion_p95: float | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CLASSIFIER: EncoderClassifier | None = None


def _get_classifier() -> EncoderClassifier:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
    return _CLASSIFIER


def _estimate_quality_weight(segment: SelectedSegment) -> float:
    duration_component = min(1.0, segment.duration / 8.0)
    words_component = min(1.0, segment.num_words / 25.0)
    weight = 0.6 * duration_component + 0.4 * words_component
    return round(max(0.1, min(1.0, weight)), 4)


def _load_audio_mono_16k(audio_path: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(audio_path))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    return waveform


def _slice_segment(waveform: torch.Tensor, start_sec: float, end_sec: float) -> torch.Tensor:
    start_sample = max(0, int(round(start_sec * 16000)))
    end_sample = max(start_sample + 1, int(round(end_sec * 16000)))
    sliced = waveform[:, start_sample:end_sample]

    if sliced.numel() == 0:
        raise ValueError("Empty audio slice after segment extraction.")

    return sliced


def _encode_segment(segment_waveform: torch.Tensor) -> list[float]:
    classifier = _get_classifier()
    with torch.no_grad():
        embedding = classifier.encode_batch(segment_waveform)
    return embedding.squeeze().detach().cpu().tolist()


def extract_segment_embeddings(
    speaker: str,
    selected_segments: list[SelectedSegment],
    audio_path: Path,
) -> list[SegmentEmbedding]:
    if not selected_segments:
        return []

    waveform = _load_audio_mono_16k(audio_path)
    result: list[SegmentEmbedding] = []

    for segment in selected_segments:
        try:
            segment_waveform = _slice_segment(waveform, segment.start, segment.end)
            primary_vector = _encode_segment(segment_waveform)
            status = "ok"
        except Exception:
            primary_vector = None
            status = "segment_embedding_failed"

        result.append(
            SegmentEmbedding(
                speaker=speaker,
                start=segment.start,
                end=segment.end,
                duration=segment.duration,
                num_words=segment.num_words,
                primary_model="ecapa_tdnn",
                secondary_model=None,
                primary_vector=primary_vector,
                secondary_vector=None,
                embedding_status=status,
                quality_weight=_estimate_quality_weight(segment),
            )
        )

    return result


def _weighted_centroid(vectors: list[list[float]], weights: list[float]) -> list[float]:
    if not vectors:
        raise ValueError("No vectors for centroid.")
    if len(vectors) != len(weights):
        raise ValueError("Vectors and weights length mismatch.")

    dim = len(vectors[0])
    accum = [0.0] * dim
    total_weight = 0.0

    for vec, weight in zip(vectors, weights):
        if len(vec) != dim:
            raise ValueError("Inconsistent embedding dimensions.")
        for i, value in enumerate(vec):
            accum[i] += value * weight
        total_weight += weight

    if total_weight <= 0:
        raise ValueError("Total weight must be positive.")

    return [value / total_weight for value in accum]


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_dispersion(
    vectors: list[list[float]],
    centroid: list[float],
) -> tuple[float | None, float | None]:
    if len(vectors) < 2:
        return None, None

    distances = [1.0 - _cosine_similarity(vec, centroid) for vec in vectors]
    distances.sort()

    mean_value = sum(distances) / len(distances)
    p95_index = min(len(distances) - 1, int(round(0.95 * (len(distances) - 1))))

    return round(mean_value, 6), round(distances[p95_index], 6)


def aggregate_biometric_profile(
    speaker: str,
    segment_embeddings: list[SegmentEmbedding],
) -> AggregatedBiometricProfile:
    usable = [
        item for item in segment_embeddings
        if item.primary_vector is not None and item.embedding_status == "ok"
    ]

    total_duration_used = round(sum(item.duration for item in usable), 3)

    if not usable:
        return AggregatedBiometricProfile(
            speaker=speaker,
            primary_model="ecapa_tdnn",
            secondary_model=None,
            primary_embedding=None,
            secondary_embedding=None,
            aggregation_method="quality_weighted_centroid",
            num_segments_used=0,
            total_duration_used=0.0,
            intra_speaker_dispersion_mean=None,
            intra_speaker_dispersion_p95=None,
            status="no_usable_segment_embeddings",
        )

    vectors = [item.primary_vector for item in usable if item.primary_vector is not None]
    weights = [item.quality_weight for item in usable]

    centroid = _weighted_centroid(vectors, weights)
    disp_mean, disp_p95 = _compute_dispersion(vectors, centroid)

    return AggregatedBiometricProfile(
        speaker=speaker,
        primary_model="ecapa_tdnn",
        secondary_model=None,
        primary_embedding=centroid,
        secondary_embedding=None,
        aggregation_method="quality_weighted_centroid",
        num_segments_used=len(usable),
        total_duration_used=total_duration_used,
        intra_speaker_dispersion_mean=disp_mean,
        intra_speaker_dispersion_p95=disp_p95,
        status="ok",
    )
