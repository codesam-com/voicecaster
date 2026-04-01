# src/voicecaster/diarization/debug_report.py

from __future__ import annotations

from typing import Any

from .qa import QAResult


def build_debug_report(
    *,
    engine_metadata: dict[str, Any],
    label_map: dict[str, str],
    normalization_warnings: list[str],
    assignment_stats: dict[str, Any],
    qa_result: QAResult,
) -> dict[str, Any]:
    return {
        "engine_metadata": engine_metadata,
        "label_map": label_map,
        "normalization_warnings": normalization_warnings,
        "assignment_stats": assignment_stats,
        "qa": qa_result.to_dict(),
    }
