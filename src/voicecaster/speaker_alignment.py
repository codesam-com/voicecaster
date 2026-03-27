from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _format_timestamp(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000))
    hours = total_milliseconds // 3_600_000
    minutes = (total_milliseconds % 3_600_000) // 60_000
    secs = (total_milliseconds % 60_000) // 1000
    millis = total_milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers_to_transcript_segments(
    transcript_segments_path: Path,
    speaker_segments_path: Path,
    output_aligned_json: Path,
    output_doubtful_json: Path,
    min_overlap_seconds: float = 0.15,
) -> dict:
    transcript_segments = _load_json(transcript_segments_path)
    speaker_segments = _load_json(speaker_segments_path)

    aligned_segments = []
    doubtful_segments = []
    speaker_time = defaultdict(float)
    speaker_segment_count = defaultdict(int)

    for seg in transcript_segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        seg_text = str(seg["text"]).strip()

        overlaps_by_speaker = defaultdict(float)
        for spk in speaker_segments:
            ov = _overlap(seg_start, seg_end, float(spk["start"]), float(spk["end"]))
            if ov > 0:
                overlaps_by_speaker[str(spk["speaker_id"])] += ov

        sorted_overlaps = sorted(overlaps_by_speaker.items(), key=lambda item: item[1], reverse=True)
        best_speaker = "speaker_unknown"
        best_overlap = 0.0
        second_best_overlap = 0.0

        if sorted_overlaps:
            best_speaker, best_overlap = sorted_overlaps[0]
            if len(sorted_overlaps) > 1:
                _, second_best_overlap = sorted_overlaps[1]

        if best_overlap < min_overlap_seconds:
            best_speaker = "speaker_unknown"

        is_overlap_flag = second_best_overlap > 0.0
        is_doubtful = (
            best_speaker == "speaker_unknown"
            or is_overlap_flag
            or (best_overlap > 0 and second_best_overlap / max(best_overlap, 1e-6) > 0.6)
            or (seg_end - seg_start) < 1.0
        )

        aligned_segment = {
            "id": seg.get("id"),
            "start": seg_start,
            "end": seg_end,
            "duration": round(seg_end - seg_start, 3),
            "speaker_id": best_speaker,
            "overlap_seconds": round(best_overlap, 3),
            "second_best_overlap_seconds": round(second_best_overlap, 3),
            "is_doubtful": is_doubtful,
            "text": seg_text,
        }
        aligned_segments.append(aligned_segment)

        if is_doubtful:
            doubtful_segments.append(aligned_segment)

        speaker_time[best_speaker] += seg_end - seg_start
        speaker_segment_count[best_speaker] += 1

    output_aligned_json.parent.mkdir(parents=True, exist_ok=True)
    output_aligned_json.write_text(
        json.dumps(aligned_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_doubtful_json.parent.mkdir(parents=True, exist_ok=True)
    output_doubtful_json.write_text(
        json.dumps(doubtful_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_aligned_time = sum(item["duration"] for item in aligned_segments)
    metrics = []
    for speaker_id in sorted(speaker_time.keys()):
        speaking_time = round(speaker_time[speaker_id], 3)
        participation_pct = round((speaking_time / total_aligned_time) * 100, 3) if total_aligned_time > 0 else 0.0
        metrics.append(
            {
                "speaker_id": speaker_id,
                "speaking_time_seconds": speaking_time,
                "participation_pct": participation_pct,
                "num_assigned_segments": speaker_segment_count[speaker_id],
            }
        )

    return {
        "num_aligned_segments": len(aligned_segments),
        "num_distinct_speakers": len({m["speaker_id"] for m in metrics}),
        "total_aligned_time_seconds": round(total_aligned_time, 3),
        "num_doubtful_segments": len(doubtful_segments),
        "speaker_metrics": metrics,
    }


def write_full_transcript_with_speakers(aligned_segments_json_path: Path, output_srt_path: Path) -> int:
    aligned_segments = _load_json(aligned_segments_json_path)
    lines: list[str] = []
    kept = 0

    for seg in aligned_segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        kept += 1
        start = _format_timestamp(float(seg["start"]))
        end = _format_timestamp(float(seg["end"]))
        speaker_id = str(seg["speaker_id"])
        prefix = "[OVERLAP] " if seg.get("is_doubtful") else ""
        lines.append(str(kept))
        lines.append(f"{start} --> {end}")
        lines.append(f"[{speaker_id}] {prefix}{text}")
        lines.append("")

    output_srt_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return kept


def write_per_speaker_srts(aligned_segments_json_path: Path, output_dir: Path) -> dict:
    aligned_segments = _load_json(aligned_segments_json_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for seg in aligned_segments:
        grouped[str(seg["speaker_id"])].append(seg)

    result = {}
    for speaker_id, items in grouped.items():
        if speaker_id == "speaker_unknown":
            continue

        lines: list[str] = []
        kept = 0

        for seg in items:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue

            kept += 1
            start = _format_timestamp(float(seg["start"]))
            end = _format_timestamp(float(seg["end"]))
            prefix = "[OVERLAP] " if seg.get("is_doubtful") else ""

            lines.append(str(kept))
            lines.append(f"{start} --> {end}")
            lines.append(f"{prefix}{text}")
            lines.append("")

        output_path = output_dir / f"{speaker_id}.srt"
        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        result[speaker_id] = {"path": str(output_path), "num_segments": kept}

    return result
