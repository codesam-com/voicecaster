from __future__ import annotations

import json
import os
import shutil
import subprocess
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

INPUTS_PATH = Path("inputs/inputs.json")
WORK_ROOT = Path("work")
TARGET_STATUS = "transcript"
SUCCESS_STATUS = "diarization"
RUINED_STATUS = "ruined"
WORKFLOW_NAME = "transcription"
MAX_RETRIES = 10

DEFAULT_MODEL = os.getenv("VOICECASTER_ASR_MODEL", "turbo")
DEFAULT_ENGINE = os.getenv("VOICECASTER_ASR_ENGINE", "faster-whisper")
DEFAULT_LANGUAGE = os.getenv("VOICECASTER_ASR_LANGUAGE") or None
DEFAULT_SAMPLE_RATE = int(os.getenv("VOICECASTER_ASR_SAMPLE_RATE", "16000"))

DEFAULT_BEAM_SIZE = int(os.getenv("VOICECASTER_ASR_BEAM_SIZE", "5"))
DEFAULT_BEST_OF = int(os.getenv("VOICECASTER_ASR_BEST_OF", "5"))
DEFAULT_TEMPERATURE = float(os.getenv("VOICECASTER_ASR_TEMPERATURE", "0.0"))
DEFAULT_LOGPROB_THRESHOLD = float(os.getenv("VOICECASTER_ASR_LOGPROB_THRESHOLD", "-0.8"))
DEFAULT_NO_SPEECH_THRESHOLD = float(os.getenv("VOICECASTER_ASR_NO_SPEECH_THRESHOLD", "0.5"))
DEFAULT_COMPRESSION_RATIO_THRESHOLD = float(
    os.getenv("VOICECASTER_ASR_COMPRESSION_RATIO_THRESHOLD", "2.0")
)

DEFAULT_VAD_FILTER = os.getenv("VOICECASTER_ASR_VAD_FILTER", "true").lower() == "true"
DEFAULT_VAD_MIN_SILENCE_MS = int(
    os.getenv("VOICECASTER_ASR_VAD_MIN_SILENCE_MS", "500")
)

HTTP_TIMEOUT_SECONDS = int(os.getenv("VOICECASTER_HTTP_TIMEOUT_SECONDS", "120"))
USER_AGENT = os.getenv(
    "VOICECASTER_HTTP_USER_AGENT",
    "voicecaster-transcription/1.0 (+https://github.com/codesam-com/voicecaster)",
)


class NetworkError(RuntimeError):
    pass


class ContentError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class EpisodeSelection:
    index: int
    episode: dict[str, Any]

    @property
    def episode_id(self) -> str:
        value = self.episode.get("id")
        if not isinstance(value, str) or not value.strip():
            raise ContentError("Episode id is missing or invalid.")
        return value


class EpisodeWorkspace:
    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self.root = WORK_ROOT / episode_id
        self.intake_dir = self.root / "00_intake"
        self.logs_dir = self.root / "01_logs"
        self.transcription_dir = self.root / "02_transcription"
        self.temp_dir = self.root / "99_temp"
        self.status_path = self.root / "status.json"
        self.readme_path = self.root / "README.md"
        self.events_path = self.logs_dir / "events.jsonl"
        self.report_path = self.logs_dir / "report.json"
        self.error_path = self.logs_dir / "error.json"

    def ensure_base_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.intake_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def reset_phase_dirs(self) -> None:
        shutil.rmtree(self.transcription_dir, ignore_errors=True)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.transcription_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)


class WorkflowContext:
    def __init__(self, selection: EpisodeSelection) -> None:
        self.selection = selection
        self.workspace = EpisodeWorkspace(selection.episode_id)
        self.started_at = utc_now_iso()
        self.source_info: dict[str, Any] = {}
        self.audio_info: dict[str, Any] = {}
        self.preprocessing_info: dict[str, Any] = {}
        self.transcription_info: dict[str, Any] = {}
        self.notes: list[str] = []
        self.result: str = "running"
        self.current_step: str = "starting"

    def log_event(self, event: str, **payload: Any) -> None:
        self.workspace.logs_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": utc_now_iso(),
            "event": event,
            "episode_id": self.selection.episode_id,
            **payload,
        }
        with self.workspace.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_status(
        self,
        *,
        status: str,
        current_step: str,
        result: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        self.current_step = current_step
        payload = {
            "episode_id": self.selection.episode_id,
            "workflow": WORKFLOW_NAME,
            "status": status,
            "current_step": current_step,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "result": result,
        }
        self.workspace.status_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def write_report(self, result: str, finished_at: str | None = None) -> None:
        payload = {
            "episode_id": self.selection.episode_id,
            "workflow": WORKFLOW_NAME,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "result": result,
            "notes": self.notes,
            "source": self.source_info,
            "audio": self.audio_info,
            "preprocessing": self.preprocessing_info,
            "transcription": self.transcription_info,
        }
        self.workspace.report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def write_error(self, exc: BaseException, *, retry_consumed: bool) -> None:
        payload = {
            "workflow": WORKFLOW_NAME,
            "episode_id": self.selection.episode_id,
            "error_type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at": utc_now_iso(),
            "retry_consumed": retry_consumed,
        }
        self.workspace.error_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def load_inputs() -> list[dict[str, Any]]:
    if not INPUTS_PATH.exists():
        raise ContentError(f"Missing inputs file: {INPUTS_PATH}")
    data = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ContentError("inputs/inputs.json must contain a JSON array.")
    return data


def save_inputs(data: list[dict[str, Any]]) -> None:
    INPUTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_episode(data: list[dict[str, Any]]) -> EpisodeSelection | None:
    for index, item in enumerate(data):
        if isinstance(item, dict) and item.get("status") == TARGET_STATUS:
            return EpisodeSelection(index=index, episode=item)
    return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ContentError(f"Missing required file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ContentError(f"JSON object expected in: {path}")
    return data


def ensure_readme(workspace: EpisodeWorkspace) -> None:
    if workspace.readme_path.exists():
        return
    content = """# Work directory

This folder stores the auditable working state for one episode.

- `00_intake/` contains intake outputs.
- `01_logs/` contains structured logs and reports.
- `02_transcription/` contains transcription outputs.
- `99_temp/` contains only temporary files during the active run.
"""
    workspace.readme_path.write_text(content, encoding="utf-8")


def load_intake_context(
    ctx: WorkflowContext,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    request = load_json(ctx.workspace.intake_dir / "request.json")
    normalized_source = load_json(ctx.workspace.intake_dir / "normalized_source.json")
    source_metadata = load_json(ctx.workspace.intake_dir / "source_metadata.json")
    return request, normalized_source, source_metadata


def download_file(url: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type")
            bytes_written = 0
            with destination.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_written += len(chunk)
    except urllib.error.HTTPError as exc:
        if 500 <= exc.code < 600:
            raise NetworkError(f"HTTP {exc.code} while downloading audio.") from exc
        raise ContentError(f"HTTP {exc.code} while downloading audio.") from exc
    except urllib.error.URLError as exc:
        raise NetworkError(f"Network error while downloading audio: {exc.reason}") from exc
    except TimeoutError as exc:
        raise NetworkError("Timeout while downloading audio.") from exc

    return {
        "final_url": final_url,
        "content_type": content_type,
        "bytes_written": bytes_written,
    }


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, capture_output=True)


def ffprobe_audio(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name,bit_rate,size",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise ContentError(f"ffprobe failed: {result.stderr.strip() or result.stdout.strip()}")

    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise ContentError("Downloaded resource does not contain an audio stream.")

    audio_stream = audio_streams[0]
    format_info = payload.get("format") or {}

    def to_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def to_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "ffprobe_ok": True,
        "duration_seconds": to_float(format_info.get("duration")),
        "container": format_info.get("format_name"),
        "audio_codec": audio_stream.get("codec_name"),
        "sample_rate": to_int(audio_stream.get("sample_rate")),
        "channels": to_int(audio_stream.get("channels")),
        "bit_rate": to_int(format_info.get("bit_rate")),
        "size_bytes": to_int(format_info.get("size")),
        "streams": streams,
    }


def preprocess_audio(source_audio: Path, normalized_audio: Path) -> dict[str, Any]:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_audio),
        "-ac",
        "1",
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(normalized_audio),
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise ContentError(f"ffmpeg preprocessing failed: {result.stderr.strip() or result.stdout.strip()}")

    normalized_probe = ffprobe_audio(normalized_audio)
    return {
        "source_audio_valid": True,
        "source_duration_seconds": normalized_probe.get("duration_seconds"),
        "normalized_audio_created": normalized_audio.exists(),
        "target_sample_rate": DEFAULT_SAMPLE_RATE,
        "target_channels": 1,
        "target_format": "wav/pcm_s16le",
        "ffprobe_ok_after_preprocessing": normalized_probe.get("ffprobe_ok", False),
    }


def build_initial_prompt(episode: dict[str, Any]) -> str:
    parts: list[str] = [
        (
            "Este es un podcast en español. "
            "Transcribe con precisión literal, respetando nombres propios, términos técnicos y puntuación. "
            "No inventes contenido ni reformules."
        )
    ]

    podcast_title = episode.get("podcast_title")
    episode_title = episode.get("episode_title")
    topics = episode.get("topics")
    participants = episode.get("participants")

    if isinstance(podcast_title, str) and podcast_title.strip():
        parts.append(f"Podcast: {podcast_title.strip()}.")
    if isinstance(episode_title, str) and episode_title.strip():
        parts.append(f"Episodio: {episode_title.strip()}.")
    if isinstance(participants, list) and participants:
        joined = ", ".join(str(item).strip() for item in participants if str(item).strip())
        if joined:
            parts.append(f"Participantes declarados: {joined}.")
    if isinstance(topics, str) and topics.strip():
        parts.append(f"Temas y vocabulario relevante: {topics.strip()}.")

    return " ".join(parts).strip()


def transcribe_with_faster_whisper(
    audio_path: Path,
    episode: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install it or change VOICECASTER_ASR_ENGINE."
        ) from exc

    model = WhisperModel(DEFAULT_MODEL)
    prompt = build_initial_prompt(episode) or None

    segments, info = model.transcribe(
        str(audio_path),
        language=DEFAULT_LANGUAGE,
        word_timestamps=True,
        condition_on_previous_text=True,
        initial_prompt=prompt,
        beam_size=DEFAULT_BEAM_SIZE,
        best_of=DEFAULT_BEST_OF,
        temperature=DEFAULT_TEMPERATURE,
        compression_ratio_threshold=DEFAULT_COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=DEFAULT_LOGPROB_THRESHOLD,
        no_speech_threshold=DEFAULT_NO_SPEECH_THRESHOLD,
        vad_filter=DEFAULT_VAD_FILTER,
        vad_parameters={"min_silence_duration_ms": DEFAULT_VAD_MIN_SILENCE_MS},
    )

    serialized_segments: list[dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        item: dict[str, Any] = {
            "id": idx,
            "start": float(segment.start),
            "end": float(segment.end),
            "text": (segment.text or "").strip(),
        }
        avg_logprob = getattr(segment, "avg_logprob", None)
        compression_ratio = getattr(segment, "compression_ratio", None)
        no_speech_prob = getattr(segment, "no_speech_prob", None)
        words = getattr(segment, "words", None)
        if avg_logprob is not None:
            item["avg_logprob"] = avg_logprob
        if compression_ratio is not None:
            item["compression_ratio"] = compression_ratio
        if no_speech_prob is not None:
            item["no_speech_prob"] = no_speech_prob
        if words:
            item["words"] = [
                {
                    "word": getattr(word, "word", ""),
                    "start": getattr(word, "start", None),
                    "end": getattr(word, "end", None),
                    "probability": getattr(word, "probability", None),
                }
                for word in words
            ]
        serialized_segments.append(item)

    info_payload = {
        "engine": "faster-whisper",
        "model": DEFAULT_MODEL,
        "language": getattr(info, "language", None),
        "duration_seconds": getattr(info, "duration", None),
        "language_probability": getattr(info, "language_probability", None),
    }
    return serialized_segments, info_payload


def transcribe_with_openai_whisper(
    audio_path: Path,
    episode: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper is not installed. Install it or change VOICECASTER_ASR_ENGINE."
        ) from exc

    model = whisper.load_model(DEFAULT_MODEL)
    result = model.transcribe(
        str(audio_path),
        language=DEFAULT_LANGUAGE,
        word_timestamps=True,
        condition_on_previous_text=True,
        initial_prompt=build_initial_prompt(episode) or None,
        beam_size=DEFAULT_BEAM_SIZE,
        best_of=DEFAULT_BEST_OF,
        temperature=DEFAULT_TEMPERATURE,
        compression_ratio_threshold=DEFAULT_COMPRESSION_RATIO_THRESHOLD,
        logprob_threshold=DEFAULT_LOGPROB_THRESHOLD,
        no_speech_threshold=DEFAULT_NO_SPEECH_THRESHOLD,
        verbose=False,
    )

    segments_in = result.get("segments") or []
    segments_out: list[dict[str, Any]] = []
    for segment in segments_in:
        item = {
            "id": segment.get("id"),
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": (segment.get("text") or "").strip(),
        }
        for key in ("avg_logprob", "compression_ratio", "no_speech_prob", "tokens"):
            if key in segment:
                item[key] = segment[key]
        if "words" in segment:
            item["words"] = segment["words"]
        segments_out.append(item)

    info_payload = {
        "engine": "openai-whisper",
        "model": DEFAULT_MODEL,
        "language": result.get("language"),
        "duration_seconds": None,
    }
    return segments_out, info_payload


def transcribe_audio(audio_path: Path, episode: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    engine = (DEFAULT_ENGINE or "").strip().lower()
    if engine == "faster-whisper":
        return transcribe_with_faster_whisper(audio_path, episode)
    if engine in {"openai-whisper", "whisper"}:
        return transcribe_with_openai_whisper(audio_path, episode)
    raise RuntimeError(f"Unsupported ASR engine: {DEFAULT_ENGINE}")


def format_srt_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_transcript_outputs(
    ctx: WorkflowContext,
    segments: list[dict[str, Any]],
    info: dict[str, Any],
) -> None:
    txt_path = ctx.workspace.transcription_dir / "full_transcript.txt"
    srt_path = ctx.workspace.transcription_dir / "full_transcript.srt"
    segments_path = ctx.workspace.transcription_dir / "transcript_segments.json"

    text_chunks = [segment["text"].strip() for segment in segments if segment.get("text")]
    txt_path.write_text("\n".join(text_chunks).strip() + "\n", encoding="utf-8")

    srt_lines: list[str] = []
    for idx, segment in enumerate(segments, start=1):
        start = float(segment["start"])
        end = float(segment["end"])
        text = str(segment.get("text") or "").strip()
        srt_lines.extend(
            [
                str(idx),
                f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}",
                text,
                "",
            ]
        )
    srt_path.write_text("\n".join(srt_lines).rstrip() + "\n", encoding="utf-8")

    write_json(segments_path, segments)

    preview_payload = {
        "episode_id": ctx.selection.episode_id,
        "language": info.get("language"),
        "num_segments": len(segments),
        "first_segments": segments[:3],
        "last_segments": segments[-3:] if len(segments) > 3 else segments,
    }
    write_json(ctx.workspace.transcription_dir / "transcript_preview.json", preview_payload)

    duration_seconds = info.get("duration_seconds") or ctx.audio_info.get("duration_seconds")
    ctx.transcription_info = {
        **info,
        "duration_seconds": duration_seconds,
        "num_segments": len(segments),
        "num_characters": sum(len(chunk) for chunk in text_chunks),
        "srt_generated": True,
        "txt_generated": True,
        "segments_generated": True,
        "beam_size": DEFAULT_BEAM_SIZE,
        "best_of": DEFAULT_BEST_OF,
        "temperature": DEFAULT_TEMPERATURE,
        "vad_filter": DEFAULT_VAD_FILTER,
        "vad_min_silence_duration_ms": DEFAULT_VAD_MIN_SILENCE_MS,
        "compression_ratio_threshold": DEFAULT_COMPRESSION_RATIO_THRESHOLD,
        "logprob_threshold": DEFAULT_LOGPROB_THRESHOLD,
        "no_speech_threshold": DEFAULT_NO_SPEECH_THRESHOLD,
    }
    write_json(ctx.workspace.transcription_dir / "transcription_metadata.json", ctx.transcription_info)


def validate_outputs(ctx: WorkflowContext) -> None:
    txt_path = ctx.workspace.transcription_dir / "full_transcript.txt"
    srt_path = ctx.workspace.transcription_dir / "full_transcript.srt"
    segments_path = ctx.workspace.transcription_dir / "transcript_segments.json"

    if not txt_path.exists() or not srt_path.exists() or not segments_path.exists():
        raise ContentError("Missing one or more transcription output files.")

    txt_content = txt_path.read_text(encoding="utf-8").strip()
    if not txt_content:
        raise ContentError("full_transcript.txt is empty.")

    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    if not isinstance(segments, list) or not segments:
        raise ContentError("transcript_segments.json does not contain valid segments.")

    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        text = str(segment.get("text") or "").strip()
        if start is None or end is None:
            raise ContentError("A transcript segment is missing start/end timestamps.")
        if float(start) >= float(end):
            raise ContentError("A transcript segment has invalid timestamps.")
        if not text:
            raise ContentError("A transcript segment has empty text.")

    srt_content = srt_path.read_text(encoding="utf-8")
    if "-->" not in srt_content:
        raise ContentError("full_transcript.srt is not parseable.")


def update_episode_status(
    data: list[dict[str, Any]],
    selection: EpisodeSelection,
    *,
    status: str,
    retries: int,
) -> None:
    data[selection.index]["status"] = status
    data[selection.index]["retries"] = retries
    save_inputs(data)


def cleanup_temp_dir(workspace: EpisodeWorkspace) -> None:
    shutil.rmtree(workspace.temp_dir, ignore_errors=True)


def is_network_error(exc: BaseException) -> bool:
    return isinstance(exc, NetworkError)


def main() -> int:
    data = load_inputs()
    selection = select_episode(data)
    if selection is None:
        print("No episode pending for transcription.")
        return 0

    ctx = WorkflowContext(selection)
    ctx.workspace.ensure_base_dirs()
    ensure_readme(ctx.workspace)
    ctx.workspace.reset_phase_dirs()
    ctx.log_event("transcription_started", selected_index=selection.index)
    ctx.write_status(status=TARGET_STATUS, current_step="load_context", result="running")

    transcription_request = {
        "episode_id": selection.episode_id,
        "podcast_title": selection.episode.get("podcast_title"),
        "episode_title": selection.episode.get("episode_title"),
        "url": selection.episode.get("url"),
        "status_before": selection.episode.get("status"),
        "retries_before": selection.episode.get("retries", 0),
        "requested_at": ctx.started_at,
    }
    write_json(ctx.workspace.transcription_dir / "transcription_request.json", transcription_request)

    try:
        request_json, normalized_source, source_metadata = load_intake_context(ctx)
        _ = source_metadata  # explicit: context presence matters, even if not yet reused further

        ctx.source_info = {
            "url_original": normalized_source.get("url_original") or request_json.get("url"),
            "url_normalized": normalized_source.get("url_normalized") or request_json.get("url"),
            "source_type": normalized_source.get("source_type"),
            "drive_file_id": normalized_source.get("drive_file_id"),
        }
        ctx.log_event("transcription_context_loaded")

        source_url = ctx.source_info.get("url_normalized") or ctx.source_info.get("url_original")
        if not isinstance(source_url, str) or not source_url.strip():
            raise ContentError("No usable source URL found for transcription.")

        source_audio_path = ctx.workspace.temp_dir / "source_audio"
        normalized_audio_path = ctx.workspace.temp_dir / "audio_asr.wav"

        ctx.write_status(status=TARGET_STATUS, current_step="download_audio", result="running")
        download_info = download_file(source_url, source_audio_path)
        ctx.log_event("audio_redownloaded", **download_info)

        ctx.write_status(status=TARGET_STATUS, current_step="validate_audio", result="running")
        ctx.audio_info = ffprobe_audio(source_audio_path)
        ctx.log_event(
            "audio_validated_for_transcription",
            duration_seconds=ctx.audio_info.get("duration_seconds"),
            audio_codec=ctx.audio_info.get("audio_codec"),
            sample_rate=ctx.audio_info.get("sample_rate"),
            channels=ctx.audio_info.get("channels"),
        )

        ctx.write_status(status=TARGET_STATUS, current_step="preprocess_audio", result="running")
        ctx.preprocessing_info = preprocess_audio(source_audio_path, normalized_audio_path)
        write_json(
            ctx.workspace.transcription_dir / "preprocessing_metadata.json",
            ctx.preprocessing_info,
        )
        write_json(
            ctx.workspace.transcription_dir / "ffprobe_normalized.json",
            ffprobe_audio(normalized_audio_path),
        )
        ctx.log_event("audio_preprocessed")

        ctx.write_status(status=TARGET_STATUS, current_step="run_transcription", result="running")
        segments, info = transcribe_audio(normalized_audio_path, selection.episode)
        write_transcript_outputs(ctx, segments, info)
        ctx.log_event(
            "transcription_generated",
            num_segments=len(segments),
            language=info.get("language"),
        )

        ctx.write_status(status=TARGET_STATUS, current_step="validate_outputs", result="running")
        validate_outputs(ctx)
        ctx.log_event("transcription_outputs_validated")

        update_episode_status(data, selection, status=SUCCESS_STATUS, retries=0)

        finished_at = utc_now_iso()
        ctx.notes.append("Audio redescargado, preprocesado, transcrito y limpiado.")
        ctx.result = "success"
        write_json(
            ctx.workspace.transcription_dir / "transcription_result.json",
            {
                "result": "success",
                "status_before": TARGET_STATUS,
                "status_after": SUCCESS_STATUS,
                "retries_before": selection.episode.get("retries", 0),
                "retries_after": 0,
                "message": "Transcription completed successfully.",
                "finished_at": finished_at,
            },
        )
        cleanup_temp_dir(ctx.workspace)
        ctx.write_status(
            status=SUCCESS_STATUS,
            current_step="cleanup",
            result="success",
            finished_at=finished_at,
        )
        ctx.write_report(result="success", finished_at=finished_at)
        ctx.log_event("transcription_completed", status_after=SUCCESS_STATUS, retries_after=0)
        return 0

    except Exception as exc:
        retry_consumed = not is_network_error(exc)
        retries_before = int(selection.episode.get("retries", 0) or 0)
        retries_after = retries_before
        status_after = TARGET_STATUS
        if retry_consumed:
            retries_after += 1
            status_after = RUINED_STATUS if retries_after > MAX_RETRIES else TARGET_STATUS
            update_episode_status(data, selection, status=status_after, retries=retries_after)

        finished_at = utc_now_iso()
        ctx.notes.append(str(exc))
        ctx.result = "failed"
        ctx.write_error(exc, retry_consumed=retry_consumed)
        write_json(
            ctx.workspace.transcription_dir / "transcription_result.json",
            {
                "result": "failed",
                "status_before": TARGET_STATUS,
                "status_after": status_after,
                "retries_before": retries_before,
                "retries_after": retries_after,
                "message": str(exc),
                "finished_at": finished_at,
            },
        )
        cleanup_temp_dir(ctx.workspace)
        ctx.write_status(
            status=status_after if retry_consumed else TARGET_STATUS,
            current_step="cleanup",
            result="failed",
            finished_at=finished_at,
        )
        ctx.write_report(result="failed", finished_at=finished_at)
        ctx.log_event(
            "transcription_failed",
            status_after=status_after,
            retries_after=retries_after,
            retry_consumed=retry_consumed,
            error_type=exc.__class__.__name__,
        )
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
