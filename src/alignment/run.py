from pathlib import Path

from .loader import load_transcript_preview, load_speaker_segments
from .assign_words import assign_words
from .split_merge import build_utterances
from .export_srt import export_srt
from .metrics import compute_metrics


def run_alignment(work_dir: Path):

    t_path = work_dir / "02_transcription" / "transcript_preview.json"
    s_path = work_dir / "03_diarization" / "speaker_segments.json"

    transcript = load_transcript_preview(t_path)
    speakers = load_speaker_segments(s_path)

    aligned_words = assign_words(transcript, speakers)

    utterances = build_utterances(aligned_words)

    export_srt(work_dir / "04_alignment" / "subtitles_speakers.srt", utterances)

    metadata = compute_metrics(aligned_words, utterances)

    print("Alignment OK")
    print(metadata)
