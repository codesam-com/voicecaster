from __future__ import annotations

from dataclasses import replace

from .schemas import AlignedUtterance, AlignedWord, SubtitleCue


def shape_subtitles(
    utterances: list[AlignedUtterance],
    *,
    max_cue_duration: float,
    target_cue_duration: float,
    min_cue_duration: float,
    max_chars_per_line: int,
    max_lines_per_cue: int,
    pause_split_threshold: float,
) -> list[SubtitleCue]:
    """
    Convert canonical aligned utterances into human-readable subtitle cues.

    Important:
    - does not change speaker identity
    - does not mix speakers
    - does not cross utterance boundaries
    """
    cues: list[SubtitleCue] = []
    cue_counter = 0

    for utterance in utterances:
        utterance_cues = shape_utterance_into_cues(
            utterance,
            max_cue_duration=max_cue_duration,
            target_cue_duration=target_cue_duration,
            min_cue_duration=min_cue_duration,
            max_chars_per_line=max_chars_per_line,
            max_lines_per_cue=max_lines_per_cue,
            pause_split_threshold=pause_split_threshold,
        )

        for cue in utterance_cues:
            cues.append(
                replace(
                    cue,
                    cue_id=f"cue_{cue_counter:06d}",
                )
            )
            cue_counter += 1

    return cues


def shape_utterance_into_cues(
    utterance: AlignedUtterance,
    *,
    max_cue_duration: float,
    target_cue_duration: float,
    min_cue_duration: float,
    max_chars_per_line: int,
    max_lines_per_cue: int,
    pause_split_threshold: float,
) -> list[SubtitleCue]:
    """
    Split one utterance into shorter subtitle cues.
    """
    if not utterance.words:
        lines = split_text_into_lines(
            utterance.text,
            max_chars_per_line=max_chars_per_line,
            max_lines_per_cue=max_lines_per_cue,
        )
        return [
            SubtitleCue(
                cue_id="",
                utterance_id=utterance.utterance_id,
                source_segment_ids=utterance.source_segment_ids,
                start=utterance.start,
                end=utterance.end,
                speaker=utterance.speaker,
                text="\n".join(lines),
                line_count=len(lines),
                char_count=sum(len(line) for line in lines),
                word_count=0,
                flags=["no_words_in_utterance"],
            )
        ]

    words = utterance.words
    cues: list[SubtitleCue] = []

    start_index = 0
    while start_index < len(words):
        end_index = choose_best_chunk_end(
            words=words,
            start_index=start_index,
            max_cue_duration=max_cue_duration,
            target_cue_duration=target_cue_duration,
            min_cue_duration=min_cue_duration,
            max_chars_per_line=max_chars_per_line,
            max_lines_per_cue=max_lines_per_cue,
            pause_split_threshold=pause_split_threshold,
        )

        chunk_words = words[start_index:end_index]
        raw_text = join_words_text(chunk_words)
        lines = split_text_into_lines(
            raw_text,
            max_chars_per_line=max_chars_per_line,
            max_lines_per_cue=max_lines_per_cue,
        )

        flags: list[str] = []
        duration = chunk_words[-1].end - chunk_words[0].start
        if duration > max_cue_duration:
            flags.append("overlong_cue")

        cues.append(
            SubtitleCue(
                cue_id="",
                utterance_id=utterance.utterance_id,
                source_segment_ids=utterance.source_segment_ids,
                start=chunk_words[0].start,
                end=chunk_words[-1].end,
                speaker=utterance.speaker,
                text="\n".join(lines),
                line_count=len(lines),
                char_count=max((len(line) for line in lines), default=0),
                word_count=len(chunk_words),
                flags=flags,
            )
        )

        start_index = end_index

    return cues


def choose_best_chunk_end(
    *,
    words: list[AlignedWord],
    start_index: int,
    max_cue_duration: float,
    target_cue_duration: float,
    min_cue_duration: float,
    max_chars_per_line: int,
    max_lines_per_cue: int,
    pause_split_threshold: float,
) -> int:
    """
    Return exclusive end index for the next subtitle chunk.
    """
    best_index: int | None = None
    best_score = float("-inf")

    max_chars_per_cue = max_chars_per_line * max_lines_per_cue
    start_time = words[start_index].start

    for end_index in range(start_index + 1, len(words) + 1):
        chunk = words[start_index:end_index]
        chunk_duration = chunk[-1].end - start_time
        chunk_text = join_words_text(chunk)
        chunk_chars = len(chunk_text)

        if chunk_duration > max_cue_duration * 1.35:
            break

        score = 0.0

        if chunk_duration >= min_cue_duration:
            score += 15.0
        else:
            score -= 20.0

        score -= abs(chunk_duration - target_cue_duration) * 4.0

        if chunk_chars <= max_chars_per_cue:
            score += 15.0
        else:
            score -= (chunk_chars - max_chars_per_cue) * 2.0

        if end_index < len(words):
            score += score_breakpoint(
                words=words,
                break_index=end_index,
                pause_split_threshold=pause_split_threshold,
            )
        else:
            score += 10.0

        if score > best_score:
            best_score = score
            best_index = end_index

        if chunk_duration >= max_cue_duration and chunk_chars >= max_chars_per_cue * 0.7:
            break

    if best_index is None:
        return min(start_index + 1, len(words))

    return best_index


def score_breakpoint(
    *,
    words: list[AlignedWord],
    break_index: int,
    pause_split_threshold: float,
) -> float:
    """
    Score a breakpoint between words[break_index - 1] and words[break_index].
    Higher is better.
    """
    prev_word = words[break_index - 1]
    next_word = words[break_index]

    score = 0.0
    prev_text = prev_word.word.strip()
    next_text = next_word.word.strip().lower()

    if prev_text.endswith((".", "?", "!")):
        score += 100.0
    elif prev_text.endswith((",", ";", ":")):
        score += 50.0

    gap = max(0.0, next_word.start - prev_word.end)
    if gap >= pause_split_threshold:
        score += 35.0
    elif gap >= pause_split_threshold * 0.5:
        score += 12.0

    if prev_text.lower() in AVOID_BREAK_AFTER:
        score -= 25.0

    if next_text in AVOID_BREAK_BEFORE:
        score -= 20.0

    return score


def split_text_into_lines(
    text: str,
    *,
    max_chars_per_line: int,
    max_lines_per_cue: int,
) -> list[str]:
    """
    Break subtitle text into 1 or 2 lines.

    Conservative implementation:
    - 1 line if it fits
    - otherwise split near the middle on word boundaries
    """
    text = " ".join(text.split())

    if len(text) <= max_chars_per_line or max_lines_per_cue <= 1:
        return [text]

    words = text.split()
    if len(words) <= 1:
        return [text]

    best_split: int | None = None
    best_score = float("inf")

    for idx in range(1, len(words)):
        left = " ".join(words[:idx])
        right = " ".join(words[idx:])

        if len(left) > max_chars_per_line or len(right) > max_chars_per_line:
            continue

        score = abs(len(left) - len(right))

        if words[idx - 1].endswith((",", ";", ":")):
            score -= 8

        if words[idx - 1].lower() in AVOID_BREAK_AFTER:
            score += 20

        if words[idx].lower() in AVOID_BREAK_BEFORE:
            score += 20

        if score < best_score:
            best_score = score
            best_split = idx

    if best_split is None:
        return [text]

    return [
        " ".join(words[:best_split]),
        " ".join(words[best_split:]),
    ]


def join_words_text(words: list[AlignedWord]) -> str:
    """
    Conservative token join.

    Current repo words already behave like tokenized text with punctuation attached.
    """
    parts: list[str] = []
    for word in words:
        token = word.word.strip()
        if not token:
            continue
        if not parts:
            parts.append(token)
            continue

        if token in {".", ",", ";", ":", "!", "?"}:
            parts[-1] = parts[-1] + token
        else:
            parts.append(token)

    return " ".join(parts).strip()


AVOID_BREAK_AFTER = {
    "de", "la", "el", "y", "o", "que", "en", "por", "con", "un", "una", "del", "al"
}

AVOID_BREAK_BEFORE = {
    "de", "la", "el", "y", "o", "que", "en", "por", "con", "un", "una", "del", "al"
}
