from __future__ import annotations

from .models import AlignedTurn, AlignedUtterance, AlignedWord, QAIssue, QAResult


def _is_monotonic_utterances(utterances: list[AlignedUtterance]) -> bool:
    prev_start = None
    prev_end = None

    for utt in utterances:
        if utt.end <= utt.start:
            return False
        if utt.start < 0:
            return False

        if prev_start is not None and utt.start < prev_start:
            return False
        if prev_end is not None and utt.end < prev_end:
            return False

        prev_start = utt.start
        prev_end = utt.end

    return True


def _is_monotonic_turns(turns: list[AlignedTurn]) -> bool:
    prev_start = None
    prev_end = None

    for turn in turns:
        if turn.end <= turn.start:
            return False
        if turn.start < 0:
            return False

        if prev_start is not None and turn.start < prev_start:
            return False
        if prev_end is not None and turn.end < prev_end:
            return False

        prev_start = turn.start
        prev_end = turn.end

    return True


def _text_non_empty_ratio(utterances: list[AlignedUtterance]) -> float:
    if not utterances:
        return 0.0
    non_empty = sum(1 for utt in utterances if utt.text.strip())
    return non_empty / len(utterances)


def _turn_reconstruction_ok(
    utterances: list[AlignedUtterance],
    turns: list[AlignedTurn],
) -> bool:
    if not utterances or not turns:
        return False

    utterance_ids = {utt.utterance_id for utt in utterances}
    turn_utterance_ids: set[str] = set()

    for turn in turns:
        for utt_id in turn.utterance_ids:
            if utt_id in turn_utterance_ids:
                return False
            turn_utterance_ids.add(utt_id)

    return utterance_ids == turn_utterance_ids


def _word_coverage_ok(
    utterances: list[AlignedUtterance],
    words: list[AlignedWord],
) -> bool:
    words_expected = any(len(utt.words) > 0 for utt in utterances)
    return (not words_expected) or bool(words)


def _word_membership_ok(
    utterances: list[AlignedUtterance],
    words: list[AlignedWord],
) -> bool:
    utterance_ids = {utt.utterance_id for utt in utterances}
    return all(word.utterance_id in utterance_ids for word in words)


def _utterance_word_bounds_ok(utterances: list[AlignedUtterance]) -> tuple[bool, int]:
    invalid_count = 0

    for utt in utterances:
        for word in utt.words:
            start = word.get("start")
            end = word.get("end")

            if start is None or end is None:
                continue

            start = float(start)
            end = float(end)

            if end < start:
                invalid_count += 1
                continue

            if start < utt.start or end > utt.end:
                invalid_count += 1

    return invalid_count == 0, invalid_count


def _turn_text_consistency_ok(
    utterances: list[AlignedUtterance],
    turns: list[AlignedTurn],
) -> tuple[bool, int]:
    utterance_map = {utt.utterance_id: utt for utt in utterances}
    mismatch_count = 0

    for turn in turns:
        rebuilt = " ".join(
            utterance_map[utt_id].text.strip()
            for utt_id in turn.utterance_ids
            if utt_id in utterance_map and utterance_map[utt_id].text.strip()
        ).strip()

        if rebuilt != turn.text.strip():
            mismatch_count += 1

    return mismatch_count == 0, mismatch_count


def _speaker_index_consistency_ok(
    utterances: list[AlignedUtterance],
    turns: list[AlignedTurn],
    words: list[AlignedWord],
) -> tuple[bool, dict[str, int]]:
    utterance_speakers = {utt.speaker or "unknown" for utt in utterances}
    turn_speakers = {turn.speaker or "unknown" for turn in turns}
    word_speakers = {word.speaker or "unknown" for word in words}

    all_ok = turn_speakers.issubset(utterance_speakers) and word_speakers.issubset(utterance_speakers)

    details = {
        "utterance_speakers": len(utterance_speakers),
        "turn_speakers": len(turn_speakers),
        "word_speakers": len(word_speakers),
    }
    return all_ok, details


def _low_confidence_count(utterances: list[AlignedUtterance]) -> int:
    return sum(
        1
        for utt in utterances
        if utt.speaker_confidence is not None and float(utt.speaker_confidence) < 0.75
    )


def _unknown_speaker_count(utterances: list[AlignedUtterance]) -> int:
    return sum(1 for utt in utterances if utt.speaker is None)


def _words_without_timestamps_count(words: list[AlignedWord]) -> int:
    return sum(1 for word in words if word.start is None or word.end is None)


def _long_turns_count(turns: list[AlignedTurn], threshold_seconds: float = 45.0) -> int:
    return sum(1 for turn in turns if float(turn.duration) > threshold_seconds)


def run_alignment_qa(
    utterances: list[AlignedUtterance],
    turns: list[AlignedTurn],
    words: list[AlignedWord],
) -> QAResult:
    warnings: list[QAIssue] = []
    issues: list[QAIssue] = []

    utterance_word_bounds_ok, invalid_word_bounds_count = _utterance_word_bounds_ok(utterances)
    turn_text_consistency_ok, turn_text_mismatch_count = _turn_text_consistency_ok(utterances, turns)
    speaker_index_consistency_ok, speaker_index_details = _speaker_index_consistency_ok(
        utterances, turns, words
    )

    checks = {
        "monotonic_utterance_timestamps": _is_monotonic_utterances(utterances),
        "monotonic_turn_timestamps": _is_monotonic_turns(turns),
        "speaker_presence": any(utt.speaker is not None for utt in utterances),
        "text_non_empty_ratio_ok": _text_non_empty_ratio(utterances) >= 0.95,
        "turn_reconstruction_ok": _turn_reconstruction_ok(utterances, turns),
        "word_coverage_ok": _word_coverage_ok(utterances, words),
        "word_membership_ok": _word_membership_ok(utterances, words),
        "utterance_word_bounds_ok": utterance_word_bounds_ok,
        "turn_text_consistency_ok": turn_text_consistency_ok,
        "speaker_index_consistency_ok": speaker_index_consistency_ok,
    }

    low_confidence_count = _low_confidence_count(utterances)
    if low_confidence_count > 0:
        warnings.append(
            QAIssue(
                severity="warning",
                code="low_confidence_assignments_present",
                message="Some utterances have low speaker confidence.",
                details={"count": low_confidence_count},
            )
        )

    unknown_speaker_count = _unknown_speaker_count(utterances)
    if unknown_speaker_count > 0:
        warnings.append(
            QAIssue(
                severity="warning",
                code="unknown_speakers_present",
                message="Some utterances have no speaker assigned.",
                details={"count": unknown_speaker_count},
            )
        )

    words_without_timestamps = _words_without_timestamps_count(words)
    if words_without_timestamps > 0:
        warnings.append(
            QAIssue(
                severity="warning",
                code="words_without_timestamps",
                message="Some words are missing timestamps.",
                details={"count": words_without_timestamps},
            )
        )

    long_turns_count = _long_turns_count(turns)
    if long_turns_count > 0:
        warnings.append(
            QAIssue(
                severity="warning",
                code="long_turns_present",
                message="Some turns are longer than the recommended threshold.",
                details={"count": long_turns_count, "threshold_seconds": 45.0},
            )
        )

    if invalid_word_bounds_count > 0:
        warnings.append(
            QAIssue(
                severity="warning",
                code="word_bounds_inconsistent",
                message="Some words fall outside their utterance time bounds.",
                details={"count": invalid_word_bounds_count},
            )
        )

    if turn_text_mismatch_count > 0:
        warnings.append(
            QAIssue(
                severity="warning",
                code="turn_text_mismatches_present",
                message="Some turn texts do not match the concatenation of their utterances.",
                details={"count": turn_text_mismatch_count},
            )
        )

    for check_name, passed in checks.items():
        if not passed:
            extra_details = {}

            if check_name == "utterance_word_bounds_ok":
                extra_details = {"invalid_word_bounds_count": invalid_word_bounds_count}
            elif check_name == "turn_text_consistency_ok":
                extra_details = {"turn_text_mismatch_count": turn_text_mismatch_count}
            elif check_name == "speaker_index_consistency_ok":
                extra_details = speaker_index_details

            issues.append(
                QAIssue(
                    severity="error",
                    code=check_name,
                    message=f"QA check failed: {check_name}",
                    details=extra_details,
                )
            )

    passed = len(issues) == 0

    return QAResult(
        passed=passed,
        checks=checks,
        warnings=warnings,
        issues=issues,
    )
