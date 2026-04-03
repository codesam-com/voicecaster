from __future__ import annotations

from .models import AlignedTurn, AlignedUtterance, AlignedWord, QAIssue, QAResult


def _is_monotonic_utterances(utterances: list[AlignedUtterance]) -> bool:
    prev_end = None
    for utt in utterances:
        if utt.end <= utt.start:
            return False
        if prev_end is not None and utt.start < 0:
            return False
        prev_end = utt.end
    return True


def _is_monotonic_turns(turns: list[AlignedTurn]) -> bool:
    for turn in turns:
        if turn.end <= turn.start:
            return False
    return True


def run_alignment_qa(
    utterances: list[AlignedUtterance],
    turns: list[AlignedTurn],
    words: list[AlignedWord],
) -> QAResult:
    warnings: list[QAIssue] = []
    issues: list[QAIssue] = []

    checks = {
        "monotonic_utterance_timestamps": _is_monotonic_utterances(utterances),
        "monotonic_turn_timestamps": _is_monotonic_turns(turns),
        "speaker_presence": any(utt.speaker is not None for utt in utterances),
        "text_non_empty_ratio_ok": False,
        "turn_reconstruction_ok": False,
        "word_coverage_ok": False,
        "word_membership_ok": False,
    }

    if utterances:
        non_empty_text = sum(1 for utt in utterances if utt.text.strip())
        text_ratio = non_empty_text / len(utterances)
        checks["text_non_empty_ratio_ok"] = text_ratio >= 0.95
    else:
        checks["text_non_empty_ratio_ok"] = False

    turn_utterance_ids = set()
    for turn in turns:
        for utt_id in turn.utterance_ids:
            turn_utterance_ids.add(utt_id)

    utterance_ids = {utt.utterance_id for utt in utterances}
    checks["turn_reconstruction_ok"] = utterance_ids == turn_utterance_ids and bool(turns)

    words_expected = any(len(utt.words) > 0 for utt in utterances)
    checks["word_coverage_ok"] = (not words_expected) or bool(words)

    word_utterance_ids = {word.utterance_id for word in words}
    checks["word_membership_ok"] = word_utterance_ids.issubset(utterance_ids)

    low_confidence_count = sum(
        1
        for utt in utterances
        if utt.speaker_confidence is not None and float(utt.speaker_confidence) < 0.75
    )
    if low_confidence_count > 0:
        warnings.append(
            QAIssue(
                severity="warning",
                code="low_confidence_assignments_present",
                message="Some utterances have low speaker confidence.",
                details={"count": low_confidence_count},
            )
        )

    unknown_speaker_count = sum(1 for utt in utterances if utt.speaker is None)
    if unknown_speaker_count > 0:
        warnings.append(
            QAIssue(
                severity="warning",
                code="unknown_speakers_present",
                message="Some utterances have no speaker assigned.",
                details={"count": unknown_speaker_count},
            )
        )

    words_without_timestamps = sum(
        1 for word in words if word.start is None or word.end is None
    )
    if words_without_timestamps > 0:
        warnings.append(
            QAIssue(
                severity="warning",
                code="words_without_timestamps",
                message="Some words are missing timestamps.",
                details={"count": words_without_timestamps},
            )
        )

    for check_name, passed in checks.items():
        if not passed:
            issues.append(
                QAIssue(
                    severity="error",
                    code=check_name,
                    message=f"QA check failed: {check_name}",
                    details={},
                )
            )

    passed = len(issues) == 0

    return QAResult(
        passed=passed,
        checks=checks,
        warnings=warnings,
        issues=issues,
    )
