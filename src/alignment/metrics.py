from .schemas import AlignedWord, Utterance, AlignmentMetadata


def compute_metrics(words: list[AlignedWord], utterances: list[Utterance]):

    total = len(words)
    unknown = sum(1 for w in words if w.speaker == "UNKNOWN")

    ratio = 1 - (unknown / total if total else 0)

    warnings = []
    if unknown / total > 0.01:
        warnings.append("high_unknown_word_ratio")

    return AlignmentMetadata(
        word_assignment_ratio=ratio,
        utterance_count=len(utterances),
        unknown_word_ratio=unknown / total if total else 0,
        warnings=warnings,
    )
