from .schemas import Utterance, AlignedWord


def build_utterances(words: list[AlignedWord]) -> list[Utterance]:
    if not words:
        return []

    utterances = []
    current = [words[0]]

    for w in words[1:]:
        if w.speaker == current[-1].speaker:
            current.append(w)
        else:
            utterances.append(_make_utt(current))
            current = [w]

    utterances.append(_make_utt(current))
    return utterances


def _make_utt(words):
    return Utterance(
        start=words[0].start,
        end=words[-1].end,
        speaker=words[0].speaker,
        text=" ".join(w.word for w in words),
        words=words,
    )
