from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


SELF_ID_PATTERNS = [
    re.compile(r"\bsoy\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\b"),
    re.compile(r"\byo\s+soy\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\b"),
    re.compile(r"\bmi\s+nombre\s+es\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\b"),
    re.compile(r"\bos\s+habla\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\b"),
]

NAME_MENTION_PATTERN = re.compile(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\b")


@dataclass(slots=True)
class SpeakerTextEvidence:
    speaker: str
    self_identification_detected: bool
    self_identification_names: list[str]
    mentioned_names: list[str]
    participant_matches: list[str]
    text_score_hint: float
    evidence_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_name(name: str) -> str:
    return " ".join(str(name).strip().split()).casefold()


def _build_participant_index(participants: Any) -> dict[str, str]:
    result: dict[str, str] = {}

    if not isinstance(participants, list):
        return result

    for item in participants:
        if not isinstance(item, str):
            continue
        normalized = _normalize_name(item)
        if normalized:
            result[normalized] = item.strip()

        parts = item.strip().split()
        if parts:
            first = _normalize_name(parts[0])
            if first and first not in result:
                result[first] = item.strip()

    return result


def _extract_text_by_speaker(
    transcript_with_speakers: list[dict[str, Any]],
) -> dict[str, list[str]]:
    texts_by_speaker: dict[str, list[str]] = {}

    for item in transcript_with_speakers:
        speaker = str(item.get("speaker") or "").strip()
        text = str(item.get("text") or "").strip()

        if not speaker or not text:
            continue

        texts_by_speaker.setdefault(speaker, []).append(text)

    return texts_by_speaker


def _find_self_identification_names(text: str) -> list[str]:
    found: list[str] = []

    for pattern in SELF_ID_PATTERNS:
        for match in pattern.finditer(text):
            name = match.group(1).strip()
            if name and name not in found:
                found.append(name)

    return found


def _find_mentioned_names(text: str) -> list[str]:
    found: list[str] = []

    for match in NAME_MENTION_PATTERN.finditer(text):
        name = match.group(1).strip()

        if len(name) < 3:
            continue
        if name.lower() in {"sí", "no"}:
            continue

        if name not in found:
            found.append(name)

    return found


def build_text_evidence(
    transcript_with_speakers: list[dict[str, Any]],
    participants: Any,
) -> list[SpeakerTextEvidence]:
    texts_by_speaker = _extract_text_by_speaker(transcript_with_speakers)
    participant_index = _build_participant_index(participants)

    results: list[SpeakerTextEvidence] = []

    for speaker, text_chunks in texts_by_speaker.items():
        joined_text = "\n".join(text_chunks)

        self_names = _find_self_identification_names(joined_text)
        mentioned_names = _find_mentioned_names(joined_text)

        participant_matches: list[str] = []
        for name in self_names + mentioned_names:
            normalized = _normalize_name(name)
            if normalized in participant_index:
                resolved = participant_index[normalized]
                if resolved not in participant_matches:
                    participant_matches.append(resolved)

        notes: list[str] = []
        text_score_hint = 0.0

        if self_names:
            notes.append("explicit_self_identification_detected")
            text_score_hint += 0.75

        if participant_matches:
            notes.append("participant_name_match_detected")
            text_score_hint += 0.15

        if mentioned_names:
            notes.append("name_mentions_detected")
            text_score_hint += 0.05

        text_score_hint = min(1.0, round(text_score_hint, 4))

        results.append(
            SpeakerTextEvidence(
                speaker=speaker,
                self_identification_detected=bool(self_names),
                self_identification_names=self_names,
                mentioned_names=mentioned_names,
                participant_matches=participant_matches,
                text_score_hint=text_score_hint,
                evidence_notes=notes,
            )
        )

    return results
