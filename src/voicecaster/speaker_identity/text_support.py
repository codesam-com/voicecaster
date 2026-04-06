from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


SELF_ID_PATTERNS = [
    re.compile(r"\bsoy\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)\b"),
    re.compile(r"\byo\s+soy\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)\b"),
    re.compile(r"\bmi\s+nombre\s+es\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)\b"),
    re.compile(r"\bos\s+habla\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)\b"),
]

# Uno o dos tokens con mayúscula inicial
NAME_CANDIDATE_PATTERN = re.compile(
    r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)\b"
)

# Stopwords / conectores / palabras frecuentes que no deben tomarse como nombres
STOPWORDS = {
    "a", "al", "algo", "algún", "alguna", "algunas", "algunos",
    "antes", "aquí", "así", "aunque", "aun",
    "bien", "bueno", "buenos",
    "cada", "casi", "claro", "como", "cómo", "con", "contra", "continuamos",
    "cual", "cuál", "cualquiera", "cuando", "cuándo",
    "de", "del", "desde", "después", "donde", "dos",
    "e", "el", "él", "ella", "ellas", "ellos", "en", "entre", "entonces", "era", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta", "está", "están", "estar", "este", "esto", "estos",
    "exacto", "efectivamente", "energía", "era",
    "familia", "fijaros",
    "gracias",
    "ha", "hasta", "hay",
    "igual",
    "la", "las", "le", "les", "lo", "los", "luego",
    "más", "mal", "me", "mi", "mis", "mientras", "muy",
    "nada", "no", "nos", "nosotros", "nuestra", "nuestro",
    "o", "otra", "otras", "otro", "otros", "os",
    "para", "pero", "poco", "por", "porque", "pues",
    "que", "qué", "quien", "quién", "quiero", "quizás",
    "se", "sí", "si", "siempre", "sin", "sobre", "son", "soy", "su", "sus",
    "también", "te", "tiene", "todo", "todos",
    "un", "una", "uno", "unos", "unas",
    "vale", "vamos", "viene",
    "ya", "yo",
}

# Palabras frecuentes al inicio de frase que a veces aparecen capitalizadas por puntuación
COMMON_SENTENCE_STARTERS = {
    "Entonces", "Pero", "Bueno", "Vale", "También", "Porque", "Cuando",
    "Cómo", "Como", "Vamos", "Continuamos", "Gracias", "Exacto", "Pues",
    "Luego", "Siempre", "Quiero", "Otra", "Entre", "Hasta", "Claro",
    "Esto", "Eso", "Ese", "Esa", "Los", "Las", "Son", "Por", "Qué", "Que",
    "Fijaros", "Efectivamente", "Cualquiera", "Quizás",
}


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


def _normalize_spaces(text: str) -> str:
    return " ".join(str(text).strip().split())


def _normalize_name(name: str) -> str:
    return _normalize_spaces(name).casefold()


def _build_participant_index(participants: Any) -> dict[str, str]:
    result: dict[str, str] = {}

    if not isinstance(participants, list):
        return result

    for item in participants:
        if not isinstance(item, str):
            continue

        cleaned = _normalize_spaces(item)
        if not cleaned:
            continue

        full_norm = _normalize_name(cleaned)
        result[full_norm] = cleaned

        parts = cleaned.split()
        if parts:
            first_norm = _normalize_name(parts[0])
            if first_norm and first_norm not in result:
                result[first_norm] = cleaned

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


def _is_valid_name_candidate(name: str) -> bool:
    cleaned = _normalize_spaces(name)
    if not cleaned:
        return False

    if cleaned in COMMON_SENTENCE_STARTERS:
        return False

    tokens = cleaned.split()
    if not tokens:
        return False

    # todos los tokens deben ser plausibles
    for token in tokens:
        token_norm = token.casefold()

        if len(token) < 3:
            return False

        if token_norm in STOPWORDS:
            return False

        # evitar tokens con signos raros
        if not re.fullmatch(r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+", token):
            return False

    return True


def _dedup_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for item in items:
        norm = _normalize_name(item)
        if norm in seen:
            continue
        seen.add(norm)
        result.append(_normalize_spaces(item))

    return result


def _find_self_identification_names(text: str) -> list[str]:
    found: list[str] = []

    for pattern in SELF_ID_PATTERNS:
        for match in pattern.finditer(text):
            name = _normalize_spaces(match.group(1))
            if _is_valid_name_candidate(name):
                found.append(name)

    return _dedup_preserve_order(found)


def _find_mentioned_names(text: str) -> list[str]:
    found: list[str] = []

    for match in NAME_CANDIDATE_PATTERN.finditer(text):
        name = _normalize_spaces(match.group(1))
        if _is_valid_name_candidate(name):
            found.append(name)

    return _dedup_preserve_order(found)


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
