# 🧠 PR — 04_alignment

## 🧩 Contexto

Describe brevemente qué hace esta PR.

- Tipo de cambio:
  - [ ] Nueva implementación
  - [ ] Fix
  - [ ] Refactor
  - [ ] Mejora de calidad
  - [ ] Otro:

- Objetivo:
  - [ ] Integración base
  - [ ] Mejora algorítmica
  - [ ] Corrección de bug
  - [ ] Mejora de outputs
  - [ ] Otro:

---

## 🏗️ Arquitectura

Confirma que esta PR respeta:

- [ ] `filesystem = source of truth`
- [ ] 1 workflow = 1 episodio
- [ ] 1 workflow = 1 responsabilidad
- [ ] No introduce estado externo (DB, cache, etc.)
- [ ] No persiste audio
- [ ] No rompe contratos de `02_transcription` ni `03_diarization`

---

## 📁 Estructura

- [ ] Existe `src/voicecaster/alignment/`
- [ ] Archivos incluidos:

```text
__init__.py
run.py
loader.py
normalizer.py
assign_words.py
assign_segments.py
split_merge.py
export_srt.py
metrics.py
schemas.py
