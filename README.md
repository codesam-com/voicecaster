# voicecaster

Sistema automático y modular para procesar podcasts tipo tertulia desde URLs de audio.

## Entrada única
- `inputs/episodes.yaml`

## Salida histórica
- `inputs/episodes_processed.yaml`

## Estados
- `pending`
- `processing`
- `pending_review`
- `done`
- `failed`
- `incompatible`

## Módulos
- PREAUDITORÍA: automático
- POSTAUDITORÍA: solo si `reviews/<episode_id>/audit.yaml` contiene `approved_as_source_of_truth: true`

## Regla crítica
Nunca actualizar `data/speakers/` sin validación humana previa.
