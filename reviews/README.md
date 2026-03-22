Cada episodio en `pending_review` debe tener:

- `reviews/<episode_id>/audit.yaml`

El usuario revisa identidades y corrige los `.srt` directamente.
Solo cuando `approved_as_source_of_truth: true` puede ejecutarse POSTAUDITORÍA.
