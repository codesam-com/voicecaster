````markdown id="inputs-readme-v2"
# 📥 inputs/ — Entrada del sistema

Este directorio contiene el **único punto de entrada** y el **histórico de salida** del sistema.

---

# 🧠 Principio clave

```text
filesystem = source of truth
````

* No hay base de datos externa
* No hay estado oculto
* Todo se deduce de los archivos del repositorio

---

# 📂 Archivos del sistema

## 🔹 Entrada activa

```text
inputs/inputs.json
```

* Cola de episodios a procesar
* Editable por el usuario

---

## 🔹 Histórico procesado

```text
inputs/processed.jsonl.gz
```

* Histórico append-only de episodios finalizados
* Comprimido
* No editable manualmente

---

# 📄 Formato: inputs/inputs.json

## 🧱 Estructura

* JSON válido
* Lista de episodios

---

## 📋 Plantilla (copiar y pegar)

```json
  {
    "id": "episodio_001",
    "podcast_title": "Nombre del podcast",
    "episode_title": "Título del episodio",
    "url": "https://...",

    "participants": null,
    "status": "intake",
    "retries": 0
  }
```

---

## 🔍 Descripción de campos

### 🔹 id

* Identificador único
* Obligatorio

---

### 🔹 podcast_title

* Nombre del podcast
* Obligatorio

---

### 🔹 episode_title

* Nombre del episodio
* Obligatorio

---

### 🔹 url

* URL directa a audio descargable
* Obligatorio

---

### 🔹 participants

* Lista opcional proporcionada por el usuario
* Puede ser `null`
* ⚠️ No es fuente de verdad

---

### 🔹 status

```json
"status": "intake"
```

* Campo clave del sistema
* Define qué workflow procesa el episodio
* Valor inicial obligatorio: `"intake"`

---

### 🔹 retries

* Número de intentos
* Inicial: `0`
* Gestionado automáticamente

---

# ⚙️ Funcionamiento

```text
Usuario añade episodio → status=intake → sistema procesa automáticamente
```

* Solo se procesa **1 episodio por ejecución**
* El sistema actualiza el estado internamente

---

# 🔄 Estados (visión general)

```text
intake → processing → pending_review → done
```

Errores:

```text
failed
incompatible
```

⚠️ El usuario no debe modificar estados manualmente

---

# 🚫 Reglas importantes

* No duplicar `id`
* No editar episodios en proceso
* No cambiar estados manualmente
* La URL debe ser descargable

---

# ✅ Ejemplo: inputs/inputs.json

```json
[
  {
    "id": "episodio_001",
    "podcast_title": "Podcast de prueba",
    "episode_title": "Episodio largo de validación",
    "url": "https://drive.google.com/uc?export=download&id=1A4AjcUrZR8TN8vqvOvU-GB97jZpnQaRZ",
    "participants": null,
    "status": "intake",
    "retries": 0
  },
  {
    "id": "episodio_002",
    "podcast_title": "Podcast IA",
    "episode_title": "Debate sobre AGI",
    "url": "https://example.com/audio.mp3",
    "participants": ["Ana López", "Carlos Ruiz"],
    "status": "intake",
    "retries": 0
  }
]
```

---

# 📦 Formato: inputs/processed.jsonl.gz

## 🧱 Características

* Formato: JSON Lines (`.jsonl`)
* Cada línea = 1 episodio finalizado
* Comprimido con gzip (`.gz`)
* Append-only (nunca modificar entradas existentes)

---

## 📄 Estructura de cada línea

Cada línea es un JSON independiente:

```json
{"id":"episodio_001","podcast_title":"Podcast de prueba","episode_title":"Episodio largo de validación","url":"https://...","status":"done","processed_at":"2026-03-30T12:00:00Z"}
```

---

## 🔍 Campos típicos

* `id`
* `podcast_title`
* `episode_title`
* `url`
* `status`: siempre `"done"`
* `processed_at`: timestamp ISO
* (opcional) metadatos adicionales generados por el sistema

---

## ✅ Ejemplo (sin compresión, para visualizar)

```json
{"id":"episodio_001","podcast_title":"Podcast de prueba","episode_title":"Episodio largo de validación","url":"https://...","status":"done","processed_at":"2026-03-30T12:00:00Z"}
{"id":"episodio_002","podcast_title":"Podcast IA","episode_title":"Debate sobre AGI","url":"https://example.com/audio.mp3","status":"done","processed_at":"2026-03-30T15:42:10Z"}
```

---

# 🔄 Flujo completo

```text
inputs/inputs.json
   ↓
(procesamiento automático)
   ↓
inputs/processed.jsonl.gz (append)
```

---

# 🚀 Resumen

```text
Añades episodios en inputs.json → status=intake → el sistema procesa → se almacenan en processed.jsonl.gz
```

```
```
