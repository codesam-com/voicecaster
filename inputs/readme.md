Perfecto, ajustamos el README con el nombre correcto del archivo.

Aquí lo tienes listo para copiar y pegar:

````markdown
# 📥 inputs/ — Cola de episodios

Este directorio contiene el **único punto de entrada del sistema**.

```text
inputs/inputs.json
````

Aquí el usuario añade episodios que serán procesados automáticamente por las GitHub Actions.

---

# 🧠 Principio clave

```text
filesystem = source of truth
```

* No hay base de datos externa
* No hay estado oculto
* Todo el sistema se guía por este archivo

---

# 📄 Formato del archivo

El archivo debe ser:

```text
JSON válido
```

Estructura:

* Lista (`[]`) de episodios
* Cada episodio es un objeto independiente

---

# 🧱 Plantilla (copiar y pegar)

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

# 🔍 Descripción de campos

### 🔹 id

* Identificador único del episodio
* Obligatorio
* No debe repetirse

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

* URL directa a un archivo de audio descargable (normalmente `.mp3`)
* Obligatorio

---

### 🔹 participants

* Lista opcional de participantes proporcionada por el usuario
* Puede ser `null`
* ⚠️ No se considera fuente de verdad (solo pista inicial)

Ejemplo:

```json
"participants": ["Nombre 1", "Nombre 2"]
```

---

### 🔹 status

* Estado del episodio
* Define qué workflow lo procesará

Valor inicial obligatorio:

```json
"status": "intake"
```

---

### 🔹 retries

* Número de intentos de procesamiento
* Inicial: `0`
* Gestionado automáticamente por el sistema

---

# ⚙️ Cómo funciona el sistema

1. El usuario añade un episodio con:

```json
"status": "intake"
```

2. El workflow correspondiente lo detecta

3. El sistema cambia el estado automáticamente según avanza

---

# 🔄 Estados (visión general)

```text
intake → processing → pending_review → done
```

Errores posibles:

```text
failed
incompatible
```

⚠️ El usuario **NO debe modificar manualmente estos estados** salvo para añadir nuevos episodios.

---

# 🚫 Reglas importantes

* No duplicar `id`
* No modificar episodios en proceso
* No cambiar estados manualmente (salvo añadir nuevos con `intake`)
* La URL debe ser descargable directamente

---

# ✅ Ejemplo completo

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
    "podcast_title": "Otro podcast",
    "episode_title": "Entrevista sobre IA",
    "url": "https://example.com/audio.mp3",
    "participants": ["Ana", "Carlos"],
    "status": "intake",
    "retries": 0
  }
]
```

---

# 🧩 Notas finales

* Este archivo actúa como **cola de procesamiento**
* Solo se procesa **1 episodio por ejecución**
* Los episodios completados se moverán a:

```text
inputs/inputs_processed.json
```

---

# 🚀 Resumen

```text
Añade episodios → status=intake → el sistema hace el resto
```

```
```
