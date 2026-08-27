---
title: Knowledge Nexus LATAM
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Knowledge Nexus LATAM — Prototipo

Sistema de descubrimiento, priorización y trazabilidad de conocimiento institucional
sobre **Data V1.0 RC2** (universidad ficticia UNEXA), construido para el hackathon
Knowledge Nexus LATAM (Talento TECH, nivel avanzado).

Responde a las siete preguntas del Documento Técnico Maestro (§17): **qué conecta,
cómo se relaciona, qué tan relevante es, por qué, con qué evidencia, de dónde proviene
y para qué sirve** — para cualquier consulta nueva, no solo para ejemplos precargados.

## 1. Qué hace

```
Consulta (texto libre o necesidad del catálogo)
   -> antecedentes priorizados (proyectos, tesis, publicaciones, líneas)
   -> propagación por el grafo institucional (investigadores, grupos)
   -> articulación curricular (asignaturas) y capacidades activables
   -> oportunidades tipadas (continuidad, colaboración, integración curricular, ...)
   -> explicación determinista + evidencia citada campo a campo
```

Cada resultado trae: origen, destino, tipo de relación, relevancia (score 0-1 con
desglose por componente), explicación, evidencia textual literal y su procedencia
exacta (`Data V1.0 / archivo.csv / ID / campo`).

## 2. Stack técnico (100% gratuito, sin claves ni cuotas)

| Capa | Tecnología |
|---|---|
| Datos | Python 3.13, `pandas`, `pyarrow` |
| Grafo de conocimiento | `networkx` (MultiDiGraph), persistido con `pickle` |
| Embeddings semánticos | `fastembed` + `paraphrase-multilingual-MiniLM-L12-v2` (384 dim, Apache-2.0), **local sobre ONNX Runtime, sin red tras la primera descarga del modelo** |
| Señal léxica | `scikit-learn` (TF-IDF) |
| Interfaz | `dash` + `dash-bootstrap-components` + `plotly` |
| Capa narrativa | **Opcional y desactivada por defecto** (`LLM_HABILITADO=false`); conector enchufable para Gemini/Groq free tier si se desea prosa generada, nunca sustituye la explicación determinista |

**No se usa ninguna API externa, clave ni servicio de pago en el camino crítico.**
Esto no es solo ahorro: elimina el riesgo de que una API externa falle durante la
evaluación (Documento Técnico §14, "¿qué pasa si una API externa falla?") y hace el
sistema 100% reproducible.

## 3. Instalación

```bash
py -m venv venv --copies
venv/Scripts/pip install -r requirements.txt
```

La primera ejecución descarga el modelo de embeddings (~220 MB, una sola vez, se
cachea en `modelos/`). Todo lo demás corre offline.

## 4. Ejecución

```bash
# 1. Construir el índice (ingesta + grafo + facetas + embeddings). Primera vez: ~8 min
#    en CPU (vectoriza 3232 entidades). Ejecuciones siguientes: segundos (caché por hash
#    del corpus, ver src/embeddings.py).
python scripts/construir_indice.py

# 2. Levantar el dashboard
python -m src.app
# abre http://127.0.0.1:8050
```

Reproducir la demo: escribe una consulta nueva en tus propias palabras (o carga una de
las 42 necesidades del catálogo), presiona **Buscar**, y navega las pestañas
*Conexiones -> Grafo -> Oportunidades -> ¿Por qué A antes que B?*.

## 5. Datos

Data V1.0 RC2 se usa **intacta**, sin sobrescribir (`KNOWLEDGE_NEXUS_LATAM_DATA_V1_RC2_PARTICIPANTS/`).
Los artefactos derivados (parquet normalizado, grafo, embeddings, vocabulario de
facetas) viven aparte en `data/processed/` y `data/artifacts/`.

Todos los CSV se leen con `encoding='utf-8-sig'`: los 21 archivos traen BOM.

## 6. Mecanismo de descubrimiento y ranking

Motor híbrido en dos etapas ([`src/motor.py`](src/motor.py)):

**Etapa A — entidades documentales** (proyecto/tesis/publicación/línea). Score
descompuesto y visible:

```
S(Q,E) = w_sem·similitud_semántica + w_lex·similitud_léxica
       + w_tema·solape_TEMA + w_dominio·solape_DOMINIO + w_metodo·solape_MÉTODO
       + w_evidencia·fuerza_de_evidencia
       (penalizado si el único solape es de método, sin tema en común)
```

Reglas del Documento Técnico §16 implementadas literalmente:
- **"Mismo método no significa mismo problema"**: penalización explícita + re-etiquetado
  a `metodo_transferible` cuando el solape es solo de método.
- **"Misma facultad no significa mayor pertinencia"**: no existe `w_facultad`.
- **"Más resultados no significa mejor solución"**: MMR (Maximal Marginal Relevance)
  obligatorio, no opcional, para controlar redundancia entre resultados casi idénticos.

**Etapa B — propagación por el grafo** hacia investigadores y grupos (por aristas
reales: `researcher_project.csv`, `thesis_advisor.csv`, `publication_researcher.csv`,
con peso por rol — PI pesa más que co-investigador), currículo (puente explícito por
programa compartido + semántica) y capacidades institucionales (semántico puro,
**siempre etiquetado como inferido**, porque el dataset no tiene arista explícita
capacidad↔proyecto).

El score de un investigador se reporta **descompuesto**:
`0.58 = 0.34 por autoría real (evidencia recuperada) + 0.24 por afinidad de perfil (inferida)`.

La escala de relevancia (0-1) se traduce a banda Alta/Media/Baja con umbrales
**calibrados empíricamente** corriendo las 42 necesidades reales (ver
`src/motor.py::_banda` y `docs/casos_demostrables.md`).

## 7. Evidencia y trazabilidad

Toda `Conexion` trae evidencias con procedencia resoluble: `Data V1.0 / archivo.csv /
ID / campo`, apuntando a un registro que existe de verdad en el CSV original (no un
resumen ni un texto generado). Verificado en `tests/test_motor.py` y medido en
`scripts/evaluar.py` (trazabilidad = 100% sobre las 42 necesidades).

## 8. Oportunidades

[`src/oportunidades.py`](src/oportunidades.py) ensambla 5 tipos a partir del ranking:
`RESEARCH_CONTINUITY`, `COLLABORATION`, `CURRICULAR_INTEGRATION`,
`CAPABILITY_ACTIVATION`, `THESIS_OPPORTUNITY`. Cada una cita entidades reales y su
evidencia; ninguna se inventa sin al menos un hecho verificable detrás.

## 9. Métricas (proxy, no oficiales)

Data V1.0 pública **no trae Gold Standard** (Guía Oficial §4.2), así que
[`src/evaluacion.py`](src/evaluacion.py) construye un **silver standard autoderivado**:
para cada necesidad, los proyectos/tesis que comparten ≥2 términos de la faceta TEMA
(vocabulario controlado extraído del propio dataset) actúan como referencia débil.
Se declara así explícitamente — nunca como verdad oficial — y solo cubre 20 de las 42
necesidades (las que tienen plantilla léxica explícita; ver hallazgo en la sección 10).

```bash
python scripts/evaluar.py
```

Resultado de referencia (ver salida completa del script):

| Métrica | Valor |
|---|---|
| Precision@10 (20 necesidades con referencia léxica) | ~0.96 |
| NDCG@10 | ~0.97 |
| Cobertura de evidencia (las 42 necesidades) | 100% |
| Trazabilidad (las 42 necesidades) | 100% |

**Aviso metodológico honesto**: el silver standard se construye con el mismo tipo de
señal (solape léxico de facetas) que el motor puede usar, así que las variantes
basadas en facetas puntúan más alto contra él casi por definición — es un sesgo
circular conocido, declarado en la propia salida de `evaluar.py`, no escondido. El
valor real de la capa semántica **no se mide con este número**: se demuestra en las
22 necesidades sin plantilla léxica (NEED-021..042) y en consultas libres nuevas del
evaluador — ver el caso "deserción estudiantil" en `docs/casos_demostrables.md`, que
es literalmente el ejemplo que la Guía Oficial usa para explicar por qué un buscador
de palabras clave no basta.

## 10. Hallazgos del dataset que condicionaron el diseño

- Las 42 necesidades institucionales usan **dos plantillas distintas**: 20
  (NEED-001..020) enumeran términos temáticos explícitos; 22 (NEED-021..042) no,
  y verificamos que comparten casi cero vocabulario léxico con proyectos/tesis. El
  motor semántico existe precisamente para no fallar en ese segundo grupo.
- `research_lines.csv` fragmenta su campo `keywords` palabra por palabra
  (`'predicción;de;permanencia;estudiantil'`) en vez de frase por frase como el resto
  de archivos — se reconstruye antes de indexar (`src/ingesta.py`,
  `config.CAMPOS_FRAGMENTADOS_POR_PALABRA`).
- `institutional_capabilities.csv` tiene 96 filas que colapsan a solo **12 capacidades
  reales** (nombre+facultad), cada una repetida ~8 veces por tipo de recurso. Se
  deduplica en la capa de oportunidades para no ofrecer la misma capacidad tres veces.
- `source_catalog.csv` es intencionalmente ruidoso: solo cubre 5 de 21 archivos, con
  hasta 7 entradas contradictorias (HIGH/MEDIUM) por archivo. Se resuelve por moda
  estadística, documentado en `src/ingesta.py::_cargar_confiabilidad`.

## 11. Limitaciones conocidas

- El "silver standard" es un proxy autoderivado, no una verdad oficial — se declara
  así en todo momento (ver sección 9).
- La articulación curricular usa el puente explícito `program_id` cuando existe; sin
  eso, depende de similitud semántica pura y puede ser menos precisa.
- El emparejamiento de capacidades es siempre inferido (no hay arista explícita en el
  dataset entre capacidad y proyecto/necesidad).
- Los umbrales de banda (Alta/Media/Baja) están calibrados sobre Data V1.0 RC2
  específicamente; no se garantiza que generalicen a otro dataset sin recalibrar.

## 12. Declaración de componentes externos

- **Modelo de embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (Apache-2.0), ejecutado **localmente** vía `fastembed`/ONNX Runtime. Se usa
  únicamente para vectorizar texto en 384 dimensiones; no genera contenido ni toma
  decisiones de puntuación por sí solo (el score es una combinación explícita y
  auditable de varias señales, ver sección 6).
- **Ningún servicio en la nube, API de pago ni modelo generativo** se usa en el camino
  crítico. La capa narrativa opcional (Gemini/Groq, `src/explicacion.py`) está
  **desactivada por defecto** y, si se activa, todo texto que produzca se marca en la
  UI como `IA generada`, distinto de la evidencia institucional.
- No se usan datasets externos ni se inventan hechos sobre entidades de Data V1.0.

## 13. Estructura del repositorio

```
src/
  config.py         rutas, pesos del score, mapeos estáticos del dataset
  trazabilidad.py   Procedencia, Evidencia, DesgloseScore
  ingesta.py        carga y normalización de los 21 CSV + 60 documentos .md
  facetas.py        vocabulario controlado TEMA / MÉTODO / DOMINIO
  embeddings.py     motor semántico local + TF-IDF, con caché en disco
  grafo.py          construcción del grafo de conocimiento (NetworkX)
  motor.py          recuperación, scoring híbrido, MMR, propagación
  oportunidades.py  ensamblado de oportunidades tipadas
  explicacion.py    explicación determinista + narrador LLM opcional
  evaluacion.py     métricas proxy y ablación
  app.py            dashboard Dash
scripts/
  construir_indice.py   pipeline offline idempotente
  evaluar.py            reporte de métricas
tests/              45 tests (pytest) cubriendo ingesta, grafo, facetas,
                    embeddings, motor y oportunidades/explicación
docs/
  arquitectura.md          diagrama de lo realmente implementado
  casos_demostrables.md    3 casos end-to-end con evidencia
```

## 14. Tests

```bash
python -m pytest tests/ -q
```
