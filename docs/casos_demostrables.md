# Casos demostrables

Tres casos reales, ejecutados contra el sistema en funcionamiento (no precargados).
Cada uno sigue la cadena exigida por el Documento Técnico §5:
**Necesidad -> antecedentes -> investigadores/grupos -> capacidades -> currículo ->
oportunidad -> relevancia -> evidencia**.

---

## Caso 1 — Consulta libre nueva: "deserción estudiantil"

Este caso reproduce **literalmente el ejemplo de la Guía Oficial §2**: la palabra
"deserción" es deliberadamente ajena al vocabulario controlado del dataset (las
necesidades relacionadas hablan de *permanencia estudiantil*, *riesgo académico* y
*trayectorias educativas*, nunca de "deserción"). Un sistema que solo buscara esa
palabra no encontraría nada. Verificado en `tests/test_facetas.py::test_extraer_es_identico_para_entidad_y_consulta_libre`.

**Entrada** (texto libre, tal cual escribiría un evaluador): `"deserción estudiantil"`

**Salida real del motor** (top-3, tipos proyecto/tesis/línea/publicación):

> Predicción de permanencia estudiantil (línea de investigación LIN-033) se identifica
> como antecedente relevante (score 0.30, banda Media). Los factores que más pesaron:
> similitud semántica de contenido (0.23), completitud y confiabilidad de la fuente
> (0.05). La relación es inferida: no existe una arista explícita para este tipo de
> vínculo en Data V1.0.
> **Evidencia**: `Data V1.0 / research_lines.csv / LIN-033 / line_name` |
> `Data V1.0 / research_lines.csv / LIN-033 / description`

> Estrategia basada en clasificación supervisada para estudiar *student attrition*
> (proyecto PRJ-004), score 0.27, banda Media.
> **Evidencia**: `Data V1.0 / projects.csv / PRJ-004 / problem_statement`

> Resultados aplicados en bienestar universitario (publicación PUB-279), score 0.26.
> **Evidencia**: `Data V1.0 / publications.csv / PUB-279 / abstract`

**Por qué importa**: la señal léxica pura (TF-IDF) no habría encontrado ninguno de
estos tres resultados (comparten cero palabras literales con "deserción"). Es el
motor semántico local (`fastembed`, embeddings de 384 dimensiones) el que reconoce
que *permanencia estudiantil* y *student attrition* son la misma idea. Esto es
justo el punto ciego que la propia Guía Oficial usa para explicar por qué un
"buscador por palabras clave" no basta (§2, §12: "generar recomendaciones con IA sin
evidencia rastreable" y "presentar coincidencias sin distinguir relevancia" son los
errores que este caso evita).

---

## Caso 2 — Necesidad de dominio limpio: NEED-005 "Monitoreo remoto cardiovascular"

Muestra la cadena completa **necesidad -> antecedentes -> investigadores reales -> evidencia**,
y cómo el sistema evita forzar interdisciplinariedad donde no la hay.

**Antecedentes** (Etapa A, banda Alta, scores 0.49-0.67 sobre 8 candidatos).

**Propagación a investigadores** (Etapa B, por aristas reales del grafo):

> Miguel Rodríguez Rojas (investigador INV-071) se conecta a la consulta por autoría
> o dirección real de las fuentes recuperadas (**score 0.67, banda Alta**). Los
> factores que más pesaron: **evidencia directa de autoría/dirección (0.49)**,
> afinidad semántica de perfil, inferida (0.17).
> **Evidencia**: `Data V1.0 / thesis_advisor.csv / INV-071|THS-081 / role` |
> `Data V1.0 / researcher_project.csv / INV-071|PRJ-033 / role`

Los 8 investigadores propagados pertenecen todos a **FAC-002** (Ciencias de la
Salud) — el sistema no inventa una colaboración interdisciplinaria aquí porque
genuinamente no la hay en la evidencia: es un antecedente de dominio único, bien
sustentado, con score descompuesto y explícito en su mayor parte (autoría real de
`thesis_advisor.csv` / `researcher_project.csv`), no solo inferencia semántica.

**Oportunidad generada**: `RESEARCH_CONTINUITY` a partir de la tesis dirigida por
Miguel Rodríguez Rojas, con su grupo activo asociado.

---

## Caso 3 — Necesidad institucional (sin plantilla léxica) con colaboración real: NEED-037 "Fortalecimiento de semilleros interdisciplinarios"

NEED-037 pertenece al grupo de 22 necesidades (NEED-021..042) cuya descripción **no**
enumera términos temáticos explícitos (a diferencia de NEED-001..020). Verificado:
comparten casi cero vocabulario léxico con `projects.keywords`/`theses.keywords`. Es
uno de los casos más difíciles del dataset — y el más honesto para demostrar que el
sistema no infla su confianza cuando la señal es débil.

**Antecedentes**: banda **Media** en todos los resultados (scores 0.28-0.29, muy por
debajo del 0.45+ típico de necesidades con plantilla léxica limpia — ver calibración
de bandas en `src/motor.py::_banda`). El sistema no finge certeza que no tiene.

**Propagación a investigadores** — aquí sí emerge interdisciplinariedad real:

| Investigador | Facultad | Score |
|---|---|---|
| Alejandro Salazar Martínez | FAC-006 | 0.41 |
| David Pérez Vargas | FAC-004 | 0.35 |
| Ana Gómez López | FAC-001 | 0.32 |

**Oportunidad `COLLABORATION` generada**:

> Alejandro Salazar Martínez (FAC-006) y David Pérez Vargas (FAC-004) llegan a esta
> necesidad por evidencia independiente y pertenecen a facultades distintas: su
> combinación amplía el abordaje más allá de un solo dominio. (Prioridad: Media)

Se generan 3 oportunidades de este tipo (una por cada par de las 3 facultades
representadas). Todas quedan etiquetadas con prioridad **Media**, consistente con la
confianza real del antecedente — no se les asigna "Alta" solo porque cruzan
facultades, porque la fórmula de prioridad (`src/oportunidades.py::_prioridad`)
pondera explícitamente la relevancia del antecedente, no solo el tipo de oportunidad.

**Patrón general observado** (verificado corriendo las 42 necesidades): las
oportunidades `COLLABORATION` emergen casi exclusivamente en las 22 necesidades
institucionales/de gobernanza, prácticamente nunca en las 20 necesidades de dominio
limpio — porque estas últimas tienen un cluster de expertise claro en una sola
facultad, y el sistema no fuerza una narrativa interdisciplinaria donde la evidencia
no la sostiene.

---

## Resumen de lo que estos tres casos demuestran juntos

| Capacidad exigida (Guía Oficial §5) | Dónde se ve |
|---|---|
| Descubrir sin depender de coincidencia literal | Caso 1 |
| Priorizar y explicar por qué A antes que B | Los tres (ver pestaña "¿Por qué A antes que B?" de la UI) |
| Trazar hasta archivo/registro/campo | Los tres |
| No sobre-afirmar cuando la evidencia es débil | Caso 3 |
| Generar oportunidades reales, no genéricas | Los tres |
| Explotar aristas explícitas cuando existen | Caso 2 |
