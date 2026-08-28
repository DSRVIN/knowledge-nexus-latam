# Knowledge Nexus LATAM — Contexto completo del proyecto

Este documento es el resumen de todo lo que se construyó en esta sesión de trabajo:
qué problema resuelve, qué se implementó, cómo funciona cada pieza por dentro, y la
historia completa de decisiones técnicas (incluyendo las que cambiaron sobre la marcha
por restricciones reales que fuimos descubriendo).

No reemplaza al [README.md](../README.md) (el entregable técnico formal para el
hackathon) ni a [arquitectura.md](arquitectura.md) o [casos_demostrables.md](casos_demostrables.md)
— es el documento para que **vos** entiendas el proyecto de punta a punta, en lenguaje
más narrativo.

---

## 1. El problema que había que resolver

Es un hackathon (**Knowledge Nexus LATAM**, organizado por Talento TECH, nivel
avanzado). El escenario: una universidad ficticia latinoamericana (**UNEXA**) tiene
muchísima información — facultades, investigadores, proyectos, tesis, publicaciones,
necesidades institucionales, currículo — pero **fragmentada**. Nadie tiene una vista
integrada. El reto pide construir un sistema que:

1. Tome una **necesidad institucional** (o una consulta libre nueva) como entrada.
2. **Descubra** qué proyectos, tesis, investigadores, grupos, capacidades y asignaturas
   se conectan con esa necesidad — **sin depender solo de que compartan las mismas
   palabras** (ese es el punto central del reto: un buscador de keywords no alcanza).
3. **Priorice** esas conexiones con algún criterio explicable, no solo un score mágico.
4. **Explique** por qué cada conexión es relevante y **muestre la evidencia exacta**
   (archivo, registro, campo) que la sustenta.
5. **Genere oportunidades** institucionales concretas a partir de esas conexiones
   (nuevas colaboraciones, continuidad de investigación, integración curricular, etc.).
6. Todo esto con una interfaz donde se pueda demostrar en vivo, ante un evaluador,
   con una consulta que él mismo escriba en el momento (no ejemplos precargados).

La evaluación oficial vale 100 puntos y mira, sobre todo: arquitectura, funcionamiento
real del prototipo, innovación, calidad de las conexiones, explicabilidad y
trazabilidad. **No** valora tecnología por sofisticación — valora que las conexiones
sean pertinentes y se puedan justificar con evidencia real.

## 2. Qué construimos — de un vistazo

Un sistema Python completo, de punta a punta:

```
Data V1.0 (21 CSV + 60 documentos .md, dataset oficial del reto)
   ↓
Ingesta y normalización
   ↓
Grafo de conocimiento (quién se conecta con quién, de forma explícita/real)
   +
Vectorización semántica y léxica (para descubrir conexiones que no comparten palabras)
   +
Vocabulario de facetas (TEMA / MÉTODO / DOMINIO, extraído del propio dataset)
   ↓
Motor de scoring híbrido → conexiones priorizadas, con score descompuesto y explicado
   ↓
Propagación por el grafo → de proyectos/tesis a investigadores, grupos, currículo, capacidades
   ↓
Generador de oportunidades (5 tipos: continuidad, colaboración, integración curricular,
                              activación de capacidades, oportunidad de tesis)
   ↓
Dashboard interactivo (Dash) donde se prueba todo esto en vivo
```

Todo corre **sin ninguna API de pago, sin claves, sin internet en tiempo real** —
una decisión deliberada para que nunca falle por depender de un tercero durante la
evaluación (justamente uno de los riesgos que la guía del hackathon advierte
explícitamente: "¿qué pasa si una API externa falla?").

Está desplegado y funcionando en vivo en:
**https://knowledge-nexus-latam.onrender.com**

Y el código está en: **https://github.com/DSRVIN/knowledge-nexus-latam**

---

## 3. Cómo funciona por dentro, módulo por módulo

### 3.1 Los datos: Data V1.0

El dataset oficial (`KNOWLEDGE_NEXUS_LATAM_DATA_V1_RC2_PARTICIPANTS/`) trae 21 archivos
CSV organizados en 3 capas:

- **01_institution**: facultades, programas, grupos de investigación, líneas de
  investigación, capacidades institucionales.
- **02_people_curriculum**: investigadores, su expertise, asignaturas, competencias,
  resultados de aprendizaje.
- **03_knowledge_needs**: necesidades institucionales (42), proyectos (320), tesis
  (650), publicaciones (360), y las tablas que los relacionan entre sí (quién es autor
  de qué, qué grupo ejecuta qué proyecto, etc.), más 60 documentos `.md` complementarios.

Antes de escribir una sola línea de código, exploramos este dataset a fondo y
encontramos varias cosas que **cambiaron el diseño**:

**Hallazgo 1 — el dataset tiene una trampa incorporada.** Las 42 necesidades
institucionales describen su problema con una plantilla de texto. Para 20 de ellas
(NEED-001 a NEED-020), esa plantilla lista explícitamente los términos temáticos del
problema ("el problema se expresa en términos de X, Y, Z"). Medimos: esos términos
coinciden **100%** con las keywords de los proyectos y tesis. Es decir, ese salto
(necesidad → proyecto) se puede resolver con un simple `if palabra in texto`. Si nos
quedábamos ahí, estaríamos construyendo exactamente el "buscador de keywords" que la
guía del reto penaliza.

**Hallazgo 2 — ahí es donde está el reto de verdad.** El salto siguiente (necesidad →
investigador, necesidad → asignatura) es mucho más difícil: de esos mismos términos,
solo 6 aparecen en los intereses de investigación de los investigadores, y solo 2 en
los temas de las asignaturas. Ahí es donde un motor puramente léxico se queda corto y
hace falta entender el **significado**, no solo la palabra.

**Hallazgo 3 — las otras 22 necesidades (NEED-021 a NEED-042) ni siquiera tienen esa
plantilla de términos.** Son necesidades más "de gobernanza institucional" (ej.
"fortalecimiento de semilleros interdisciplinarios") sin vocabulario explícito. Son las
más difíciles del dataset y las usamos como prueba de fuego para el motor semántico.

**Hallazgo 4 — el grafo de relaciones explícitas está completo.** Los 320 proyectos
tienen investigador principal y grupo ejecutor: los 650 trabajos de grado tienen
director; hay 746 vínculos investigador-proyecto y 720 publicación-investigador. Esto
significa que podemos **propagar relevancia por relaciones reales**, no solo adivinar
con similitud de texto.

**Hallazgos "sucios" del dataset** (cosas que si no las manejábamos bien, rompían todo
en silencio):
- Los 21 CSV traen BOM (marca de encoding invisible) — sin leerlos con
  `utf-8-sig`, la primera columna de cada tabla sale corrupta.
- El archivo de líneas de investigación separa sus keywords **palabra por palabra**
  con `;` (ej. `predicción;de;permanencia;estudiantil`), no frase por frase como el
  resto de archivos — había que reconstruir la frase antes de usarla.
- El catálogo de confiabilidad de fuentes (`source_catalog.csv`) es intencionalmente
  ruidoso: solo cubre 5 de los 21 archivos, y trae hasta 7 entradas contradictorias
  (algunas dicen "alta confiabilidad", otras "media") para el mismo archivo.
- Las 96 filas de "capacidades institucionales" en realidad son solo **12 capacidades
  reales**, cada una repetida 8 veces (una por tipo de recurso: infraestructura,
  metodológico, computacional, etc.) — si no se detecta esto, el sistema termina
  recomendando "la misma capacidad" tres veces como si fueran tres cosas distintas.

### 3.2 Ingesta y trazabilidad (`src/ingesta.py`, `src/trazabilidad.py`)

Esta es la base de todo: cada dato que el sistema usa tiene que poder decir de dónde
salió. Por eso lo primero que se construyó fue el concepto de **Procedencia**: un
objeto que sabe decir "esto viene de `projects.csv`, fila `PRJ-001`, campo
`methodology`". Cualquier afirmación que el sistema haga después (una conexión, una
oportunidad) tiene que traer una o más de estas procedencias colgando. Así se cumple
el requisito de trazabilidad end-to-end del reto.

`ingesta.py` lee los 21 CSV y los 60 documentos `.md`, corrige los problemas del punto
anterior (BOM, keywords fragmentadas, confiabilidad ruidosa resuelta por moda
estadística), y arma un diccionario de **Entidades** en memoria — cada una con su tipo
(proyecto, investigador, tesis, etc.), sus campos, y la capacidad de generar su propia
evidencia citable.

### 3.3 Facetas (`src/facetas.py`)

Para poder comparar "de qué habla" una necesidad contra "de qué habla" un proyecto sin
depender de coincidencia literal de palabras, se construyó un vocabulario controlado en
tres dimensiones, **extraído automáticamente del propio dataset** (nunca escrito a
mano):

- **TEMA**: de qué trata (ej. "permanencia estudiantil", "calidad del agua").
- **MÉTODO**: cómo se aborda (ej. "clasificación supervisada", "series temporales") —
  son 16 valores cerrados que ya existen en el campo `methodological_expertise` de los
  investigadores.
- **DOMINIO**: en qué campo de aplicación (ej. "salud", "educación", "ambiente").

Un mismo término puede pertenecer a más de una faceta (por ejemplo, "series
temporales" es a la vez un tema, un dominio y un método, según en qué campo del
dataset aparezca) — eso es intencional y refleja la realidad, no se fuerza a que cada
palabra tenga un solo significado.

### 3.4 Grafo de conocimiento (`src/grafo.py`)

Se construyó un grafo (con la librería `networkx`) donde cada entidad del dataset es
un nodo, y cada relación **que existe de verdad en una tabla del dataset** es una
arista explícita — con su propia procedencia. Por ejemplo: "investigador X dirige tesis
Y" es una arista que viene literalmente de `thesis_advisor.csv`. El grafo tiene 3.232
nodos y 6.909 aristas explícitas.

Esto es clave: **nunca se mezclan relaciones reales con relaciones inferidas**. Todo lo
que el sistema "adivina" por similitud de texto se marca aparte, como inferido, y se
muestra distinto en la interfaz (línea sólida vs. punteada en el grafo visual).

### 3.5 Motor semántico — embeddings (`src/embeddings.py`)

Esta es la pieza que resuelve el Hallazgo 2 de arriba (encontrar conexiones que no
comparten palabras). La idea: convertir cada texto en un vector de números (un
"embedding") tal que textos con significado parecido queden cerca en ese espacio,
aunque usen palabras distintas. Así, "deserción estudiantil" y "permanencia
estudiantil" — que no comparten ni una palabra clave del vocabulario controlado —
terminan siendo vectores parecidos.

Se calculan **dos** espacios de vectores en paralelo, no uno solo:
- **Semántico**: usando un modelo de lenguaje (ver la sección 6 para la historia
  completa de qué modelo se usó y por qué cambió).
- **Léxico** (TF-IDF): la señal clásica de coincidencia de palabras, en formato
  disperso (`scipy.sparse`) por razones de memoria que se explican más abajo.

Ambas señales se combinan en el score final — no se elige una sobre la otra.

### 3.6 El motor de scoring híbrido (`src/motor.py`) — el corazón del sistema

Este es el módulo más importante. Funciona en dos etapas:

**Etapa A — sobre proyectos, tesis, publicaciones y líneas de investigación.**
Cada candidato recibe un score que es la **suma explícita** de varios componentes:

```
score = peso_semántico · similitud_semántica
      + peso_léxico    · similitud_léxica (TF-IDF)
      + peso_tema      · cuánto solapan los TEMAS
      + peso_dominio   · cuánto solapan los DOMINIOS
      + peso_método    · cuánto solapan los MÉTODOS
      + peso_evidencia · qué tan completa/confiable es la fuente
      (con una penalización si el único solape es de método, sin tema en común)
```

Ese desglose se guarda entero — no se pierde, y la interfaz lo muestra como un
gráfico de barras por cada resultado. Esto es lo que permite responder "¿por qué A
quedó antes que B?": se comparan los dos desgloses componente por componente.

Hay tres reglas explícitas que vienen literalmente del documento técnico del reto y
que se implementaron a propósito:
- **"Mismo método no es mismo problema"**: si dos textos solo coinciden en el método
  (por ejemplo, ambos usan "clasificación supervisada") pero no en el tema, el score
  se penaliza y se etiqueta distinto ("método transferible", no "antecedente
  relevante").
- **"Misma facultad no es más pertinencia"**: no existe ningún bono en el score por
  pertenecer a la misma facultad.
- **"Más resultados no es mejor solución"**: se aplica un algoritmo de diversidad
  (MMR — Maximal Marginal Relevance) para que el top-10 no sean diez proyectos casi
  idénticos, sino resultados que realmente cubran ángulos distintos.

**Etapa B — propagación hacia investigadores, grupos, currículo y capacidades.**
Una vez que la Etapa A encontró los proyectos/tesis relevantes, el motor **camina por
el grafo real** para encontrar quién los hizo: si el proyecto X es relevante y el
investigador Y es su director (dato real de `thesis_advisor.csv`), entonces Y hereda
relevancia de X. Esa relevancia heredada se pondera por el rol (un investigador
principal pesa más que un co-investigador) y se sigue documentando cada componente:

> *"María López tiene score 0.66 = 0.46 por autoría real de dos proyectos + 0.20 por
> afinidad semántica de su perfil (esto último es inferido, no viene de una tabla)."*

Para el **currículo** (asignaturas), se usa un puente real cuando existe (la tesis y
la asignatura pertenecen al mismo programa académico) combinado con semántica. Para
las **capacidades institucionales**, no existe ninguna relación explícita en el
dataset hacia proyectos o necesidades — así que esa conexión siempre se marca como
inferida, sin fingir una certeza que no hay.

### 3.7 Oportunidades (`src/oportunidades.py`)

A partir de todo lo anterior, se ensamblan 5 tipos de oportunidades institucionales
concretas, cada una con su propia evidencia:

1. **Continuidad investigativa**: una tesis relevante + su director + su grupo activo.
2. **Colaboración interdisciplinaria**: dos investigadores de facultades distintas que
   llegan a la misma necesidad por caminos de evidencia independientes.
3. **Integración curricular**: o bien una "brecha" (el programa produce antecedentes
   pero ninguna asignatura cubre el tema) o una oportunidad positiva (ya hay una
   asignatura alineada, se sugiere usar el proyecto como caso de estudio).
4. **Activación de capacidades**: una capacidad institucional madura y activa,
   relacionada semánticamente (deduplicando las 96 filas a las 12 capacidades reales
   mencionadas en el Hallazgo del dataset).
5. **Oportunidad de tesis**: cuando el mejor antecedente disponible es viejo, se
   sugiere que hay lugar para un trabajo de grado que lo actualice.

### 3.8 Explicación (`src/explicacion.py`)

Cada conexión y oportunidad se traduce a una explicación en español, generada por
**plantilla determinista** (no por IA generativa) que menciona: qué facetas se
comparten, qué componentes del score pesaron más, si la relación es explícita o
inferida, y cita la evidencia exacta. Esta capa **funciona sin red, sin clave, siempre
igual** — es la explicación "de verdad" que se le muestra al evaluador.

Existe además un conector opcional (**desactivado por defecto**) para que un modelo de
lenguaje (Gemini/Groq) redacte esa misma explicación en prosa más natural — pero nunca
inventa datos ni reemplaza la explicación determinista; si se activara, todo texto que
produzca se marcaría en la interfaz como "IA generada", distinto de la evidencia real.

### 3.9 Métricas (`src/evaluacion.py`)

El dataset oficial **no trae "respuestas correctas"** (a propósito, es parte del
diseño del reto). Así que se construyó una referencia propia ("silver standard")
usando el mismo criterio de solape de facetas TEMA, y con eso se calculan métricas de
ranking (Precision@10, NDCG@10, etc.) — pero **siempre declarando explícitamente que
es una referencia autoderivada, no una verdad oficial**, para no aparentar más certeza
de la que hay. También se mide trazabilidad (100%) y cobertura de evidencia (100%)
sobre las 42 necesidades.

### 3.10 El dashboard (`src/app.py`)

Construido con Dash (Flask + React por debajo). Tiene:
- Una caja de texto libre + un selector de las 42 necesidades del catálogo.
- Sliders para los pesos del score, que **re-rankean en vivo** al moverlos — la mejor
  prueba de que nada está precargado.
- Pestaña **Conexiones**: resultados con su explicación, gráfico de desglose del
  score, y evidencia citada.
- Pestaña **Grafo**: visualización de la red de conexiones alrededor de la consulta.
- Pestaña **Oportunidades**: las oportunidades generadas.
- Pestaña **"¿Por qué A antes que B?"**: comparador directo entre dos resultados,
  componente a componente.

---

## 4. La historia del despliegue (y por qué cambiaron cosas sobre la marcha)

Esto es importante porque el proyecto terminó siendo distinto de la propuesta inicial
en dos puntos grandes, y vale la pena que sepas por qué:

**Cambio 1 — de Gemini API a embeddings locales.** El prompt original que trajiste
proponía usar la API de Google Gemini para los embeddings. Antes de escribir código,
verificamos con datos reales (no solo el modelo hackathon "gratis" en el papel) que un
modelo de embeddings corriendo **localmente** (sin clave, sin cuota, sin depender de
que Google esté arriba durante la evaluación) era más seguro para un hackathon donde
justo se penaliza depender de una API externa que puede fallar.

**Cambio 2 — de Vercel a Render.** Pediste desplegar en Vercel. Vercel es una
plataforma de funciones "serverless" (sin servidor persistente, con límites de tiempo
de arranque muy cortos) — no está pensada para una app con un modelo de IA cargado en
memoria. Antes de intentarlo y fallar, medimos: el modelo + las dependencias ya
superaban el límite de tamaño de función de Vercel. Recomendé alternativas con
servidor persistente real.

**Cambio 3 — de Hugging Face Spaces a Render.** El plan era usar Hugging Face Spaces
(gratis, pensado justo para este tipo de apps). Al intentar crearlo, Hugging Face
había cambiado su política de precios en julio de 2026 (después de mi fecha de
conocimiento): los Spaces con Docker ahora piden suscripción PRO. Pivoteamos a
**Render**, que sí tiene un plan Docker gratuito real.

**Cambio 4 — el más grande: el modelo de embeddings tuvo que cambiar de multilingüe a
español.** Este fue el hallazgo más caro de la sesión. Al desplegar por primera vez en
Render, la primera búsqueda real hacía caer el servidor (error 503, "memoria
insuficiente"). Investigamos a fondo y medimos, paso por paso, dónde se iba la
memoria:

- El modelo multilingüe original (`paraphrase-multilingual-MiniLM-L12-v2`, pensado
  para cubrir ~50 idiomas) usa un **tokenizador de 250.000 palabras/subpalabras**.
  Medimos: cargar *solo ese tokenizador*, sin el modelo, ya cuesta ~260MB de RAM.
- Sumado al modelo en sí (~250MB) y a la base de Python/Flask/Dash/pandas
  (~150-190MB), el total rondaba **650-700MB** — por encima del límite de 512MB del
  plan gratuito de Render.
- La solución: como el dataset es 100% en español, no hace falta un modelo que cubra
  50 idiomas. Buscamos y exportamos nosotros mismos a un formato optimizado (ONNX,
  cuantizado a 8 bits) un modelo específico de español
  (`hiiamsid/sentence_similarity_spanish_es`, basado en BETO), cuyo vocabulario es de
  solo ~31.000 palabras. Resultado medido: **~190MB en total**, con margen holgado
  bajo el límite.
- De paso, también se optimizó la matriz de similitud léxica (TF-IDF) para guardarse
  en formato disperso en vez de denso — otro ahorro de ~108MB que siempre estaban
  ocupados en memoria sin necesidad.

El trade-off honesto de este último cambio: el modelo en español discrimina con un
poco menos de margen entre temas relacionados y no relacionados que el modelo
multilingüe original (medido: separación de ~0.40 vs ~0.78 entre un par de textos
relacionados y uno no relacionado). Esto está documentado explícitamente en el README
y en los casos demostrables — el motor híbrido compensa esa pérdida con las señales
léxica y de facetas, que no dependen de esta limitación.

## 5. Estado actual

- **App en vivo**: https://knowledge-nexus-latam.onrender.com (plan gratuito — se
  duerme tras 15 min sin uso, tarda ~30-60s en despertar en la siguiente visita, es
  normal del plan free, no un error).
- **Código**: https://github.com/DSRVIN/knowledge-nexus-latam (repositorio público).
- **46 tests automatizados** (pytest) cubriendo ingesta, grafo, facetas, embeddings,
  motor y oportunidades — todos verdes.
- El modelo de embeddings en español (~111MB, formato ONNX) está commiteado en el
  repositorio vía Git LFS — no se descarga nada en tiempo de ejecución.

## 6. Limitaciones que hay que poder explicar si un evaluador pregunta

- El "silver standard" de las métricas es una referencia que nosotros mismos
  derivamos del dataset, no una verdad oficial (el reto no entrega una).
- El emparejamiento con capacidades institucionales siempre es inferido: el dataset
  no tiene ninguna tabla que conecte capacidades con proyectos o necesidades
  directamente.
- Los umbrales de "banda" (Alta/Media/Baja) están calibrados específicamente sobre
  este dataset — no está garantizado que sirvan igual con otro dataset sin recalibrar.
- El modelo semántico en español tiene menos margen de discriminación que el
  multilingüe original evaluado al principio — trade-off consciente para caber en
  hosting gratuito, documentado y no escondido.
