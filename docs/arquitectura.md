# Arquitectura

Diagrama de **lo realmente implementado** (no una arquitectura aspiracional).
Corresponde 1:1 a los módulos en `src/`.

```mermaid
flowchart TD
    subgraph Fuentes["Fuentes — Data V1.0 RC2 (intacta)"]
        CSV["21 archivos CSV\n(3 capas: institution / people_curriculum / knowledge_needs)"]
        MD["60 documentos .md\n(project_profile / thesis_summary / institutional_need)"]
    end

    subgraph Procesamiento["Procesamiento offline — scripts/construir_indice.py"]
        ING["ingesta.py\nRepositorioInstitucional\n(BOM-safe, normaliza multivalor,\nresuelve confiabilidad por moda)"]
        FAC["facetas.py\nVocabularioFacetas\nTEMA / MÉTODO / DOMINIO\n(extraído del propio dataset)"]
        GRA["grafo.py\nGrafoConocimiento\nNetworkX MultiDiGraph\n(aristas EXPLÍCITAS con procedencia)"]
        EMB["embeddings.py\nIndiceEmbeddings\nfastembed local (ONNX) + TF-IDF\n(dos señales independientes, caché por hash)"]
    end

    subgraph Artefactos["data/artifacts — persistidos, no en el dataset original"]
        ART1[["grafo.gpickle"]]
        ART2[["embeddings.npz + tfidf.joblib"]]
        ART3[["vocabulario_facetas.json"]]
    end

    subgraph Motor["src/motor.py — MotorConexiones"]
        ETA["Etapa A\nrecuperación híbrida sobre\nPROJECT/THESIS/PUBLICATION/LINE\nscore descompuesto + MMR"]
        ETB["Etapa B\npropagación por grafo (investigadores, grupos)\npuente explícito (currículo)\nsemántico puro, siempre inferido (capacidades)"]
    end

    subgraph Razonamiento["Razonamiento y salida"]
        OPO["oportunidades.py\nGeneradorOportunidades\n5 tipos tipados"]
        EXP["explicacion.py\nexplicación determinista\n(+ narrador LLM opcional, desactivado)"]
        EVA["evaluacion.py\nmétricas proxy + ablación"]
    end

    UI["app.py — Dash\nConexiones / Grafo / Oportunidades / Comparador"]

    CSV --> ING
    MD --> ING
    ING --> FAC
    ING --> GRA
    ING --> EMB
    FAC --> ART3
    GRA --> ART1
    EMB --> ART2

    ART1 --> ETB
    ART2 --> ETA
    ART3 --> ETA
    ART3 --> ETB

    ETA --> ETB
    ETA --> OPO
    ETB --> OPO
    ETA --> EXP
    ETB --> EXP
    ETA --> EVA

    OPO --> UI
    EXP --> UI
    EVA --> UI
```

## Flujo de una consulta en vivo (no offline)

```mermaid
sequenceDiagram
    actor Evaluador
    participant UI as app.py (Dash)
    participant M as MotorConexiones
    participant I as IndiceEmbeddings
    participant G as GrafoConocimiento
    participant O as GeneradorOportunidades

    Evaluador->>UI: escribe consulta libre o elige necesidad
    UI->>M: buscar(texto, pesos, top_k)
    M->>I: vectorizar_consulta(texto)  [semántico + léxico]
    M->>M: extraer_facetas(texto)      [TEMA/MÉTODO/DOMINIO]
    M->>M: score por candidato + MMR   [Etapa A]
    M->>G: propagar por aristas reales [Etapa B: investigadores, grupos]
    M->>M: puente programa + semántica [Etapa B: currículo]
    M->>M: emparejamiento semántico    [Etapa B: capacidades, siempre inferido]
    M-->>UI: Conexion[] con desglose + evidencia + procedencia
    UI->>O: generar(resultados, prioridad_necesidad)
    O-->>UI: Oportunidad[] con evidencia
    UI-->>Evaluador: Conexiones / Grafo / Oportunidades / Comparador
```

## Por qué estas decisiones

- **Grafo persistido aparte del dataset**: Data V1.0 se conserva intacta
  (requisito §9 del Documento Técnico); todo lo derivado vive en `data/`.
- **Dos señales de vectorización simultáneas** (semántica local + léxica TF-IDF),
  no una con fallback silencioso a la otra: cada una aporta una parte del score
  visible, y si el motor semántico no está disponible en la máquina, su peso se
  redistribuye explícitamente sobre el léxico (`MotorConexiones._pesos_efectivos`),
  documentado y probado (`tests/test_motor.py::test_degradacion_controlada_sin_motor_semantico`).
- **Aristas explícitas vs. inferidas nunca se mezclan**: cada arista del grafo trae
  `naturaleza` y `procedencia`; el motor solo añade aristas inferidas en tiempo de
  consulta, nunca las persiste como si fueran parte del dataset original.
- **UI**: el grafo interactivo se implementó primero con `dash-cytoscape`, pero en
  las pruebas de este entorno el widget no compositaba nada en el canvas (verificado
  con un ejemplo mínimo oficial de la librería, cero relación con la lógica de la
  app). Se sustituyó por un grafo de red con Plotly + NetworkX (`spring_layout`),
  que ya es dependencia de Dash y renderiza de forma confirmada (SVG, no canvas).
