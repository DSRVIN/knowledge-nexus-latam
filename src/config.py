"""Configuración central: rutas, pesos del motor de scoring y parámetros del modelo.

Todos los pesos viven aquí para que el desglose de score en la UI y en los
tests referencien la misma fuente única de verdad.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent

DATA_ROOT = RAIZ_PROYECTO / os.environ.get(
    "DATA_ROOT", "KNOWLEDGE_NEXUS_LATAM_DATA_V1_RC2_PARTICIPANTS"
)
DIR_INSTITUTION = DATA_ROOT / "01_institution"
DIR_PEOPLE_CURRICULUM = DATA_ROOT / "02_people_curriculum"
DIR_KNOWLEDGE_NEEDS = DATA_ROOT / "03_knowledge_needs"
DIR_DOCUMENTS = DIR_KNOWLEDGE_NEEDS / "documents"

DIR_PROCESSED = RAIZ_PROYECTO / "data" / "processed"
DIR_ARTIFACTS = RAIZ_PROYECTO / "data" / "artifacts"
DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
DIR_ARTIFACTS.mkdir(parents=True, exist_ok=True)

RUTA_GRAFO = DIR_ARTIFACTS / "grafo.gpickle"
RUTA_EMBEDDINGS = DIR_ARTIFACTS / "embeddings.npz"
RUTA_MANIFIESTO_EMBEDDINGS = DIR_ARTIFACTS / "manifiesto_embeddings.json"
RUTA_TFIDF = DIR_ARTIFACTS / "tfidf.joblib"
RUTA_TFIDF_MATRIZ = DIR_ARTIFACTS / "tfidf_matriz.npz"
RUTA_VOCABULARIO_FACETAS = DIR_ARTIFACTS / "vocabulario_facetas.json"

# --- Modelo de embeddings (100% local, sin claves ni cuotas ni red en runtime) ---
# hiiamsid/sentence_similarity_spanish_es (BETO fine-tuned para similitud
# semántica en español), exportado a ONNX y cuantizado a int8. Vocabulario de
# 31k tokens (monolingüe) en vez de los 250k del modelo multilingüe original:
# ver la nota de diseño al inicio de embeddings.py.
EMBED_MODEL = "hiiamsid/sentence_similarity_spanish_es (ONNX int8, local)"
DIR_MODELO_ESPANOL = RAIZ_PROYECTO / "data" / "modelo_espanol"
RUTA_MODELO_ESPANOL_ONNX = DIR_MODELO_ESPANOL / "model_quantized.onnx"
RUTA_MODELO_ESPANOL_TOKENIZER = DIR_MODELO_ESPANOL / "tokenizer.json"

# --- Capa narrativa opcional (desactivada por defecto) ---
LLM_HABILITADO = os.environ.get("LLM_HABILITADO", "false").strip().lower() == "true"
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

# --- Conteos esperados según los manifiestos oficiales del dataset ---
# Usados por tests/test_ingesta.py para detectar corrupción silenciosa en la carga.
CONTEOS_ESPERADOS = {
    "faculties": 6,
    "programs": 18,
    "research_groups": 24,
    "research_lines": 60,
    "institutional_capabilities": 96,
    "researchers": 180,
    "researcher_expertise": 720,
    "subjects": 126,
    "competencies": 252,
    "learning_outcomes": 378,
    "researcher_group": 180,
    "institutional_needs": 42,
    "projects": 320,
    "theses": 650,
    "publications": 360,
    "researcher_project": 746,
    "project_group": 320,
    "thesis_advisor": 650,
    "publication_researcher": 720,
    "publication_project": 200,
}

# --- Pesos del score híbrido (Etapa A: entidades documentales) ---
# w_facultad no existe deliberadamente: "misma facultad no significa mayor
# pertinencia" (Documento Técnico §16).
PESOS_SCORE = {
    "w_sem": 0.35,   # similitud coseno de embeddings locales
    "w_lex": 0.15,   # similitud TF-IDF (señal léxica independiente)
    "w_tema": 0.25,  # solape de faceta TEMA
    "w_dominio": 0.10,
    "w_metodo": 0.10,
    "w_evidencia": 0.05,  # completitud de campos x confiabilidad x recencia
}
PENALIZACION_METODO_SIN_TEMA = 0.5  # factor multiplicativo, no aditivo

# --- Propagación en grafo (Etapa B) ---
AMORTIGUACION_PROPAGACION = 0.6
PESO_ROL_INVESTIGADOR = {"PI": 1.0, "CO_INVESTIGATOR": 0.6, "ADVISOR": 1.0, "AUTHOR": 0.8}
PESO_ROL_GRUPO = {"LEADER": 1.0, "MEMBER": 0.6}
PESO_AFINIDAD_PERFIL = 0.4  # peso de la similitud semántica directa perfil-consulta en Etapa B

# --- Diversidad (MMR) ---
MMR_LAMBDA = 0.7  # 1.0 = solo relevancia, 0.0 = solo diversidad

# --- Confiabilidad de fuentes (source_catalog.csv) ---
# El catálogo es intencionalmente ruidoso: solo cubre 5 de 21 archivos y
# trae niveles mixtos por archivo. Se resuelve por moda estadística; los
# archivos no catalogados reciben un nivel neutro documentado.
NIVEL_CONFIABILIDAD_NO_CATALOGADO = "NO_CATALOGADO"
PESO_CONFIABILIDAD = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4, "NO_CATALOGADO": 0.6}

# --- Campos de mayor valor semántico por tipo de entidad (Documento Técnico §3) ---
# Usado para construir el texto_indexable (embeddings.py) y para extraer
# facetas (facetas.py) de forma consistente en todo el sistema.
CAMPOS_SEMANTICOS_POR_TIPO = {
    "FACULTY": ["faculty_name", "description", "strategic_focus"],
    "PROGRAM": ["program_name", "description", "disciplinary_area", "graduate_profile", "strategic_topics"],
    "GROUP": ["group_name", "description", "mission", "main_area"],
    "LINE": ["line_name", "description", "keywords"],
    "CAPABILITY": ["capability_name", "capability_type", "description", "available_resources", "application_domains"],
    "RESEARCHER": ["profile_summary", "research_interests", "methodological_expertise", "application_domains"],
    "EXPERTISE": ["expertise_name", "expertise_type", "evidence_source"],
    "SUBJECT": ["subject_name", "description", "purpose", "main_topics", "disciplinary_area"],
    "COMPETENCY": ["competency_type", "description"],
    "OUTCOME": ["outcome_description", "cognitive_level", "evidence_type"],
    "NEED": ["title", "description", "context", "expected_impact"],
    "PROJECT": ["title", "problem_statement", "abstract", "general_objective", "methodology", "expected_results", "application_context", "keywords"],
    "THESIS": ["title", "abstract", "problem_statement", "general_objective", "methodology", "main_results", "conclusions", "application_context", "keywords"],
    "PUBLICATION": ["title", "abstract", "keywords"],
}

# --- Campos multivalor (separados por ';' en el CSV origen, un término por token) ---
CAMPOS_MULTIVALOR_POR_TIPO = {
    "FACULTY": ["strategic_focus"],
    "PROGRAM": ["strategic_topics"],
    "CAPABILITY": ["available_resources", "application_domains"],
    "RESEARCHER": ["research_interests", "methodological_expertise", "application_domains"],
    "SUBJECT": ["main_topics"],
    "PROJECT": ["keywords"],
    "THESIS": ["keywords"],
    "PUBLICATION": ["keywords", "researchers"],
}

# --- Anomalía verificada en Data V1.0: research_lines.keywords no separa
# frases (como el resto de campos "keywords") sino PALABRAS SUELTAS de una
# misma frase (ej. 'predicción;de;permanencia;estudiantil'). Confirmado en
# las 60 filas del archivo. Tratarlo como multivalor introduciría términos
# basura de una sola palabra ("de", "y", "en") en el vocabulario de facetas.
# Se reconstruye la frase completa uniendo con espacios antes de indexar.
CAMPOS_FRAGMENTADOS_POR_PALABRA = {"LINE": ["keywords"]}

# --- Ubicación física de cada tipo de entidad (capa, archivo) ---
# Fuente única de verdad para reconstruir Procedencia sin duplicar el dato
# en cada Entidad cargada.
ARCHIVO_POR_TIPO = {
    "FACULTY": ("01_institution", "faculties.csv"),
    "PROGRAM": ("01_institution", "programs.csv"),
    "GROUP": ("01_institution", "research_groups.csv"),
    "LINE": ("01_institution", "research_lines.csv"),
    "CAPABILITY": ("01_institution", "institutional_capabilities.csv"),
    "RESEARCHER": ("02_people_curriculum", "researchers.csv"),
    "EXPERTISE": ("02_people_curriculum", "researcher_expertise.csv"),
    "SUBJECT": ("02_people_curriculum", "subjects.csv"),
    "COMPETENCY": ("02_people_curriculum", "competencies.csv"),
    "OUTCOME": ("02_people_curriculum", "learning_outcomes.csv"),
    "NEED": ("03_knowledge_needs", "institutional_needs.csv"),
    "PROJECT": ("03_knowledge_needs", "projects.csv"),
    "THESIS": ("03_knowledge_needs", "theses.csv"),
    "PUBLICATION": ("03_knowledge_needs", "publications.csv"),
}

# Campo(s) de cada tipo que aportan término(s) de FACETA MÉTODO (frente a TEMA/DOMINIO)
CAMPOS_FACETA_METODO = {"methodological_expertise", "methodology"}
CAMPOS_FACETA_DOMINIO = {"application_domains", "application_context", "disciplinary_area"}
# Todo lo demás (keywords, main_topics, research_interests, títulos/descr.) alimenta TEMA.

TOP_K_DEFECTO = 10
