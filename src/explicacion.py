"""Explicación determinista de una Conexion: por qué existe, con qué peso
contribuyó cada señal y qué la sustenta. Es la capa PRINCIPAL, no un
respaldo — funciona sin red, sin clave y sin no-determinismo, y es lo que
se demuestra ante el evaluador.

Además expone un narrador LLM opcional (desactivado por defecto,
LLM_HABILITADO=false) que solo redacta en prosa lo que esta capa ya calculó;
nunca puntúa, nunca recupera, nunca introduce entidades nuevas. Todo texto
que produzca se marca explícitamente como generado, nunca como evidencia.
"""
from __future__ import annotations

from typing import Protocol

from src import config
from src.ingesta import Entidad, RepositorioInstitucional
from src.motor import Conexion

_NOMBRES_TIPO = {
    "FACULTY": "facultad", "PROGRAM": "programa", "GROUP": "grupo", "LINE": "línea de investigación",
    "CAPABILITY": "capacidad", "RESEARCHER": "investigador", "EXPERTISE": "expertise",
    "SUBJECT": "asignatura", "COMPETENCY": "competencia", "OUTCOME": "resultado de aprendizaje",
    "NEED": "necesidad", "PROJECT": "proyecto", "THESIS": "tesis", "PUBLICATION": "publicación",
}

_ETIQUETAS_COMPONENTE = {
    "semantico": "similitud semántica de contenido",
    "lexico": "coincidencia léxica (TF-IDF)",
    "tema": "solape de tema",
    "dominio": "solape de dominio de aplicación",
    "metodo": "solape de método",
    "evidencia": "completitud y confiabilidad de la fuente",
    "evidencia_directa": "evidencia directa de autoría/dirección",
    "afinidad_perfil_inferida": "afinidad semántica de perfil (inferida)",
    "puente_programa_compartido": "programa académico compartido",
    "afinidad_semantica_inferida": "afinidad semántica (inferida)",
    "afinidad_semantica": "afinidad semántica",
    "solape_dominio": "solape de dominio de aplicación",
}


def _nombre(entidad: Entidad) -> str:
    for campo in ("full_name", "title", "group_name", "program_name", "faculty_name",
                  "line_name", "capability_name", "subject_name"):
        valor = entidad.valor(campo)
        if valor:
            return valor
    return entidad.id


def _frase_componentes(conexion: Conexion) -> str:
    positivos = sorted(
        ((k, v) for k, v in conexion.desglose.componentes.items() if v > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if not positivos:
        return ""
    principales = positivos[:2]
    partes = [f"{_ETIQUETAS_COMPONENTE.get(k, k)} ({v:.2f})" for k, v in principales]
    frase = "Los factores que más pesaron: " + ", ".join(partes) + "."
    if conexion.desglose.penalizaciones:
        frase += " Se aplicó una penalización porque el único solape encontrado fue de método, sin tema en común."
    return frase


def _frase_facetas(conexion: Conexion) -> str:
    compartidas = conexion.facetas_compartidas or {}
    partes = []
    if compartidas.get("tema"):
        partes.append("tema: " + ", ".join(sorted(compartidas["tema"])[:3]))
    if compartidas.get("dominio"):
        partes.append("dominio: " + ", ".join(sorted(compartidas["dominio"])[:3]))
    if compartidas.get("metodo"):
        partes.append("método: " + ", ".join(sorted(compartidas["metodo"])[:3]))
    if not partes:
        return ""
    return "Términos del vocabulario controlado que comparten: " + "; ".join(partes) + "."


def _frase_naturaleza(conexion: Conexion) -> str:
    if conexion.naturaleza == "explicita":
        return "La relación consta explícitamente en una tabla de Data V1.0 (no es una inferencia)."
    if conexion.naturaleza == "mixta":
        return "Parte de la relación es explícita (tabla del dataset) y parte es inferida (similitud semántica)."
    return "La relación es inferida: no existe una arista explícita para este tipo de vínculo en Data V1.0."


def _frase_evidencia(conexion: Conexion) -> str:
    if not conexion.evidencias:
        return "No se encontró evidencia textual asociada."
    citas = [e.procedencia.cita() for e in conexion.evidencias[:2]]
    return "Evidencia: " + " | ".join(citas) + "."


def explicar(conexion: Conexion, repo: RepositorioInstitucional) -> str:
    entidad = repo.entidades[conexion.destino_id]
    tipo_legible = _NOMBRES_TIPO.get(entidad.tipo, entidad.tipo.lower())
    nombre = _nombre(entidad)

    if conexion.tipo_relacion == "metodo_transferible":
        apertura = (
            f"{nombre} ({tipo_legible} {entidad.id}) comparte únicamente el método con la consulta, no el tema: "
            f"se clasifica como método transferible, no como antecedente directo (banda {conexion.banda})."
        )
    elif conexion.tipo_relacion == "antecedente_relevante":
        apertura = (
            f"{nombre} ({tipo_legible} {entidad.id}) se identifica como antecedente relevante "
            f"(score {conexion.relevancia:.2f}, banda {conexion.banda})."
        )
    elif conexion.tipo_relacion == "propagada_por_autoria":
        apertura = (
            f"{nombre} ({tipo_legible} {entidad.id}) se conecta a la consulta por autoría o dirección real "
            f"de las fuentes recuperadas (score {conexion.relevancia:.2f}, banda {conexion.banda})."
        )
    elif conexion.tipo_relacion == "articulacion_curricular":
        apertura = (
            f"{nombre} ({tipo_legible} {entidad.id}) se propone como punto de articulación curricular "
            f"(score {conexion.relevancia:.2f}, banda {conexion.banda})."
        )
    elif conexion.tipo_relacion == "capacidad_activable":
        apertura = (
            f"{nombre} ({tipo_legible} {entidad.id}) es una capacidad institucional activable para esta "
            f"necesidad (score {conexion.relevancia:.2f}, banda {conexion.banda})."
        )
    else:
        apertura = f"{nombre} ({tipo_legible} {entidad.id}) obtiene score {conexion.relevancia:.2f} (banda {conexion.banda})."

    partes = [apertura, _frase_facetas(conexion), _frase_componentes(conexion), _frase_naturaleza(conexion), _frase_evidencia(conexion)]
    return " ".join(p for p in partes if p)


def enriquecer_con_explicacion(conexiones: list[Conexion], repo: RepositorioInstitucional) -> list[Conexion]:
    for c in conexiones:
        c.explicacion = explicar(c, repo)
    return conexiones


# --------------------------------------------------------------------- #
# Narrador LLM opcional — desactivado por defecto (LLM_HABILITADO=false).
# Nunca sustituye a la explicación determinista; solo la redacta en prosa
# si se activa explícitamente. Ver requirements-opcional.txt.
# --------------------------------------------------------------------- #
class NarradorLLM(Protocol):
    def narrar(self, explicacion_determinista: str) -> str | None: ...


class NarradorDesactivado:
    def narrar(self, explicacion_determinista: str) -> str | None:
        return None


def obtener_narrador() -> NarradorLLM:
    if not config.LLM_HABILITADO or not config.LLM_API_KEY:
        return NarradorDesactivado()
    try:
        from google import genai  # noqa: F401  (requiere requirements-opcional.txt)
    except ImportError:
        return NarradorDesactivado()

    class NarradorGemini:
        def __init__(self) -> None:
            self._cliente = genai.Client(api_key=config.LLM_API_KEY)

        def narrar(self, explicacion_determinista: str) -> str | None:
            try:
                respuesta = self._cliente.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=(
                        "Redacta en un párrafo breve y natural, en español, la siguiente explicación "
                        "técnica. No agregues datos, entidades ni cifras que no estén ya presentes:\n\n"
                        f"{explicacion_determinista}"
                    ),
                )
                return respuesta.text
            except Exception:
                return None  # degradación silenciosa: la determinista sigue disponible

    return NarradorGemini()
