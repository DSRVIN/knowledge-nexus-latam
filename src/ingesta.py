"""Carga de las 3 capas de Data V1.0 (21 CSV + 60 documentos .md) hacia un
repositorio en memoria de Entidades trazables.

Todos los CSV se leen con encoding utf-8-sig: los 21 archivos traen BOM y
sin esto la primera columna de cada tabla se corrompe silenciosamente
(ej. "\\ufefffaculty_id" en vez de "faculty_id").
"""
from __future__ import annotations

import collections
import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import config
from src.trazabilidad import Evidencia, Procedencia

# (carpeta, archivo, tipo, columna_id)
_ENTIDADES_CONFIG = [
    (config.DIR_INSTITUTION, "faculties.csv", "FACULTY", "faculty_id"),
    (config.DIR_INSTITUTION, "programs.csv", "PROGRAM", "program_id"),
    (config.DIR_INSTITUTION, "research_groups.csv", "GROUP", "group_id"),
    (config.DIR_INSTITUTION, "research_lines.csv", "LINE", "line_id"),
    (config.DIR_INSTITUTION, "institutional_capabilities.csv", "CAPABILITY", "capability_id"),
    (config.DIR_PEOPLE_CURRICULUM, "researchers.csv", "RESEARCHER", "researcher_id"),
    (config.DIR_PEOPLE_CURRICULUM, "researcher_expertise.csv", "EXPERTISE", "expertise_id"),
    (config.DIR_PEOPLE_CURRICULUM, "subjects.csv", "SUBJECT", "subject_id"),
    (config.DIR_PEOPLE_CURRICULUM, "competencies.csv", "COMPETENCY", "competency_id"),
    (config.DIR_PEOPLE_CURRICULUM, "learning_outcomes.csv", "OUTCOME", "outcome_id"),
    (config.DIR_KNOWLEDGE_NEEDS, "institutional_needs.csv", "NEED", "need_id"),
    (config.DIR_KNOWLEDGE_NEEDS, "projects.csv", "PROJECT", "project_id"),
    (config.DIR_KNOWLEDGE_NEEDS, "theses.csv", "THESIS", "thesis_id"),
    (config.DIR_KNOWLEDGE_NEEDS, "publications.csv", "PUBLICATION", "publication_id"),
]

# (carpeta, archivo, nombre de la relación)
_RELACIONES_CONFIG = [
    (config.DIR_PEOPLE_CURRICULUM, "researcher_group.csv", "researcher_group"),
    (config.DIR_KNOWLEDGE_NEEDS, "researcher_project.csv", "researcher_project"),
    (config.DIR_KNOWLEDGE_NEEDS, "project_group.csv", "project_group"),
    (config.DIR_KNOWLEDGE_NEEDS, "thesis_advisor.csv", "thesis_advisor"),
    (config.DIR_KNOWLEDGE_NEEDS, "publication_project.csv", "publication_project"),
    (config.DIR_KNOWLEDGE_NEEDS, "publication_researcher.csv", "publication_researcher"),
]


@dataclass
class Entidad:
    id: str
    tipo: str
    campos: dict = field(default_factory=dict)
    campos_multivalor: dict = field(default_factory=dict)
    archivo_documento_md: str | None = None
    texto_documento_md: str | None = None

    def valor(self, campo: str) -> str:
        return (self.campos.get(campo) or "").strip()

    def valores_multivalor(self, campo: str) -> list[str]:
        return self.campos_multivalor.get(campo, [])

    def procedencia(self, campo: str) -> Procedencia:
        if campo == "documento_markdown":
            if not self.archivo_documento_md:
                raise ValueError(f"{self.id} no tiene documento markdown asociado")
            return Procedencia(
                "03_knowledge_needs/documents", self.archivo_documento_md, self.id, "contenido"
            )
        capa, archivo = config.ARCHIVO_POR_TIPO[self.tipo]
        return Procedencia(capa, archivo, self.id, campo)

    def evidencia(self, campo: str) -> Evidencia:
        texto = self.texto_documento_md if campo == "documento_markdown" else self.valor(campo)
        return Evidencia(procedencia=self.procedencia(campo), texto=texto, tipo="recuperada")

    def texto_indexable(self) -> str:
        """Concatenación ponderada de los campos de mayor valor semántico
        (Documento Técnico §3). Los campos temáticos se repiten para que
        pesen más en la vectorización (TF-IDF y embeddings)."""
        campos = config.CAMPOS_SEMANTICOS_POR_TIPO.get(self.tipo, list(self.campos.keys()))
        partes = []
        for campo in campos:
            if campo in self.campos_multivalor:
                valores = self.valores_multivalor(campo)
                if valores:
                    partes.append(" ".join(valores))
            else:
                valor = self.valor(campo)
                if valor:
                    partes.append(valor)
        return " . ".join(partes)


class RepositorioInstitucional:
    """Punto único de acceso a Data V1.0: entidades, relaciones explícitas
    y confiabilidad declarada por fuente."""

    def __init__(self) -> None:
        self.entidades: dict[str, Entidad] = {}
        self.tablas_relacion: dict[str, list[dict]] = {}
        self.confiabilidad_por_archivo: dict[str, str] = {}

    def cargar(self) -> "RepositorioInstitucional":
        self._cargar_confiabilidad()
        for carpeta, archivo, tipo, col_id in _ENTIDADES_CONFIG:
            self._cargar_tabla_entidad(carpeta / archivo, tipo, col_id)
        for carpeta, archivo, nombre in _RELACIONES_CONFIG:
            self.tablas_relacion[nombre] = self._leer_csv(carpeta / archivo)
        self._vincular_documentos_markdown()
        return self

    @staticmethod
    def _leer_csv(ruta: Path) -> list[dict]:
        with open(ruta, encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))

    def _cargar_tabla_entidad(self, ruta: Path, tipo: str, col_id: str) -> None:
        filas = self._leer_csv(ruta)
        multivalor_campos = config.CAMPOS_MULTIVALOR_POR_TIPO.get(tipo, [])
        fragmentados_campos = config.CAMPOS_FRAGMENTADOS_POR_PALABRA.get(tipo, [])
        for fila in filas:
            eid = fila[col_id]
            fila = dict(fila)
            for campo in fragmentados_campos:
                # ver nota en config.CAMPOS_FRAGMENTADOS_POR_PALABRA
                fila[campo] = (fila.get(campo, "") or "").replace(";", " ")
            campos_mv = {}
            for campo in multivalor_campos:
                valor = fila.get(campo, "") or ""
                campos_mv[campo] = [t.strip() for t in valor.split(";") if t.strip()]
            self.entidades[eid] = Entidad(
                id=eid, tipo=tipo, campos=fila, campos_multivalor=campos_mv
            )

    def _cargar_confiabilidad(self) -> None:
        """source_catalog.csv es intencionalmente ruidoso: solo cubre 5 de
        los 21 archivos y trae hasta 7 entradas contradictorias (HIGH/MEDIUM)
        por archivo. Se resuelve por moda estadística; el resto de archivos
        queda sin catalogar (peso neutro documentado en config.py)."""
        filas = self._leer_csv(config.DIR_INSTITUTION / "source_catalog.csv")
        por_archivo = collections.defaultdict(list)
        for fila in filas:
            por_archivo[fila["file_name"]].append(fila["reliability_level"])
        for archivo, niveles in por_archivo.items():
            nivel, _ = collections.Counter(niveles).most_common(1)[0]
            self.confiabilidad_por_archivo[archivo] = nivel

    def nivel_confiabilidad(self, tipo: str) -> str:
        _, archivo = config.ARCHIVO_POR_TIPO[tipo]
        return self.confiabilidad_por_archivo.get(archivo, config.NIVEL_CONFIABILIDAD_NO_CATALOGADO)

    def _vincular_documentos_markdown(self) -> None:
        filas = self._leer_csv(config.DIR_KNOWLEDGE_NEEDS / "document_catalog.csv")
        for fila in filas:
            entidad = self.entidades.get(fila["entity_id"])
            if entidad is None:
                continue
            ruta_md = config.DIR_DOCUMENTS / fila["file_name"]
            entidad.archivo_documento_md = fila["file_name"]
            entidad.texto_documento_md = ruta_md.read_text(encoding="utf-8")

    def por_tipo(self, tipo: str) -> list[Entidad]:
        return [e for e in self.entidades.values() if e.tipo == tipo]

    def guardar_procesado(self) -> None:
        tipos = {e.tipo for e in self.entidades.values()}
        for tipo in tipos:
            filas = [e.campos for e in self.por_tipo(tipo)]
            pd.DataFrame(filas).to_parquet(
                config.DIR_PROCESSED / f"{tipo.lower()}.parquet", index=False
            )
        for nombre, filas in self.tablas_relacion.items():
            pd.DataFrame(filas).to_parquet(
                config.DIR_PROCESSED / f"rel_{nombre}.parquet", index=False
            )
