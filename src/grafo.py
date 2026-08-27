"""Construcción del grafo de conocimiento institucional sobre NetworkX.

Todas las aristas construidas aquí son EXPLÍCITAS: nacen de un identificador
o de una tabla de relación real del dataset, nunca de una inferencia. Cada
arista lleva su propia Procedencia para que sea auditable igual que un nodo.
Las aristas inferidas (semánticas, propagadas) se añaden en tiempo de
consulta por motor.py, con naturaleza="inferida", y nunca se mezclan aquí.

networkx>=3.0 retiró write_gpickle/read_gpickle: la persistencia se hace con
pickle estándar sobre el objeto MultiDiGraph.
"""
from __future__ import annotations

import pickle

import networkx as nx

from src import config
from src.ingesta import Entidad, RepositorioInstitucional
from src.trazabilidad import Procedencia

# (tipo_origen, campo_fk, tipo_destino, etiqueta_relacion)
# Aristas que nacen de una columna de clave foránea en la propia tabla del origen.
_ARISTAS_POR_FK = [
    ("PROGRAM", "faculty_id", "FACULTY", "pertenece_a_facultad"),
    ("SUBJECT", "program_id", "PROGRAM", "pertenece_a_programa"),
    ("COMPETENCY", "program_id", "PROGRAM", "asociada_a_programa"),
    ("COMPETENCY", "subject_id", "SUBJECT", "asociada_a_asignatura"),
    ("OUTCOME", "subject_id", "SUBJECT", "resultado_de_asignatura"),
    ("LINE", "group_id", "GROUP", "linea_de_grupo"),
    ("GROUP", "faculty_id", "FACULTY", "grupo_de_facultad"),
    ("CAPABILITY", "responsible_unit", "FACULTY", "capacidad_de_facultad"),
    ("RESEARCHER", "faculty_id", "FACULTY", "investigador_de_facultad"),
    ("RESEARCHER", "primary_program_id", "PROGRAM", "investigador_de_programa"),
    ("EXPERTISE", "researcher_id", "RESEARCHER", "expertise_de_investigador"),
    ("PROJECT", "faculty_id", "FACULTY", "proyecto_de_facultad"),
    ("PROJECT", "program_id", "PROGRAM", "proyecto_de_programa"),
    ("PROJECT", "group_id", "GROUP", "proyecto_ejecutado_por_grupo"),
    ("THESIS", "program_id", "PROGRAM", "tesis_de_programa"),
    ("PUBLICATION", "related_project_id", "PROJECT", "publicacion_deriva_de_proyecto"),
]

# (nombre_tabla_relacion, columna_origen, columna_destino, columna_rol, etiqueta_relacion)
_ARISTAS_POR_TABLA = [
    ("researcher_group", "researcher_id", "group_id", "role", "miembro_de_grupo"),
    ("researcher_project", "researcher_id", "project_id", "role", "participa_en_proyecto"),
    ("thesis_advisor", "researcher_id", "thesis_id", "role", "dirige_tesis"),
    ("project_group", "project_id", "group_id", "relation", "proyecto_grupo_ejecutor"),
    ("publication_project", "publication_id", "project_id", "relation", "publicacion_de_proyecto"),
    ("publication_researcher", "researcher_id", "publication_id", "role", "autor_de_publicacion"),
]


class GrafoConocimiento:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    def construir(self, repo: RepositorioInstitucional) -> "GrafoConocimiento":
        for entidad in repo.entidades.values():
            self._agregar_nodo(entidad)
        for tipo_origen, campo_fk, tipo_destino, etiqueta in _ARISTAS_POR_FK:
            self._agregar_aristas_fk(repo, tipo_origen, campo_fk, tipo_destino, etiqueta)
        for nombre_tabla, col_origen, col_destino, col_rol, etiqueta in _ARISTAS_POR_TABLA:
            self._agregar_aristas_tabla(
                repo, nombre_tabla, col_origen, col_destino, col_rol, etiqueta
            )
        return self

    def _agregar_nodo(self, entidad: Entidad) -> None:
        nombre = _nombre_entidad(entidad)
        self.g.add_node(entidad.id, tipo=entidad.tipo, nombre=nombre)

    def _agregar_aristas_fk(
        self, repo: RepositorioInstitucional, tipo_origen: str, campo_fk: str, tipo_destino: str, etiqueta: str
    ) -> None:
        for entidad in repo.por_tipo(tipo_origen):
            destino_id = entidad.valor(campo_fk)
            if not destino_id or destino_id not in repo.entidades:
                continue  # campo vacío: la info no está disponible, no se inventa la arista
            proc = entidad.procedencia(campo_fk)
            self.g.add_edge(
                entidad.id,
                destino_id,
                tipo_relacion=etiqueta,
                naturaleza="explicita",
                procedencia=proc.to_dict(),
            )

    def _agregar_aristas_tabla(
        self,
        repo: RepositorioInstitucional,
        nombre_tabla: str,
        col_origen: str,
        col_destino: str,
        col_rol: str,
        etiqueta: str,
    ) -> None:
        capa, archivo = "?", f"{nombre_tabla}.csv"
        # las tablas de relación viven en las capas B o C según el archivo
        capa = (
            "02_people_curriculum"
            if nombre_tabla in {"researcher_group"}
            else "03_knowledge_needs"
        )
        for i, fila in enumerate(repo.tablas_relacion[nombre_tabla]):
            origen_id, destino_id = fila[col_origen], fila[col_destino]
            if origen_id not in repo.entidades or destino_id not in repo.entidades:
                continue
            rol = fila.get(col_rol, "")
            registro_id = f"{origen_id}|{destino_id}"
            proc = Procedencia(capa, archivo, registro_id, col_rol)
            self.g.add_edge(
                origen_id,
                destino_id,
                tipo_relacion=etiqueta,
                naturaleza="explicita",
                rol=rol,
                procedencia=proc.to_dict(),
            )

    def guardar(self, ruta=None) -> None:
        ruta = ruta or config.RUTA_GRAFO
        with open(ruta, "wb") as fh:
            pickle.dump(self.g, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def cargar(cls, ruta=None) -> "GrafoConocimiento":
        ruta = ruta or config.RUTA_GRAFO
        instancia = cls()
        with open(ruta, "rb") as fh:
            instancia.g = pickle.load(fh)
        return instancia


def _nombre_entidad(entidad: Entidad) -> str:
    for campo in ("full_name", "title", "group_name", "program_name", "faculty_name",
                  "line_name", "capability_name", "subject_name"):
        valor = entidad.valor(campo)
        if valor:
            return valor
    return entidad.id
