"""Ensamblado de oportunidades tipadas (Documento Técnico §7) a partir del
ranking del motor. Cada oportunidad combina varias entidades y siempre debe
poder explicar su valor institucional con evidencia trazable — nunca se
inventa una oportunidad sin al menos una entidad real detrás.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.ingesta import RepositorioInstitucional
from src.motor import Conexion, MotorConexiones
from src.trazabilidad import Evidencia, Procedencia

TIPOS_OPORTUNIDAD = (
    "RESEARCH_CONTINUITY",
    "COLLABORATION",
    "CURRICULAR_INTEGRATION",
    "CAPABILITY_ACTIVATION",
    "THESIS_OPPORTUNITY",
)


@dataclass
class Oportunidad:
    tipo: str
    titulo: str
    entidades_relacionadas: list[str]
    razon: str
    prioridad: str  # "Alta" | "Media" | "Baja"
    evidencias: list[Evidencia] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "opportunity": self.titulo,
            "type": self.tipo,
            "related_entities": self.entidades_relacionadas,
            "reason": self.razon,
            "priority": self.prioridad,
            "evidence": [e.to_dict() for e in self.evidencias],
        }


_PATRON_SUFIJO_CAPACIDAD = re.compile(r"\s*—\s*capacidad\s*\d+\s*$")


def _nombre_base_capacidad(capacidad) -> str:
    """Data V1.0 modela cada capacidad institucional (96 filas) como ~8 filas
    por tipo de recurso (INFRASTRUCTURE/METHODOLOGICAL/COMPUTATIONAL/...) que
    comparten nombre base y facultad, solo variando el sufijo '— capacidad NN'
    y el capability_type. Verificado: las 96 filas colapsan a 12 capacidades
    reales distintas (nombre base, facultad)."""
    base = _PATRON_SUFIJO_CAPACIDAD.sub("", capacidad.valor("capability_name"))
    return f"{base}|{capacidad.valor('responsible_unit')}"


def _prioridad(prioridad_necesidad: str, relevancias: list[float]) -> str:
    promedio = sum(relevancias) / len(relevancias) if relevancias else 0.0
    if prioridad_necesidad == "HIGH" and promedio >= 0.45:
        return "Alta"
    if prioridad_necesidad in ("HIGH", "MEDIUM") and promedio >= 0.25:
        return "Media"
    if promedio >= 0.45:
        return "Media"
    return "Baja"


class GeneradorOportunidades:
    def __init__(self, repo: RepositorioInstitucional, motor: MotorConexiones) -> None:
        self.repo = repo
        self.motor = motor

    def generar(
        self,
        prioridad_necesidad: str,
        resultados_a: list[Conexion],
        investigadores: list[Conexion],
        capacidades: list[Conexion],
        asignaturas: list[Conexion],
    ) -> list[Oportunidad]:
        oportunidades: list[Oportunidad] = []
        oportunidades += self._continuidad_investigativa(prioridad_necesidad, resultados_a)
        oportunidades += self._colaboracion(prioridad_necesidad, investigadores)
        oportunidades += self._integracion_curricular(prioridad_necesidad, resultados_a, asignaturas)
        oportunidades += self._activacion_capacidades(prioridad_necesidad, capacidades)
        oportunidades += self._oportunidad_tesis(prioridad_necesidad, resultados_a)
        orden = {"Alta": 2, "Media": 1, "Baja": 0}
        oportunidades.sort(key=lambda o: orden[o.prioridad], reverse=True)
        return oportunidades

    # ------------------------------------------------------------------ #
    def _continuidad_investigativa(self, prioridad_necesidad: str, resultados_a: list[Conexion]) -> list[Oportunidad]:
        g = self.motor.grafo.g
        oportunidades = []
        for c in [r for r in resultados_a if r.tipo_destino == "THESIS"][:3]:
            tesis = self.repo.entidades[c.destino_id]
            for u, _, datos in g.in_edges(c.destino_id, data=True):
                if datos["tipo_relacion"] != "dirige_tesis":
                    continue
                investigador = self.repo.entidades.get(u)
                if investigador is None:
                    continue
                grupos_activos = [
                    self.repo.entidades[v]
                    for _, v, d in g.out_edges(u, data=True)
                    if d["tipo_relacion"] == "miembro_de_grupo" and self.repo.entidades[v].valor("status") == "ACTIVE"
                ]
                if not grupos_activos:
                    continue
                grupo = grupos_activos[0]
                oportunidades.append(
                    Oportunidad(
                        tipo="RESEARCH_CONTINUITY",
                        titulo=f"Continuidad investigativa a partir de {tesis.id}",
                        entidades_relacionadas=[tesis.id, investigador.id, grupo.id],
                        razon=(
                            f"{tesis.valor('title')} es antecedente relevante (score {c.relevancia:.2f}). "
                            f"Su director, {investigador.valor('full_name')}, pertenece al grupo activo "
                            f"{grupo.valor('group_name')}, lo que habilita dar continuidad a esta línea."
                        ),
                        prioridad=_prioridad(prioridad_necesidad, [c.relevancia]),
                        evidencias=[
                            tesis.evidencia("abstract"),
                            Evidencia(
                                procedencia=Procedencia.from_dict(datos["procedencia"]),
                                texto=f"ADVISOR: {investigador.valor('full_name')} dirige {tesis.id}",
                            ),
                            grupo.evidencia("status"),
                        ],
                    )
                )
        return oportunidades

    def _colaboracion(self, prioridad_necesidad: str, investigadores: list[Conexion]) -> list[Oportunidad]:
        por_facultad: dict[str, Conexion] = {}
        for c in investigadores:
            fac = self.repo.entidades[c.destino_id].valor("faculty_id")
            if fac and fac not in por_facultad:
                por_facultad[fac] = c  # ya vienen ordenados por relevancia
        facultades = list(por_facultad.items())
        oportunidades = []
        for i in range(len(facultades)):
            for j in range(i + 1, len(facultades)):
                fac_a, c_a = facultades[i]
                fac_b, c_b = facultades[j]
                inv_a = self.repo.entidades[c_a.destino_id]
                inv_b = self.repo.entidades[c_b.destino_id]
                oportunidades.append(
                    Oportunidad(
                        tipo="COLLABORATION",
                        titulo=f"Colaboración interdisciplinaria: {fac_a} + {fac_b}",
                        entidades_relacionadas=[inv_a.id, inv_b.id],
                        razon=(
                            f"{inv_a.valor('full_name')} ({fac_a}) y {inv_b.valor('full_name')} ({fac_b}) "
                            "llegan a esta necesidad por evidencia independiente y pertenecen a facultades "
                            "distintas: su combinación amplía el abordaje más allá de un solo dominio."
                        ),
                        prioridad=_prioridad(prioridad_necesidad, [c_a.relevancia, c_b.relevancia]),
                        evidencias=c_a.evidencias[:1] + c_b.evidencias[:1],
                    )
                )
                if len(oportunidades) >= 3:
                    return oportunidades
        return oportunidades

    def _integracion_curricular(
        self, prioridad_necesidad: str, resultados_a: list[Conexion], asignaturas: list[Conexion]
    ) -> list[Oportunidad]:
        programas_fuente: dict[str, float] = {}
        for c in resultados_a:
            programa_id = self.repo.entidades[c.destino_id].valor("program_id")
            if programa_id:
                programas_fuente[programa_id] = max(programas_fuente.get(programa_id, 0.0), c.relevancia)

        mejor_por_programa: dict[str, float] = {}
        for c in asignaturas:
            programa_id = self.repo.entidades[c.destino_id].valor("program_id")
            mejor_por_programa[programa_id] = max(mejor_por_programa.get(programa_id, 0.0), c.relevancia)

        oportunidades = []
        for programa_id, score_fuente in programas_fuente.items():
            programa = self.repo.entidades.get(programa_id)
            if programa is None:
                continue
            cobertura = mejor_por_programa.get(programa_id, 0.0)
            if cobertura < 0.20:
                oportunidades.append(
                    Oportunidad(
                        tipo="CURRICULAR_INTEGRATION",
                        titulo=f"Brecha curricular en {programa.valor('program_name')}",
                        entidades_relacionadas=[programa_id],
                        razon=(
                            f"El programa {programa.valor('program_name')} produce antecedentes relevantes para "
                            "esta necesidad, pero ninguna de sus asignaturas muestra articulación temática clara "
                            "(mejor cobertura curricular encontrada: "
                            f"{cobertura:.2f}). Se sugiere revisar el currículo del programa."
                        ),
                        prioridad=_prioridad(prioridad_necesidad, [score_fuente]),
                        evidencias=[programa.evidencia("strategic_topics")],
                    )
                )
            elif cobertura >= 0.35:
                mejor_asignatura = max(
                    (c for c in asignaturas if self.repo.entidades[c.destino_id].valor("program_id") == programa_id),
                    key=lambda c: c.relevancia,
                )
                asignatura = self.repo.entidades[mejor_asignatura.destino_id]
                oportunidades.append(
                    Oportunidad(
                        tipo="CURRICULAR_INTEGRATION",
                        titulo=f"Integrar como caso de estudio en {asignatura.valor('subject_name')}",
                        entidades_relacionadas=[programa_id, asignatura.id],
                        razon=(
                            f"La asignatura {asignatura.valor('subject_name')} ({programa.valor('program_name')}) "
                            f"ya está alineada temáticamente (score {cobertura:.2f}); puede incorporar antecedentes "
                            "reales del proyecto/tesis relacionado como caso de estudio."
                        ),
                        prioridad=_prioridad(prioridad_necesidad, [cobertura]),
                        evidencias=[asignatura.evidencia("main_topics")],
                    )
                )
        return oportunidades

    def _activacion_capacidades(self, prioridad_necesidad: str, capacidades: list[Conexion]) -> list[Oportunidad]:
        # Data V1.0 modela una misma capacidad institucional como varias filas
        # (una por tipo de recurso: INFRASTRUCTURE/METHODOLOGICAL/COMPUTATIONAL/...)
        # que comparten nombre y facultad. Presentarlas todas como oportunidades
        # separadas sería redundante para la institución: se activa "la
        # capacidad" una vez, no cada dimensión de recurso por separado.
        vistos_por_nombre: set[str] = set()
        oportunidades = []
        for c in capacidades:
            capacidad = self.repo.entidades[c.destino_id]
            nombre_base = _nombre_base_capacidad(capacidad)
            if nombre_base in vistos_por_nombre:
                continue
            if capacidad.valor("status") != "ACTIVE":
                continue
            nivel = capacidad.valor("maturity_level")
            if not nivel.isdigit() or int(nivel) < 3:
                continue
            vistos_por_nombre.add(nombre_base)
            oportunidades.append(
                Oportunidad(
                    tipo="CAPABILITY_ACTIVATION",
                    titulo=f"Activar {capacidad.valor('capability_name')}",
                    entidades_relacionadas=[capacidad.id],
                    razon=(
                        f"{capacidad.valor('capability_name')} tiene madurez {nivel}/5, está activa y su perfil "
                        f"semántico se relaciona con la necesidad (score {c.relevancia:.2f} - inferido, "
                        "sin arista explícita capacidad-proyecto en el dataset)."
                    ),
                    prioridad=_prioridad(prioridad_necesidad, [c.relevancia]),
                    evidencias=c.evidencias[:2],
                )
            )
            if len(oportunidades) >= 3:
                break
        return oportunidades

    def _oportunidad_tesis(self, prioridad_necesidad: str, resultados_a: list[Conexion]) -> list[Oportunidad]:
        tesis_resultados = [r for r in resultados_a if r.tipo_destino == "THESIS"]
        if not tesis_resultados:
            return []
        mejor = tesis_resultados[0]
        tesis = self.repo.entidades[mejor.destino_id]
        anio_txt = tesis.valor("graduation_year")
        if not anio_txt.isdigit():
            return []
        anio = int(anio_txt)
        minimo, maximo = self.motor._rango_anios.get("THESIS", (2015, 2026))
        punto_medio = (minimo + maximo) / 2
        if anio > punto_medio:
            return []  # el antecedente ya es reciente, no hay hueco evidente
        return [
            Oportunidad(
                tipo="THESIS_OPPORTUNITY",
                titulo=f"Nuevo trabajo de grado que actualice {tesis.id}",
                entidades_relacionadas=[tesis.id],
                razon=(
                    f"El antecedente más relevante disponible ({tesis.valor('title')}) data de {anio}, "
                    f"anterior a la mitad del rango observado en el dataset ({minimo}-{maximo}). "
                    "Es un hueco razonable para proponer un nuevo trabajo de grado que actualice esta línea."
                ),
                prioridad=_prioridad(prioridad_necesidad, [mejor.relevancia]),
                evidencias=[tesis.evidencia("graduation_year"), tesis.evidencia("conclusions")],
            )
        ]
