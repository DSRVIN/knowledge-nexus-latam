"""Motor de descubrimiento y priorización. Dos etapas:

Etapa A: recuperación híbrida sobre entidades documentales (proyecto, tesis,
publicación, línea) con score descompuesto y visible, MMR para diversidad.

Etapa B: propagación por aristas EXPLÍCITAS del grafo hacia investigadores y
grupos (evidencia recuperada) + afinidad semántica de perfil (inferida);
articulación curricular vía programa compartido + semántica; capacidades
por emparejamiento semántico puro (siempre etiquetado como inferido, porque
el dataset no tiene arista explícita capacidad↔proyecto).

Reglas de interpretación del Documento Técnico §16 implementadas aquí:
- "mismo método no significa mismo problema": penalizacion_metodo_sin_tema.
- "misma facultad no significa mayor pertinencia": no existe w_facultad.
- "más resultados no significa mejor solución": MMR obligatorio, no opcional.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src import config
from src.embeddings import IndiceEmbeddings
from src.facetas import VocabularioFacetas, facetas_por_entidad
from src.grafo import GrafoConocimiento
from src.ingesta import Entidad, RepositorioInstitucional
from src.trazabilidad import DesgloseScore, Evidencia, Procedencia

TIPOS_DOCUMENTALES = ("PROJECT", "THESIS", "PUBLICATION", "LINE")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _nombre_legible(entidad: Entidad) -> str:
    for campo in ("full_name", "title", "group_name", "program_name", "faculty_name",
                  "line_name", "capability_name", "subject_name"):
        valor = entidad.valor(campo)
        if valor:
            return valor
    return entidad.id


@dataclass
class Conexion:
    origen_id: str
    destino_id: str
    tipo_destino: str
    tipo_relacion: str
    relevancia: float
    banda: str
    naturaleza: str  # "explicita" | "inferida" | "mixta"
    desglose: DesgloseScore
    evidencias: list[Evidencia] = field(default_factory=list)
    facetas_compartidas: dict = field(default_factory=dict)
    explicacion: str = ""
    nombre_destino: str = ""

    def to_dict(self) -> dict:
        return {
            "origen": {"id": self.origen_id},
            "destino": {"id": self.destino_id, "tipo": self.tipo_destino, "nombre": self.nombre_destino},
            "tipo_relacion": self.tipo_relacion,
            "relevancia": round(self.relevancia, 4),
            "banda": self.banda,
            "naturaleza": self.naturaleza,
            "explicacion": self.explicacion,
            "desglose": self.desglose.to_dict(),
            "evidencia": [e.to_dict() for e in self.evidencias],
            "sources": [e.procedencia.cita() for e in self.evidencias],
        }


def _banda(total: float) -> str:
    # Calibrado empíricamente corriendo las 42 necesidades reales contra
    # projects+theses+publications+lines (ver docs/casos_demostrables.md).
    # La distribución es bimodal: las 20 necesidades con plantilla léxica
    # (NEED-001..020) obtienen top-1 en ~0.45-0.71; las 22 sin plantilla
    # (NEED-021..042, ver facetas.py) obtienen ~0.26-0.40 porque dependen
    # casi enteramente de la señal semántica. El corte en 0.45 separa esa
    # población; 0.25 es el piso por debajo del cual ni siquiera las
    # necesidades difíciles logran su mejor candidato.
    if total >= 0.45:
        return "Alta"
    if total >= 0.25:
        return "Media"
    return "Baja"


class MotorConexiones:
    def __init__(
        self,
        repo: RepositorioInstitucional,
        grafo: GrafoConocimiento,
        indice: IndiceEmbeddings,
        vocabulario: VocabularioFacetas,
    ) -> None:
        self.repo = repo
        self.grafo = grafo
        self.indice = indice
        self.vocabulario = vocabulario
        self._id_a_idx = {eid: i for i, eid in enumerate(indice.ids)}
        self._facetas_entidad = facetas_por_entidad(repo, vocabulario)
        self._fuerza_evidencia_cache: dict[str, float] = {}
        self._rango_anios = self._calcular_rango_anios()

    # ------------------------------------------------------------------ #
    # Utilidades internas
    # ------------------------------------------------------------------ #
    def _calcular_rango_anios(self) -> dict[str, tuple[int, int]]:
        campo_por_tipo = {"PROJECT": "end_year", "THESIS": "graduation_year", "PUBLICATION": "year"}
        rangos = {}
        for tipo, campo in campo_por_tipo.items():
            anios = []
            for e in self.repo.por_tipo(tipo):
                v = e.valor(campo)
                if v.isdigit():
                    anios.append(int(v))
            rangos[tipo] = (min(anios), max(anios)) if anios else (2015, 2026)
        return rangos

    def _factor_recencia(self, entidad: Entidad) -> float:
        campo_por_tipo = {"PROJECT": "end_year", "THESIS": "graduation_year", "PUBLICATION": "year"}
        campo = campo_por_tipo.get(entidad.tipo)
        if campo is None:
            return 1.0
        valor = entidad.valor(campo)
        if not valor.isdigit():
            return 0.8  # dato faltante: neutro-conservador, no penaliza de más
        anio = int(valor)
        minimo, maximo = self._rango_anios.get(entidad.tipo, (2015, 2026))
        rango = max(1, maximo - minimo)
        frac = min(1.0, max(0.0, (anio - minimo) / rango))
        return 0.6 + 0.4 * frac  # nunca castiga a 0: lo antiguo sigue siendo evidencia válida

    def _fuerza_evidencia(self, entidad: Entidad) -> float:
        if entidad.id in self._fuerza_evidencia_cache:
            return self._fuerza_evidencia_cache[entidad.id]
        campos = config.CAMPOS_SEMANTICOS_POR_TIPO.get(entidad.tipo, [])
        if not campos:
            resultado = 0.5
        else:
            completos = sum(
                1 for c in campos if entidad.valor(c) or entidad.valores_multivalor(c)
            )
            completitud = completos / len(campos)
            nivel = self.repo.nivel_confiabilidad(entidad.tipo)
            peso_conf = config.PESO_CONFIABILIDAD.get(nivel, 0.6)
            resultado = completitud * peso_conf * self._factor_recencia(entidad)
        self._fuerza_evidencia_cache[entidad.id] = resultado
        return resultado

    def _pesos_efectivos(self, pesos: dict) -> dict:
        """Degradación controlada: si el motor semántico no está disponible,
        su peso se redistribuye sobre la señal léxica en vez de perderse."""
        if self.indice.motor_semantico_disponible:
            return pesos
        ajustado = dict(pesos)
        ajustado["w_lex"] = ajustado.get("w_lex", 0.0) + ajustado.get("w_sem", 0.0)
        ajustado["w_sem"] = 0.0
        return ajustado

    def _evidencias_principales(self, entidad: Entidad, max_campos: int = 3) -> list[Evidencia]:
        campos = config.CAMPOS_SEMANTICOS_POR_TIPO.get(entidad.tipo, [])
        evidencias = []
        for campo in campos:
            if entidad.valor(campo) or entidad.valores_multivalor(campo):
                evidencias.append(entidad.evidencia(campo))
            if len(evidencias) >= max_campos:
                break
        return evidencias

    # ------------------------------------------------------------------ #
    # Etapa A: entidades documentales
    # ------------------------------------------------------------------ #
    def _calcular_score(
        self, eid: str, vec_sem_q: np.ndarray | None, vec_lex_q: np.ndarray, facetas_q: dict, pesos: dict
    ) -> tuple[DesgloseScore, dict]:
        idx = self._id_a_idx[eid]
        entidad = self.repo.entidades[eid]
        componentes = {}

        if vec_sem_q is not None and self.indice.vectores_semanticos is not None:
            componentes["semantico"] = pesos["w_sem"] * _cos(vec_sem_q, self.indice.vectores_semanticos[idx])
        else:
            componentes["semantico"] = 0.0
        componentes["lexico"] = pesos["w_lex"] * _cos(vec_lex_q, self.indice.vectores_lexicos[idx])

        facetas_e = self._facetas_entidad[eid]
        jac_tema = _jaccard(facetas_q["tema"], facetas_e["tema"])
        jac_dominio = _jaccard(facetas_q["dominio"], facetas_e["dominio"])
        jac_metodo = _jaccard(facetas_q["metodo"], facetas_e["metodo"])
        componentes["tema"] = pesos["w_tema"] * jac_tema
        componentes["dominio"] = pesos["w_dominio"] * jac_dominio
        componentes["metodo"] = pesos["w_metodo"] * jac_metodo
        componentes["evidencia"] = pesos["w_evidencia"] * self._fuerza_evidencia(entidad)

        total = sum(componentes.values())
        penalizaciones = {}
        if jac_tema == 0.0 and jac_metodo > 0.0:
            factor = config.PENALIZACION_METODO_SIN_TEMA
            penalizaciones["metodo_sin_tema"] = round(total * (factor - 1), 4)
            total *= factor

        desglose = DesgloseScore(componentes=componentes, penalizaciones=penalizaciones, total=total)
        facetas_compartidas = {
            "tema": facetas_q["tema"] & facetas_e["tema"],
            "dominio": facetas_q["dominio"] & facetas_e["dominio"],
            "metodo": facetas_q["metodo"] & facetas_e["metodo"],
        }
        return desglose, facetas_compartidas

    def _aplicar_mmr(self, candidatos: list[tuple[str, DesgloseScore, dict]], top_k: int, lam: float = None) -> list:
        lam = config.MMR_LAMBDA if lam is None else lam
        if not candidatos:
            return []
        max_total = max(d.total for _, d, _ in candidatos) or 1.0

        def vector_diversidad(eid: str) -> np.ndarray:
            idx = self._id_a_idx[eid]
            if self.indice.vectores_semanticos is not None:
                return self.indice.vectores_semanticos[idx]
            return self.indice.vectores_lexicos[idx]

        restantes = list(candidatos)
        seleccionados: list = []
        while restantes and len(seleccionados) < top_k:
            mejor, mejor_score = None, -1e18
            for cand in restantes:
                eid, desglose, _ = cand
                relevancia = desglose.total / max_total
                if seleccionados:
                    sim_max = max(_cos(vector_diversidad(eid), vector_diversidad(s[0])) for s in seleccionados)
                else:
                    sim_max = 0.0
                mmr = lam * relevancia - (1 - lam) * sim_max
                if mmr > mejor_score:
                    mejor_score, mejor = mmr, cand
            seleccionados.append(mejor)
            restantes.remove(mejor)
        return seleccionados

    def buscar(
        self,
        texto_consulta: str,
        tipos_destino: tuple = TIPOS_DOCUMENTALES,
        top_k: int = None,
        pesos: dict = None,
        pool_previo_a_mmr: int = 50,
        mmr_lambda: float = None,
    ) -> list[Conexion]:
        top_k = top_k or config.TOP_K_DEFECTO
        pesos = self._pesos_efectivos(pesos or config.PESOS_SCORE)
        vec_sem_q, vec_lex_q = self.indice.vectorizar_consulta(texto_consulta)
        facetas_q = self.vocabulario.extraer(texto_consulta)

        candidatos = []
        for eid in self.indice.ids:
            entidad = self.repo.entidades[eid]
            if entidad.tipo not in tipos_destino:
                continue
            desglose, facetas_compartidas = self._calcular_score(eid, vec_sem_q, vec_lex_q, facetas_q, pesos)
            candidatos.append((eid, desglose, facetas_compartidas))

        candidatos.sort(key=lambda c: c[1].total, reverse=True)
        pool = candidatos[: max(pool_previo_a_mmr, top_k)]
        # mmr_lambda=1.0 equivale a "sin MMR" (ranking puro por relevancia);
        # ver tests/test_motor.py::test_mmr_con_lambda_1_equivale_a_ranking_puro_por_relevancia
        seleccionados = self._aplicar_mmr(pool, top_k, lam=mmr_lambda)

        conexiones = []
        for eid, desglose, facetas_compartidas in seleccionados:
            entidad = self.repo.entidades[eid]
            tipo_relacion = (
                "metodo_transferible" if "metodo_sin_tema" in desglose.penalizaciones else "antecedente_relevante"
            )
            conexiones.append(
                Conexion(
                    origen_id="CONSULTA",
                    destino_id=eid,
                    tipo_destino=entidad.tipo,
                    tipo_relacion=tipo_relacion,
                    relevancia=desglose.total,
                    banda=_banda(desglose.total),
                    naturaleza="inferida",
                    desglose=desglose,
                    evidencias=self._evidencias_principales(entidad),
                    facetas_compartidas=facetas_compartidas,
                    nombre_destino=_nombre_legible(entidad),
                )
            )
        return conexiones

    # ------------------------------------------------------------------ #
    # Etapa B: propagación en grafo hacia investigadores y grupos
    # ------------------------------------------------------------------ #
    def _vecinos_por_relacion(self, nodo_id: str, etiquetas: set, direccion: str) -> list[tuple[str, dict]]:
        g = self.grafo.g
        resultados = []
        if direccion in ("entrante", "ambas") and nodo_id in g:
            for u, _, datos in g.in_edges(nodo_id, data=True):
                if datos["tipo_relacion"] in etiquetas:
                    resultados.append((u, datos))
        if direccion in ("saliente", "ambas") and nodo_id in g:
            for _, v, datos in g.out_edges(nodo_id, data=True):
                if datos["tipo_relacion"] in etiquetas:
                    resultados.append((v, datos))
        return resultados

    def _propagar(
        self,
        resultados_etapa_a: list[Conexion],
        tipo_destino: str,
        etiquetas_relacion: set,
        direccion: str,
        tabla_pesos_rol: dict,
    ) -> dict[str, dict]:
        acumulado: dict[str, dict] = {}
        for conexion in resultados_etapa_a:
            fuente_id, score_fuente = conexion.destino_id, conexion.relevancia
            vecinos = self._vecinos_por_relacion(fuente_id, etiquetas_relacion, direccion)
            por_vecino: dict[str, list[dict]] = {}
            for vecino_id, datos in vecinos:
                if self.repo.entidades.get(vecino_id) is None:
                    continue
                if self.repo.entidades[vecino_id].tipo != tipo_destino:
                    continue
                por_vecino.setdefault(vecino_id, []).append(datos)
            for vecino_id, lista_datos in por_vecino.items():
                # aristas paralelas (FK + tabla de relación) describen el mismo
                # hecho: se cuenta una sola vez, con el mejor rol disponible.
                mejor_peso_rol, mejor_datos = 0.0, lista_datos[0]
                for datos in lista_datos:
                    peso_rol = tabla_pesos_rol.get(datos.get("rol", ""), 0.6)
                    if peso_rol >= mejor_peso_rol:
                        mejor_peso_rol, mejor_datos = peso_rol, datos
                contribucion = config.AMORTIGUACION_PROPAGACION * score_fuente * mejor_peso_rol
                entrada = acumulado.setdefault(vecino_id, {"evidencia_directa": 0.0, "fuentes": []})
                entrada["evidencia_directa"] += contribucion
                entrada["fuentes"].append(
                    {
                        "fuente_id": fuente_id,
                        "score_fuente": round(score_fuente, 4),
                        "rol": mejor_datos.get("rol", ""),
                        "peso_rol": mejor_peso_rol,
                        "procedencia": Procedencia.from_dict(mejor_datos["procedencia"]),
                    }
                )
        return acumulado

    def _conexion_propagada(
        self, entidad_id: str, tipo_destino: str, evidencia_directa: float, afinidad: float, fuentes: list[dict]
    ) -> Conexion:
        evidencia_directa = round(min(1.0, evidencia_directa), 4)
        afinidad = round(afinidad, 4)
        desglose = DesgloseScore(
            componentes={
                "evidencia_directa": evidencia_directa,
                "afinidad_perfil_inferida": afinidad,
            },
            total=evidencia_directa + afinidad,
        )
        evidencias = [
            Evidencia(
                procedencia=f["procedencia"],
                texto=f"{f['rol']} en {f['fuente_id']} (score de la fuente: {f['score_fuente']})",
                tipo="recuperada",
            )
            for f in fuentes
        ]
        naturaleza = "mixta" if afinidad > 0 else "explicita"
        total = desglose.total
        return Conexion(
            origen_id="CONSULTA",
            destino_id=entidad_id,
            tipo_destino=tipo_destino,
            tipo_relacion="propagada_por_autoria",
            relevancia=total,
            banda=_banda(total),
            naturaleza=naturaleza,
            desglose=desglose,
            evidencias=evidencias,
            nombre_destino=_nombre_legible(self.repo.entidades[entidad_id]),
        )

    def propagar_a_investigadores(
        self, resultados_etapa_a: list[Conexion], texto_consulta: str, top_k: int = None
    ) -> list[Conexion]:
        top_k = top_k or config.TOP_K_DEFECTO
        etiquetas = {"participa_en_proyecto", "dirige_tesis", "autor_de_publicacion"}
        acumulado = self._propagar(
            resultados_etapa_a, "RESEARCHER", etiquetas, "entrante", config.PESO_ROL_INVESTIGADOR
        )
        vec_sem_q, _ = self.indice.vectorizar_consulta(texto_consulta)
        conexiones = []
        for inv_id, datos in acumulado.items():
            afinidad = 0.0
            if vec_sem_q is not None and self.indice.vectores_semanticos is not None:
                idx = self._id_a_idx[inv_id]
                afinidad = config.PESO_AFINIDAD_PERFIL * max(
                    0.0, _cos(vec_sem_q, self.indice.vectores_semanticos[idx])
                )
            conexiones.append(
                self._conexion_propagada(inv_id, "RESEARCHER", datos["evidencia_directa"], afinidad, datos["fuentes"])
            )
        conexiones.sort(key=lambda c: c.relevancia, reverse=True)
        return conexiones[:top_k]

    def propagar_a_grupos(
        self, resultados_etapa_a: list[Conexion], texto_consulta: str, top_k: int = None
    ) -> list[Conexion]:
        top_k = top_k or config.TOP_K_DEFECTO
        etiquetas = {"proyecto_grupo_ejecutor", "proyecto_ejecutado_por_grupo", "linea_de_grupo"}
        acumulado = self._propagar(
            resultados_etapa_a, "GROUP", etiquetas, "saliente", config.PESO_ROL_GRUPO
        )
        vec_sem_q, _ = self.indice.vectorizar_consulta(texto_consulta)
        conexiones = []
        for grp_id, datos in acumulado.items():
            afinidad = 0.0
            if vec_sem_q is not None and self.indice.vectores_semanticos is not None:
                idx = self._id_a_idx[grp_id]
                afinidad = config.PESO_AFINIDAD_PERFIL * max(
                    0.0, _cos(vec_sem_q, self.indice.vectores_semanticos[idx])
                )
            conexiones.append(
                self._conexion_propagada(grp_id, "GROUP", datos["evidencia_directa"], afinidad, datos["fuentes"])
            )
        conexiones.sort(key=lambda c: c.relevancia, reverse=True)
        return conexiones[:top_k]

    # ------------------------------------------------------------------ #
    # Etapa B: currículo (puente explícito programa + semántica)
    # ------------------------------------------------------------------ #
    def articular_curriculo(
        self, resultados_etapa_a: list[Conexion], texto_consulta: str, top_k: int = None
    ) -> list[Conexion]:
        top_k = top_k or config.TOP_K_DEFECTO
        programas_relevantes: dict[str, float] = {}
        for c in resultados_etapa_a:
            entidad = self.repo.entidades[c.destino_id]
            programa_id = entidad.valor("program_id")
            if programa_id:
                programas_relevantes[programa_id] = max(programas_relevantes.get(programa_id, 0.0), c.relevancia)

        vec_sem_q, vec_lex_q = self.indice.vectorizar_consulta(texto_consulta)
        conexiones = []
        for asignatura in self.repo.por_tipo("SUBJECT"):
            programa_id = asignatura.valor("program_id")
            puente_explicito = programa_id in programas_relevantes
            idx = self._id_a_idx[asignatura.id]
            sim_sem = (
                _cos(vec_sem_q, self.indice.vectores_semanticos[idx])
                if vec_sem_q is not None and self.indice.vectores_semanticos is not None
                else 0.0
            )
            sim_lex = _cos(vec_lex_q, self.indice.vectores_lexicos[idx])
            if not puente_explicito and sim_sem < 0.35 and sim_lex < 0.15:
                continue
            evidencia_explicita = programas_relevantes.get(programa_id, 0.0) * 0.5 if puente_explicito else 0.0
            afinidad = config.PESO_AFINIDAD_PERFIL * max(sim_sem, sim_lex)
            total = evidencia_explicita + afinidad
            desglose = DesgloseScore(
                componentes={
                    "puente_programa_compartido": round(evidencia_explicita, 4),
                    "afinidad_semantica_inferida": round(afinidad, 4),
                },
                total=total,
            )
            evidencias = self._evidencias_principales(asignatura, max_campos=2)
            if puente_explicito:
                evidencias.append(asignatura.evidencia("program_id"))
            conexiones.append(
                Conexion(
                    origen_id="CONSULTA",
                    destino_id=asignatura.id,
                    tipo_destino="SUBJECT",
                    tipo_relacion="articulacion_curricular",
                    relevancia=total,
                    banda=_banda(total),
                    naturaleza="mixta" if puente_explicito else "inferida",
                    desglose=desglose,
                    evidencias=evidencias,
                    nombre_destino=_nombre_legible(asignatura),
                )
            )
        conexiones.sort(key=lambda c: c.relevancia, reverse=True)
        return conexiones[:top_k]

    # ------------------------------------------------------------------ #
    # Etapa B: capacidades (emparejamiento semántico puro, siempre inferido)
    # ------------------------------------------------------------------ #
    def emparejar_capacidades(self, texto_consulta: str, top_k: int = None) -> list[Conexion]:
        top_k = top_k or config.TOP_K_DEFECTO
        vec_sem_q, vec_lex_q = self.indice.vectorizar_consulta(texto_consulta)
        facetas_q = self.vocabulario.extraer(texto_consulta)
        conexiones = []
        for capacidad in self.repo.por_tipo("CAPABILITY"):
            idx = self._id_a_idx[capacidad.id]
            sim_sem = (
                _cos(vec_sem_q, self.indice.vectores_semanticos[idx])
                if vec_sem_q is not None and self.indice.vectores_semanticos is not None
                else 0.0
            )
            sim_lex = _cos(vec_lex_q, self.indice.vectores_lexicos[idx])
            facetas_e = self._facetas_entidad[capacidad.id]
            jac_dominio = _jaccard(facetas_q["dominio"], facetas_e["dominio"])
            total = 0.5 * max(sim_sem, sim_lex) + 0.3 * jac_dominio
            if total < 0.10:
                continue
            desglose = DesgloseScore(
                componentes={"afinidad_semantica": round(0.5 * max(sim_sem, sim_lex), 4),
                             "solape_dominio": round(0.3 * jac_dominio, 4)},
                total=total,
            )
            conexiones.append(
                Conexion(
                    origen_id="CONSULTA",
                    destino_id=capacidad.id,
                    tipo_destino="CAPABILITY",
                    tipo_relacion="capacidad_activable",
                    relevancia=total,
                    banda=_banda(total),
                    naturaleza="inferida",  # el dataset no tiene arista explícita capacidad↔proyecto
                    desglose=desglose,
                    evidencias=self._evidencias_principales(capacidad, max_campos=2),
                    nombre_destino=_nombre_legible(capacidad),
                )
            )
        conexiones.sort(key=lambda c: c.relevancia, reverse=True)
        return conexiones[:top_k]
