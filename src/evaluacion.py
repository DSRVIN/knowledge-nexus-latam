"""Métricas proxy y ablación (Documento Técnico §12).

Data V1.0 pública no trae Gold Standard, rankings esperados ni relaciones
verdaderas ocultas (Guía Oficial §4.2) — es deliberado, no un descuido.
En su ausencia, este módulo construye un "silver standard" AUTODERIVADO
directamente del propio dataset: para cada necesidad, los proyectos/tesis
que comparten >=2 términos de la faceta TEMA (vocabulario controlado,
ver facetas.py) se consideran una referencia débil de relevancia. Se declara
explícitamente como proxy, nunca como verdad oficial, y se documenta contra
qué se evaluó y cómo se calculó, como exige el Documento Técnico §12.
"""
from __future__ import annotations

import math
import time

from src.facetas import VocabularioFacetas
from src.ingesta import RepositorioInstitucional
from src.motor import MotorConexiones

TIPOS_SILVER = ("PROJECT", "THESIS")


def construir_silver_standard(
    repo: RepositorioInstitucional, vocabulario: VocabularioFacetas, min_terminos_compartidos: int = 2
) -> dict[str, set[str]]:
    facetas_necesidad = {n.id: vocabulario.extraer(n.texto_indexable())["tema"] for n in repo.por_tipo("NEED")}
    facetas_candidato = {}
    for tipo in TIPOS_SILVER:
        for e in repo.por_tipo(tipo):
            facetas_candidato[e.id] = vocabulario.extraer(e.texto_indexable())["tema"]

    referencia = {}
    for need_id, tema_need in facetas_necesidad.items():
        if not tema_need:
            continue  # sin términos TEMA (needs 021-042): no hay base léxica para un proxy honesto
        relevantes = {
            cid for cid, tema_c in facetas_candidato.items() if len(tema_need & tema_c) >= min_terminos_compartidos
        }
        if relevantes:
            referencia[need_id] = relevantes
    return referencia


def _metricas_ranking(ranking: list[str], relevantes: set[str], k: int) -> dict:
    top_k = ranking[:k]
    hits = [1 if rid in relevantes else 0 for rid in top_k]
    precision = sum(hits) / k if k else 0.0
    recall = sum(hits) / len(relevantes) if relevantes else 0.0
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevantes), k)))
    ndcg = dcg / idcg if idcg else 0.0
    mrr = next((1 / (i + 1) for i, rid in enumerate(ranking) if rid in relevantes), 0.0)
    return {"precision_at_k": precision, "recall_at_k": recall, "ndcg_at_k": ndcg, "mrr": mrr}


def evaluar_ranking(
    motor: MotorConexiones,
    repo: RepositorioInstitucional,
    silver_standard: dict[str, set[str]],
    top_k: int = 10,
    pesos: dict = None,
    mmr_lambda: float = None,
) -> dict:
    metricas_por_need = {}
    latencias = []
    for need_id, relevantes in silver_standard.items():
        need = repo.entidades[need_id]
        t0 = time.perf_counter()
        resultados = motor.buscar(
            need.texto_indexable(), tipos_destino=TIPOS_SILVER, top_k=top_k, pesos=pesos, mmr_lambda=mmr_lambda
        )
        latencias.append(time.perf_counter() - t0)
        ranking = [c.destino_id for c in resultados]
        metricas_por_need[need_id] = _metricas_ranking(ranking, relevantes, top_k)

    n = len(metricas_por_need)
    claves = ("precision_at_k", "recall_at_k", "ndcg_at_k", "mrr")
    promedio = {c: (sum(m[c] for m in metricas_por_need.values()) / n if n else 0.0) for c in claves}
    latencias.sort()
    p50 = latencias[len(latencias) // 2] if latencias else 0.0
    p95 = latencias[min(len(latencias) - 1, int(len(latencias) * 0.95))] if latencias else 0.0

    return {
        "n_necesidades_con_referencia": n,
        "n_necesidades_totales": len(repo.por_tipo("NEED")),
        "promedio": promedio,
        "por_necesidad": metricas_por_need,
        "latencia_p50_seg": round(p50, 4),
        "latencia_p95_seg": round(p95, 4),
    }


def medir_trazabilidad_y_evidencia(
    motor: MotorConexiones, repo: RepositorioInstitucional, consultas: list[str], top_k: int = 10
) -> dict:
    total, con_evidencia, con_procedencia_resoluble = 0, 0, 0
    for texto in consultas:
        for c in motor.buscar(texto, top_k=top_k):
            total += 1
            if c.evidencias:
                con_evidencia += 1
            if any(e.procedencia.registro_id in repo.entidades for e in c.evidencias):
                con_procedencia_resoluble += 1
    return {
        "cobertura_evidencia": round(con_evidencia / total, 4) if total else 0.0,
        "trazabilidad": round(con_procedencia_resoluble / total, 4) if total else 0.0,
        "n_resultados_evaluados": total,
    }


def ejecutar_ablacion(
    motor: MotorConexiones, repo: RepositorioInstitucional, silver_standard: dict[str, set[str]], top_k: int = 10
) -> dict[str, float]:
    """Compara variantes del motor contra el mismo silver standard para
    cuantificar el aporte real de cada señal (no solo argumentarlo)."""
    variantes = {
        "completo (híbrido)": {"pesos": None, "mmr_lambda": None},
        "solo léxico (TF-IDF)": {
            "pesos": {"w_sem": 0, "w_lex": 1.0, "w_tema": 0, "w_dominio": 0, "w_metodo": 0, "w_evidencia": 0},
            "mmr_lambda": None,
        },
        "solo semántico (embeddings)": {
            "pesos": {"w_sem": 1.0, "w_lex": 0, "w_tema": 0, "w_dominio": 0, "w_metodo": 0, "w_evidencia": 0},
            "mmr_lambda": None,
        },
        "solo facetas (tema+dominio+método)": {
            "pesos": {"w_sem": 0, "w_lex": 0, "w_tema": 0.5, "w_dominio": 0.3, "w_metodo": 0.2, "w_evidencia": 0},
            "mmr_lambda": None,
        },
        "híbrido sin MMR": {"pesos": None, "mmr_lambda": 1.0},
    }
    resultados = {}
    for nombre, cfg in variantes.items():
        ndcgs = []
        for need_id, relevantes in silver_standard.items():
            need = repo.entidades[need_id]
            r = motor.buscar(
                need.texto_indexable(), tipos_destino=TIPOS_SILVER, top_k=top_k,
                pesos=cfg["pesos"], mmr_lambda=cfg["mmr_lambda"],
            )
            ranking = [c.destino_id for c in r]
            ndcgs.append(_metricas_ranking(ranking, relevantes, top_k)["ndcg_at_k"])
        resultados[nombre] = round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0
    return resultados
