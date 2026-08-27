"""Genera el reporte de métricas proxy y ablación sobre Data V1.0.

No existe Gold Standard oficial (Guía Oficial §4.2): las métricas de
ranking se calculan contra un silver standard AUTODERIVADO del propio
dataset (ver src/evaluacion.py). Se declara así explícitamente en el
reporte, tal como exige el Documento Técnico §12.

Uso:
    python scripts/evaluar.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import obtener_o_construir_indice
from src.evaluacion import construir_silver_standard, ejecutar_ablacion, evaluar_ranking, medir_trazabilidad_y_evidencia
from src.facetas import cargar_vocabulario
from src.grafo import GrafoConocimiento
from src.ingesta import RepositorioInstitucional
from src.motor import MotorConexiones


def main() -> None:
    repo = RepositorioInstitucional().cargar()
    grafo = GrafoConocimiento.cargar()
    vocabulario = cargar_vocabulario()
    textos = {eid: e.texto_indexable() for eid, e in repo.entidades.items()}
    indice = obtener_o_construir_indice(textos)
    motor = MotorConexiones(repo, grafo, indice, vocabulario)

    print("=" * 70)
    print("KNOWLEDGE NEXUS LATAM — Reporte de métricas (proxy, no oficial)")
    print("=" * 70)
    print(f"Motor semántico disponible: {indice.motor_semantico_disponible}")

    silver = construir_silver_standard(repo, vocabulario)
    print(
        f"\nSilver standard autoderivado: {len(silver)}/{len(repo.por_tipo('NEED'))} necesidades "
        "tienen >=2 términos TEMA compartidos con algún proyecto/tesis."
    )
    print(
        "(Las necesidades restantes, NEED-021..042 aprox., no traen términos temáticos "
        "explícitos en su descripción — ver facetas.py — por lo que no se les asigna "
        "un proxy léxico y quedan fuera de esta métrica, no se les pone en 0 artificialmente.)"
    )

    print("\n--- Ranking (Etapa A, proyecto+tesis) contra silver standard ---")
    resultado = evaluar_ranking(motor, repo, silver, top_k=10)
    print(f"Necesidades evaluadas: {resultado['n_necesidades_con_referencia']} / {resultado['n_necesidades_totales']}")
    for k, v in resultado["promedio"].items():
        print(f"  {k}: {v:.4f}")
    print(f"  latencia p50: {resultado['latencia_p50_seg']:.4f}s | p95: {resultado['latencia_p95_seg']:.4f}s")

    print("\n--- Trazabilidad y cobertura de evidencia (todas las 42 necesidades) ---")
    consultas = [n.texto_indexable() for n in repo.por_tipo("NEED")]
    te = medir_trazabilidad_y_evidencia(motor, repo, consultas, top_k=10)
    print(f"  cobertura_evidencia: {te['cobertura_evidencia']:.4f}")
    print(f"  trazabilidad: {te['trazabilidad']:.4f}")
    print(f"  n_resultados_evaluados: {te['n_resultados_evaluados']}")

    print("\n--- Ablación: NDCG@10 promedio por variante del motor ---")
    ablacion = ejecutar_ablacion(motor, repo, silver, top_k=10)
    for nombre, ndcg in sorted(ablacion.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {nombre:<38} NDCG@10 = {ndcg:.4f}")
    print(
        "\n  AVISO METODOLÓGICO: el silver standard se construye con solape de facetas TEMA\n"
        "  (ver construir_silver_standard), así que las variantes basadas en facetas/léxico\n"
        "  puntúan más alto aquí casi por definición — es un sesgo circular conocido, no una\n"
        "  prueba de que el motor semántico 'sobre'. Esta tabla solo sirve para comparar señales\n"
        "  sobre las 20 necesidades CON plantilla léxica (NEED-001..020). El valor real del motor\n"
        "  semántico se demuestra en las 22 necesidades SIN plantilla (NEED-021..042) y en consultas\n"
        "  libres nuevas (p. ej. 'deserción estudiantil'), donde no hay señal léxica que evaluar y el\n"
        "  sistema solo puede responder bien gracias a los embeddings (ver docs/casos_demostrables.md)."
    )

    print("\nFin del reporte.")


if __name__ == "__main__":
    main()
