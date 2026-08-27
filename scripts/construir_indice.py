"""Pipeline offline: ingesta -> facetas -> embeddings -> grafo -> artifacts.

Idempotente: si el corpus no cambió desde la última ejecución, reutiliza la
caché de embeddings en vez de re-vectorizar (ver src/embeddings.py).

Uso:
    python scripts/construir_indice.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import obtener_o_construir_indice
from src.facetas import construir_vocabulario, guardar_vocabulario
from src.grafo import GrafoConocimiento
from src.ingesta import RepositorioInstitucional


def main() -> None:
    t0 = time.time()
    print("[1/4] Ingesta de Data V1.0 (21 CSV + 60 documentos .md)...")
    repo = RepositorioInstitucional().cargar()
    repo.guardar_procesado()
    print(f"      {len(repo.entidades)} entidades cargadas ({time.time() - t0:.1f}s)")

    t1 = time.time()
    print("[2/4] Construyendo grafo de conocimiento...")
    grafo = GrafoConocimiento().construir(repo)
    grafo.guardar()
    print(
        f"      {grafo.g.number_of_nodes()} nodos, {grafo.g.number_of_edges()} aristas "
        f"explícitas ({time.time() - t1:.1f}s)"
    )

    t2 = time.time()
    print("[3/4] Construyendo vocabulario de facetas (TEMA/MÉTODO/DOMINIO)...")
    vocabulario = construir_vocabulario(repo)
    guardar_vocabulario(vocabulario)
    print(
        f"      TEMA={len(vocabulario.tema)} MÉTODO={len(vocabulario.metodo)} "
        f"DOMINIO={len(vocabulario.dominio)} ({time.time() - t2:.1f}s)"
    )

    t3 = time.time()
    print("[4/4] Vectorizando corpus institucional (semántico local + TF-IDF)...")
    textos = {eid: e.texto_indexable() for eid, e in repo.entidades.items()}
    indice = obtener_o_construir_indice(textos)
    motor = "embeddings locales en español (BETO, ONNX)" if indice.motor_semantico_disponible else "TF-IDF (fallback)"
    print(f"      motor semántico: {motor} | {len(indice.ids)} vectores ({time.time() - t3:.1f}s)")

    print(f"\nÍndice construido en {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    main()
