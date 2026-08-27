import pytest

from src.grafo import GrafoConocimiento
from src.ingesta import RepositorioInstitucional


@pytest.fixture(scope="module")
def repo():
    return RepositorioInstitucional().cargar()


@pytest.fixture(scope="module")
def grafo(repo):
    return GrafoConocimiento().construir(repo)


def test_todos_los_nodos_presentes(repo, grafo):
    assert grafo.g.number_of_nodes() == len(repo.entidades)


def test_hay_aristas_explicitas(grafo):
    assert grafo.g.number_of_edges() > 3000


def test_toda_arista_tiene_naturaleza_explicita_y_procedencia(grafo):
    for _, _, datos in grafo.g.edges(data=True):
        assert datos["naturaleza"] == "explicita"
        assert "cita" in datos["procedencia"]


def test_conteo_exacto_de_aristas_por_tabla_de_relacion(repo, grafo):
    # Las tablas de relación tienen integridad referencial completa (verificado
    # en la exploración inicial): toda fila apunta a IDs que sí existen.
    conteos_esperados = {
        "miembro_de_grupo": len(repo.tablas_relacion["researcher_group"]),
        "participa_en_proyecto": len(repo.tablas_relacion["researcher_project"]),
        "dirige_tesis": len(repo.tablas_relacion["thesis_advisor"]),
        "proyecto_grupo_ejecutor": len(repo.tablas_relacion["project_group"]),
        "publicacion_de_proyecto": len(repo.tablas_relacion["publication_project"]),
        "autor_de_publicacion": len(repo.tablas_relacion["publication_researcher"]),
    }
    conteos_reales = {}
    for _, _, datos in grafo.g.edges(data=True):
        etiqueta = datos["tipo_relacion"]
        if etiqueta in conteos_esperados:
            conteos_reales[etiqueta] = conteos_reales.get(etiqueta, 0) + 1
    for etiqueta, esperado in conteos_esperados.items():
        assert conteos_reales.get(etiqueta) == esperado, etiqueta


def test_sin_aristas_colgantes(grafo):
    nodos = set(grafo.g.nodes())
    for origen, destino in grafo.g.edges():
        assert origen in nodos and destino in nodos


def test_investigador_conecta_a_proyecto_y_facultad(grafo):
    vecinos = set(grafo.g.successors("INV-001"))
    tipos_vecinos = {grafo.g.nodes[v]["tipo"] for v in vecinos}
    assert "FACULTY" in tipos_vecinos


def test_persistencia_pickle_ida_y_vuelta(grafo, tmp_path):
    ruta = tmp_path / "grafo_test.gpickle"
    grafo.guardar(ruta)
    recargado = GrafoConocimiento.cargar(ruta)
    assert recargado.g.number_of_nodes() == grafo.g.number_of_nodes()
    assert recargado.g.number_of_edges() == grafo.g.number_of_edges()
