import pytest

from src.embeddings import obtener_o_construir_indice
from src.facetas import construir_vocabulario
from src.grafo import GrafoConocimiento
from src.ingesta import RepositorioInstitucional


@pytest.fixture(scope="session")
def repo_sesion():
    return RepositorioInstitucional().cargar()


@pytest.fixture(scope="session")
def grafo_sesion(repo_sesion):
    return GrafoConocimiento().construir(repo_sesion)


@pytest.fixture(scope="session")
def vocabulario_sesion(repo_sesion):
    return construir_vocabulario(repo_sesion)


@pytest.fixture(scope="session")
def indice_sesion(repo_sesion):
    # Corpus completo (3232 entidades): la primera vectorización con el
    # modelo local en español tarda varios minutos en CPU. obtener_o_construir_indice reutiliza la caché en
    # data/artifacts/ (poblada por scripts/construir_indice.py) si el corpus
    # no cambió, para que la suite de tests corra en segundos.
    textos = {eid: e.texto_indexable() for eid, e in repo_sesion.entidades.items()}
    return obtener_o_construir_indice(textos)
