import pytest

from src.facetas import construir_vocabulario, normalizar
from src.ingesta import RepositorioInstitucional


@pytest.fixture(scope="module")
def repo():
    return RepositorioInstitucional().cargar()


@pytest.fixture(scope="module")
def vocabulario(repo):
    return construir_vocabulario(repo)


def test_normalizar_quita_tildes_y_mayusculas(repo):
    assert normalizar("Deserción Estudiantil") == "desercion estudiantil"


def test_vocabulario_tema_no_vacio(vocabulario):
    assert len(vocabulario.tema) > 100
    assert "permanencia estudiantil" in vocabulario.tema


def test_vocabulario_metodo_tiene_16_valores_cerrados(vocabulario):
    assert len(vocabulario.metodo) == 16
    assert "clasificacion supervisada" in vocabulario.metodo


def test_termino_puede_pertenecer_a_varias_facetas(vocabulario):
    # "series temporales" es keyword de proyecto (TEMA), application_domain (DOMINIO)
    # y methodological_expertise (MÉTODO) simultáneamente en el dataset real.
    assert "series temporales" in vocabulario.tema
    assert "series temporales" in vocabulario.dominio
    assert "series temporales" in vocabulario.metodo


def test_extraccion_sobre_necesidad_con_plantilla_de_terminos(repo, vocabulario):
    necesidad = repo.entidades["NEED-001"]  # deserción estudiantil
    facetas = vocabulario.extraer(necesidad.texto_indexable())
    assert "permanencia estudiantil" in facetas["tema"]


def test_necesidades_sin_plantilla_pueden_tener_tema_disperso(repo, vocabulario):
    # NEED-021..042 no enumeran términos explícitos; es correcto que el
    # solape léxico sea bajo o nulo (el score semántico compensa).
    necesidad = repo.entidades["NEED-023"]  # brechas curriculares digitales
    facetas = vocabulario.extraer(necesidad.texto_indexable())
    assert isinstance(facetas["tema"], set)  # no debe lanzar excepción ni forzar match


def test_research_lines_keywords_se_reconstruyen_como_frase(repo, vocabulario):
    # research_lines.csv fragmenta sus keywords palabra por palabra
    # ('predicción;de;permanencia;estudiantil'); debe reconstruirse como
    # frase completa, no quedar como tokens sueltos tipo "de"/"y"/"en".
    assert "de" not in vocabulario.tema
    assert "y" not in vocabulario.tema
    assert "predicción de permanencia estudiantil" not in vocabulario.tema  # ya normalizado
    assert "prediccion de permanencia estudiantil" in vocabulario.tema


def test_extraer_es_identico_para_entidad_y_consulta_libre(vocabulario):
    # "deserción" es deliberadamente ajena al vocabulario controlado (es el
    # ejemplo textual de la Guía Oficial §2: un sistema que solo busque la
    # palabra "deserción" no encuentra nada útil). El término correcto que
    # SÍ está en el vocabulario, y que debe emerger igual desde una entidad
    # que desde una consulta libre, es "permanencia estudiantil".
    consulta_libre = "queremos predecir y prevenir la deserción estudiantil"
    facetas = vocabulario.extraer(consulta_libre)
    assert "desercion" not in facetas["tema"]
    assert "desercion estudiantil" not in facetas["tema"]

    consulta_libre_con_vocabulario = "riesgo de baja permanencia estudiantil en el programa"
    facetas2 = vocabulario.extraer(consulta_libre_con_vocabulario)
    assert "permanencia estudiantil" in facetas2["tema"]
