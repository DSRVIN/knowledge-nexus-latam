import collections

import pytest

from src import config
from src.ingesta import RepositorioInstitucional


@pytest.fixture(scope="module")
def repo():
    return RepositorioInstitucional().cargar()


def test_conteos_por_tipo_coinciden_con_manifiestos(repo):
    conteos = collections.Counter(e.tipo for e in repo.entidades.values())
    esperado = {
        "FACULTY": config.CONTEOS_ESPERADOS["faculties"],
        "PROGRAM": config.CONTEOS_ESPERADOS["programs"],
        "GROUP": config.CONTEOS_ESPERADOS["research_groups"],
        "LINE": config.CONTEOS_ESPERADOS["research_lines"],
        "CAPABILITY": config.CONTEOS_ESPERADOS["institutional_capabilities"],
        "RESEARCHER": config.CONTEOS_ESPERADOS["researchers"],
        "EXPERTISE": config.CONTEOS_ESPERADOS["researcher_expertise"],
        "SUBJECT": config.CONTEOS_ESPERADOS["subjects"],
        "COMPETENCY": config.CONTEOS_ESPERADOS["competencies"],
        "OUTCOME": config.CONTEOS_ESPERADOS["learning_outcomes"],
        "NEED": config.CONTEOS_ESPERADOS["institutional_needs"],
        "PROJECT": config.CONTEOS_ESPERADOS["projects"],
        "THESIS": config.CONTEOS_ESPERADOS["theses"],
        "PUBLICATION": config.CONTEOS_ESPERADOS["publications"],
    }
    for tipo, n in esperado.items():
        assert conteos[tipo] == n, f"{tipo}: esperado {n}, obtenido {conteos[tipo]}"


def test_conteos_relaciones(repo):
    assert len(repo.tablas_relacion["researcher_group"]) == config.CONTEOS_ESPERADOS["researcher_group"]
    assert len(repo.tablas_relacion["researcher_project"]) == config.CONTEOS_ESPERADOS["researcher_project"]
    assert len(repo.tablas_relacion["project_group"]) == config.CONTEOS_ESPERADOS["project_group"]
    assert len(repo.tablas_relacion["thesis_advisor"]) == config.CONTEOS_ESPERADOS["thesis_advisor"]
    assert len(repo.tablas_relacion["publication_researcher"]) == config.CONTEOS_ESPERADOS["publication_researcher"]
    assert len(repo.tablas_relacion["publication_project"]) == config.CONTEOS_ESPERADOS["publication_project"]


def test_ninguna_cabecera_arrastra_bom(repo):
    for entidad in repo.entidades.values():
        for campo in entidad.campos:
            assert not campo.startswith("﻿"), f"BOM sin limpiar en campo {campo!r} de {entidad.id}"


def test_documentos_markdown_vinculados(repo):
    con_md = [e for e in repo.entidades.values() if e.texto_documento_md]
    assert len(con_md) == 60  # 30 proyectos + 20 tesis + 10 necesidades


def test_procedencia_cita_formato_esperado(repo):
    proyecto = repo.entidades["PRJ-001"]
    proc = proyecto.procedencia("methodology")
    assert proc.cita() == "Data V1.0 / projects.csv / PRJ-001 / methodology"


def test_texto_indexable_no_vacio_para_necesidad(repo):
    necesidad = repo.entidades["NEED-001"]
    assert "deserción" in necesidad.texto_indexable().lower()


def test_confiabilidad_resuelta_por_moda(repo):
    # source_catalog.csv es ruidoso (7 entradas contradictorias por archivo);
    # debe resolverse a un único nivel determinista, no crashear.
    nivel = repo.nivel_confiabilidad("FACULTY")
    assert nivel in {"HIGH", "MEDIUM", "LOW"}
    # researchers.csv no está en el catálogo -> nivel neutro documentado
    assert repo.nivel_confiabilidad("RESEARCHER") == config.NIVEL_CONFIABILIDAD_NO_CATALOGADO
