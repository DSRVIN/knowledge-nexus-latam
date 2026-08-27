import pytest

from src.explicacion import enriquecer_con_explicacion, explicar, NarradorDesactivado, obtener_narrador
from src.motor import MotorConexiones
from src.oportunidades import GeneradorOportunidades, TIPOS_OPORTUNIDAD


@pytest.fixture(scope="session")
def motor(repo_sesion, grafo_sesion, indice_sesion, vocabulario_sesion):
    return MotorConexiones(repo_sesion, grafo_sesion, indice_sesion, vocabulario_sesion)


@pytest.fixture(scope="session")
def paquete_need_001(motor, repo_sesion):
    necesidad = repo_sesion.entidades["NEED-001"]
    texto = necesidad.texto_indexable()
    resultados_a = motor.buscar(texto, tipos_destino=("PROJECT", "THESIS"), top_k=8)
    investigadores = motor.propagar_a_investigadores(resultados_a, texto, top_k=8)
    capacidades = motor.emparejar_capacidades(texto, top_k=8)
    asignaturas = motor.articular_curriculo(resultados_a, texto, top_k=15)
    return {
        "necesidad": necesidad,
        "resultados_a": resultados_a,
        "investigadores": investigadores,
        "capacidades": capacidades,
        "asignaturas": asignaturas,
    }


def test_explicar_no_esta_vacio_y_cita_evidencia(paquete_need_001, repo_sesion):
    c = paquete_need_001["resultados_a"][0]
    texto = explicar(c, repo_sesion)
    assert len(texto) > 20
    assert "Data V1.0 /" in texto


def test_enriquecer_rellena_explicacion_en_todas_las_conexiones(paquete_need_001, repo_sesion):
    conexiones = enriquecer_con_explicacion(list(paquete_need_001["resultados_a"]), repo_sesion)
    assert all(c.explicacion for c in conexiones)


def test_narrador_desactivado_por_defecto_no_rompe_nada():
    narrador = obtener_narrador()
    assert isinstance(narrador, NarradorDesactivado)
    assert narrador.narrar("cualquier texto") is None


def test_generador_oportunidades_produce_tipos_validos(paquete_need_001, repo_sesion, motor):
    gen = GeneradorOportunidades(repo_sesion, motor)
    prioridad = paquete_need_001["necesidad"].valor("priority")
    oportunidades = gen.generar(
        prioridad,
        paquete_need_001["resultados_a"],
        paquete_need_001["investigadores"],
        paquete_need_001["capacidades"],
        paquete_need_001["asignaturas"],
    )
    assert oportunidades
    for o in oportunidades:
        assert o.tipo in TIPOS_OPORTUNIDAD
        assert o.prioridad in {"Alta", "Media", "Baja"}
        assert o.entidades_relacionadas
        d = o.to_dict()
        for campo in ("opportunity", "type", "related_entities", "reason", "priority", "evidence"):
            assert campo in d


def test_continuidad_investigativa_cita_tesis_director_y_grupo_reales(paquete_need_001, repo_sesion, motor):
    gen = GeneradorOportunidades(repo_sesion, motor)
    oportunidades = gen._continuidad_investigativa("HIGH", paquete_need_001["resultados_a"])
    for o in oportunidades:
        assert o.tipo == "RESEARCH_CONTINUITY"
        tesis_id, investigador_id, grupo_id = o.entidades_relacionadas
        assert repo_sesion.entidades[tesis_id].tipo == "THESIS"
        assert repo_sesion.entidades[investigador_id].tipo == "RESEARCHER"
        assert repo_sesion.entidades[grupo_id].tipo == "GROUP"
        assert repo_sesion.entidades[grupo_id].valor("status") == "ACTIVE"


def test_activacion_capacidades_respeta_filtros_de_madurez_y_estado(paquete_need_001, repo_sesion, motor):
    gen = GeneradorOportunidades(repo_sesion, motor)
    oportunidades = gen._activacion_capacidades("HIGH", paquete_need_001["capacidades"])
    for o in oportunidades:
        capacidad = repo_sesion.entidades[o.entidades_relacionadas[0]]
        assert capacidad.valor("status") == "ACTIVE"
        assert int(capacidad.valor("maturity_level")) >= 3
