import pytest

from src import config
from src.motor import Conexion, MotorConexiones
from src.trazabilidad import DesgloseScore


@pytest.fixture(scope="session")
def motor(repo_sesion, grafo_sesion, indice_sesion, vocabulario_sesion):
    return MotorConexiones(repo_sesion, grafo_sesion, indice_sesion, vocabulario_sesion)


def test_desglose_score_suma_al_total_declarado(motor):
    resultados = motor.buscar("permanencia estudiantil y riesgo académico", top_k=5)
    assert resultados
    for c in resultados:
        suma_componentes = sum(c.desglose.componentes.values())
        suma_penalizaciones = sum(c.desglose.penalizaciones.values())
        assert c.desglose.total == pytest.approx(suma_componentes + suma_penalizaciones, abs=1e-6)


def test_necesidad_001_recupera_su_proyecto_antecedente_real(motor, repo_sesion):
    necesidad = repo_sesion.entidades["NEED-001"]
    resultados = motor.buscar(necesidad.texto_indexable(), tipos_destino=("PROJECT",), top_k=10)
    ids = [c.destino_id for c in resultados]
    assert "PRJ-001" in ids  # antecedente real y verificado en la exploración del dataset


def test_toda_conexion_cumple_contrato_de_salida_del_documento_tecnico(motor):
    resultados = motor.buscar("monitoreo remoto cardiovascular en telesalud", top_k=5)
    for c in resultados:
        d = c.to_dict()
        for campo in ("origen", "destino", "tipo_relacion", "relevancia", "desglose", "evidencia", "sources"):
            assert campo in d
        assert len(d["evidencia"]) >= 1
        for cita in d["sources"]:
            assert cita.startswith("Data V1.0 / ")


def test_penalizacion_metodo_sin_tema_degrada_y_reetiqueta(motor):
    # Consulta que es puramente metodológica (clasificación supervisada) sin
    # ningún término temático del vocabulario: cualquier resultado que solo
    # comparta método debe degradarse y quedar etiquetado como transferible.
    resultados = motor.buscar("clasificación supervisada", tipos_destino=("PROJECT",), top_k=15)
    con_penalizacion = [c for c in resultados if "metodo_sin_tema" in c.desglose.penalizaciones]
    if con_penalizacion:
        assert all(c.tipo_relacion == "metodo_transferible" for c in con_penalizacion)


def test_mmr_produce_top_k_ordenado_sin_duplicados(motor):
    consulta = "permanencia estudiantil y riesgo académico"
    resultados = motor.buscar(consulta, tipos_destino=("PROJECT",), top_k=5)
    ids = [c.destino_id for c in resultados]
    assert len(ids) == len(set(ids))  # MMR no debe repetir candidatos
    assert len(resultados) == 5


def test_mmr_con_lambda_1_equivale_a_ranking_puro_por_relevancia(motor):
    consulta = "permanencia estudiantil y riesgo académico"
    candidatos = []
    vec_sem_q, vec_lex_q = motor.indice.vectorizar_consulta(consulta)
    sims_lexicas = motor.indice.similitudes_lexicas(vec_lex_q)
    facetas_q = motor.vocabulario.extraer(consulta)
    pesos = motor._pesos_efectivos(config.PESOS_SCORE)
    for eid in motor.indice.ids:
        if motor.repo.entidades[eid].tipo != "PROJECT":
            continue
        idx = motor._id_a_idx[eid]
        desglose, facetas_c = motor._calcular_score(eid, vec_sem_q, float(sims_lexicas[idx]), facetas_q, pesos)
        candidatos.append((eid, desglose, facetas_c))
    candidatos.sort(key=lambda c: c[1].total, reverse=True)
    ranking_puro = [eid for eid, _, _ in candidatos[:5]]

    seleccion_lambda1 = motor._aplicar_mmr(candidatos[:50], 5, lam=1.0)
    ids_lambda1 = [eid for eid, _, _ in seleccion_lambda1]
    assert ids_lambda1 == ranking_puro


def test_propagacion_a_investigadores_cita_autoria_real(motor):
    resultados_a = motor.buscar("permanencia estudiantil y riesgo académico", tipos_destino=("PROJECT",), top_k=5)
    investigadores = motor.propagar_a_investigadores(resultados_a, "permanencia estudiantil y riesgo académico", top_k=5)
    assert investigadores
    for c in investigadores:
        assert c.tipo_destino == "RESEARCHER"
        assert c.desglose.componentes["evidencia_directa"] >= 0
        assert c.desglose.componentes["afinidad_perfil_inferida"] >= 0
        assert c.relevancia == pytest.approx(
            c.desglose.componentes["evidencia_directa"] + c.desglose.componentes["afinidad_perfil_inferida"], abs=1e-6
        )


def test_propagacion_a_grupos_no_esta_vacia_para_consulta_con_antecedentes(motor):
    resultados_a = motor.buscar("permanencia estudiantil y riesgo académico", tipos_destino=("PROJECT", "THESIS"), top_k=10)
    grupos = motor.propagar_a_grupos(resultados_a, "permanencia estudiantil y riesgo académico", top_k=5)
    assert grupos
    for c in grupos:
        assert c.tipo_destino == "GROUP"


def test_articulacion_curricular_marca_puente_explicito_cuando_existe(motor, repo_sesion):
    tesis = repo_sesion.entidades["THS-001"]
    programa_id = tesis.valor("program_id")
    conexion_tesis = Conexion(
        origen_id="CONSULTA",
        destino_id="THS-001",
        tipo_destino="THESIS",
        tipo_relacion="antecedente_relevante",
        relevancia=0.9,
        banda="Alta",
        naturaleza="inferida",
        desglose=DesgloseScore(componentes={}, total=0.9),
    )
    asignaturas = motor.articular_curriculo([conexion_tesis], tesis.texto_indexable(), top_k=20)
    del_programa = [a for a in asignaturas if repo_sesion.entidades[a.destino_id].valor("program_id") == programa_id]
    assert del_programa
    assert all(a.naturaleza == "mixta" for a in del_programa)


def test_capacidades_siempre_marcadas_como_inferidas(motor):
    resultados = motor.emparejar_capacidades("monitoreo de calidad del agua con sensores IoT", top_k=10)
    for c in resultados:
        assert c.naturaleza == "inferida"
        assert c.tipo_destino == "CAPABILITY"


def test_degradacion_controlada_sin_motor_semantico(repo_sesion, grafo_sesion, vocabulario_sesion, indice_sesion):
    import copy

    indice_degradado = copy.copy(indice_sesion)
    indice_degradado.motor_semantico_disponible = False
    indice_degradado.vectores_semanticos = None
    motor_degradado = MotorConexiones(repo_sesion, grafo_sesion, indice_degradado, vocabulario_sesion)
    resultados = motor_degradado.buscar("permanencia estudiantil y riesgo académico", tipos_destino=("PROJECT",), top_k=5)
    assert resultados  # debe seguir respondiendo, solo con señal léxica+facetas
    for c in resultados:
        assert c.desglose.componentes["semantico"] == 0.0
