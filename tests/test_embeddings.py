import shutil

import numpy as np
import pytest

from src import config
from src.embeddings import construir_indice, hash_corpus, IndiceEmbeddings


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


CORPUS = {
    "NEED-A": "predicción y prevención de deserción estudiantil, permanencia y riesgo académico",
    "PRJ-A": "análisis de permanencia estudiantil mediante clasificación supervisada y trayectorias educativas",
    "PRJ-B": "monitoreo remoto cardiovascular y señales biomédicas en telesalud",
}


def test_hash_corpus_es_determinista_y_sensible_a_cambios():
    h1 = hash_corpus(CORPUS)
    h2 = hash_corpus(dict(CORPUS))
    assert h1 == h2
    modificado = dict(CORPUS, **{"PRJ-A": "texto distinto"})
    assert hash_corpus(modificado) != h1


def test_construir_indice_motor_semantico_disponible():
    indice = construir_indice(CORPUS)
    assert indice.motor_semantico_disponible is True
    assert indice.vectores_semanticos is not None
    assert indice.vectores_semanticos.shape == (3, 384)
    assert indice.vectores_lexicos.shape[0] == 3


def test_similitud_semantica_separa_temas_relacionados_de_no_relacionados():
    indice = construir_indice(CORPUS)
    pos = {i: v for i, v in zip(indice.ids, indice.vectores_semanticos)}
    sim_relacionados = cos(pos["NEED-A"], pos["PRJ-A"])
    sim_no_relacionados = cos(pos["NEED-A"], pos["PRJ-B"])
    assert sim_relacionados > sim_no_relacionados
    assert sim_relacionados > 0.5
    assert sim_no_relacionados < 0.2


def test_vectorizar_consulta_nueva_usa_mismo_espacio(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUTA_EMBEDDINGS", tmp_path / "e.npz")
    monkeypatch.setattr(config, "RUTA_MANIFIESTO_EMBEDDINGS", tmp_path / "m.json")
    monkeypatch.setattr(config, "RUTA_TFIDF", tmp_path / "t.joblib")

    indice = construir_indice(CORPUS)
    indice.guardar()

    vec_sem, vec_lex = indice.vectorizar_consulta("riesgo de baja permanencia estudiantil")
    assert vec_sem is not None
    assert vec_sem.shape == (384,)
    assert vec_lex.shape[0] == indice.vectores_lexicos.shape[1]


def test_cache_es_idempotente_no_revectoriza_si_corpus_no_cambia(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUTA_EMBEDDINGS", tmp_path / "e.npz")
    monkeypatch.setattr(config, "RUTA_MANIFIESTO_EMBEDDINGS", tmp_path / "m.json")
    monkeypatch.setattr(config, "RUTA_TFIDF", tmp_path / "t.joblib")

    indice1 = construir_indice(CORPUS)
    indice1.guardar()

    h = hash_corpus(CORPUS)
    recargado = IndiceEmbeddings.cargar_si_valido(h)
    assert recargado is not None
    assert recargado.ids == indice1.ids
    np.testing.assert_allclose(recargado.vectores_semanticos, indice1.vectores_semanticos)


def test_cache_se_invalida_si_corpus_cambia(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUTA_EMBEDDINGS", tmp_path / "e.npz")
    monkeypatch.setattr(config, "RUTA_MANIFIESTO_EMBEDDINGS", tmp_path / "m.json")
    monkeypatch.setattr(config, "RUTA_TFIDF", tmp_path / "t.joblib")

    indice1 = construir_indice(CORPUS)
    indice1.guardar()

    otro_corpus = dict(CORPUS, **{"PRJ-C": "un texto nuevo que no estaba antes"})
    recargado = IndiceEmbeddings.cargar_si_valido(hash_corpus(otro_corpus))
    assert recargado is None
