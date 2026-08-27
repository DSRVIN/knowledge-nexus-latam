"""Dos espacios vectoriales independientes sobre el corpus institucional:

- semántico: fastembed local (ONNX, sin red tras la primera descarga del
  modelo). Aporta la señal de similitud que no depende de coincidencia
  literal de palabras.
- léxico: TF-IDF (scikit-learn). Aporta la señal de coincidencia literal,
  que sigue siendo útil (no todo debe resolverse por semántica) y sirve
  como degradación controlada si el motor semántico no está disponible.

Ambos se computan SIEMPRE que sea posible: el score híbrido en motor.py los
combina, no elige uno u otro. Si fastembed/ONNX Runtime fallan en la máquina,
motor_semantico_disponible queda en False y motor.py redistribuye el peso
w_sem sobre w_lex — el sistema sigue respondiendo, nunca se cae por completo.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config


@lru_cache(maxsize=1)
def _modelo_semantico():
    """Singleton perezoso: cargar el modelo ONNX cuesta ~10s la primera vez
    en un proceso; no queremos pagar ese costo en cada consulta de la UI."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=config.EMBED_MODEL, cache_dir=config.EMBED_CACHE_DIR)


def _vectorizar_semantico(textos: list[str]) -> np.ndarray:
    modelo = _modelo_semantico()
    return np.array(list(modelo.embed(textos)), dtype=np.float32)


def hash_corpus(textos: dict[str, str]) -> str:
    h = hashlib.sha256()
    for eid in sorted(textos):
        h.update(eid.encode("utf-8"))
        h.update(b"\0")
        h.update(textos[eid].encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


@dataclass
class IndiceEmbeddings:
    ids: list[str]
    vectores_lexicos: np.ndarray  # TF-IDF denso, siempre presente
    tfidf_vectorizador: TfidfVectorizer
    vectores_semanticos: np.ndarray | None  # None si el motor local no está disponible
    motor_semantico_disponible: bool
    hash_corpus_indexado: str

    def vectorizar_consulta(self, texto: str) -> tuple[np.ndarray | None, np.ndarray]:
        vector_lexico = self.tfidf_vectorizador.transform([texto]).toarray().astype(np.float32)[0]
        vector_semantico = None
        if self.motor_semantico_disponible:
            vector_semantico = _vectorizar_semantico([texto])[0]
        return vector_semantico, vector_lexico

    def guardar(
        self,
        ruta_npz=None,
        ruta_manifiesto=None,
        ruta_tfidf=None,
    ) -> None:
        ruta_npz = ruta_npz or config.RUTA_EMBEDDINGS
        ruta_manifiesto = ruta_manifiesto or config.RUTA_MANIFIESTO_EMBEDDINGS
        ruta_tfidf = ruta_tfidf or config.RUTA_TFIDF

        payload = {
            "ids": np.array(self.ids, dtype=object),
            "vectores_lexicos": self.vectores_lexicos,
        }
        if self.motor_semantico_disponible:
            payload["vectores_semanticos"] = self.vectores_semanticos
        np.savez_compressed(ruta_npz, **payload)
        joblib.dump(self.tfidf_vectorizador, ruta_tfidf)
        ruta_manifiesto.write_text(
            json.dumps(
                {
                    "hash_corpus": self.hash_corpus_indexado,
                    "motor_semantico_disponible": self.motor_semantico_disponible,
                    "modelo_semantico": config.EMBED_MODEL,
                    "n_entidades": len(self.ids),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def cargar_si_valido(
        cls,
        hash_corpus_actual: str,
        ruta_npz=None,
        ruta_manifiesto=None,
        ruta_tfidf=None,
    ) -> "IndiceEmbeddings | None":
        ruta_npz = ruta_npz or config.RUTA_EMBEDDINGS
        ruta_manifiesto = ruta_manifiesto or config.RUTA_MANIFIESTO_EMBEDDINGS
        ruta_tfidf = ruta_tfidf or config.RUTA_TFIDF
        if not (ruta_npz.exists() and ruta_manifiesto.exists() and ruta_tfidf.exists()):
            return None
        manifiesto = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
        if manifiesto["hash_corpus"] != hash_corpus_actual:
            return None  # el dataset cambió: no reutilizar caché obsoleta

        datos = np.load(ruta_npz, allow_pickle=True)
        motor_disponible = bool(manifiesto["motor_semantico_disponible"])
        return cls(
            ids=list(datos["ids"]),
            vectores_lexicos=datos["vectores_lexicos"],
            tfidf_vectorizador=joblib.load(ruta_tfidf),
            vectores_semanticos=datos["vectores_semanticos"] if motor_disponible else None,
            motor_semantico_disponible=motor_disponible,
            hash_corpus_indexado=manifiesto["hash_corpus"],
        )


def construir_indice(textos: dict[str, str]) -> IndiceEmbeddings:
    ids = list(textos.keys())
    lista_textos = [textos[i] for i in ids]

    tfidf_vectorizador = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    vectores_lexicos = tfidf_vectorizador.fit_transform(lista_textos).toarray().astype(np.float32)

    vectores_semanticos = None
    motor_disponible = False
    try:
        vectores_semanticos = _vectorizar_semantico(lista_textos)
        motor_disponible = True
    except Exception as exc:  # noqa: BLE001 - degradación intencional y declarada
        print(
            f"[embeddings] motor semántico local no disponible ({exc}); "
            "el sistema continúa solo con la señal léxica TF-IDF."
        )

    return IndiceEmbeddings(
        ids=ids,
        vectores_lexicos=vectores_lexicos,
        tfidf_vectorizador=tfidf_vectorizador,
        vectores_semanticos=vectores_semanticos,
        motor_semantico_disponible=motor_disponible,
        hash_corpus_indexado=hash_corpus(textos),
    )


def obtener_o_construir_indice(textos: dict[str, str]) -> IndiceEmbeddings:
    """Idempotente: si ya existe una caché válida para este corpus exacto,
    la reutiliza sin volver a vectorizar ni tocar la red."""
    h = hash_corpus(textos)
    indice = IndiceEmbeddings.cargar_si_valido(h)
    if indice is not None:
        return indice
    indice = construir_indice(textos)
    indice.guardar()
    return indice
