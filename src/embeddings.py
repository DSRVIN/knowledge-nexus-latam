"""Dos espacios vectoriales independientes sobre el corpus institucional:

- semántico: BETO fine-tuned para similitud semántica en español
  (hiiamsid/sentence_similarity_spanish_es), exportado a ONNX y cuantizado a
  int8 (ver scripts/exportar_modelo_espanol.py), corrido localmente vía
  onnxruntime — sin red en tiempo de ejecución. Aporta la señal de similitud
  que no depende de coincidencia literal de palabras.
- léxico: TF-IDF (scikit-learn), en formato DISPERSO (scipy.sparse). Aporta
  la señal de coincidencia literal y sirve como degradación controlada si el
  motor semántico no está disponible.

Ambos se computan SIEMPRE que sea posible: el score híbrido en motor.py los
combina, no elige uno u otro. Si ONNX Runtime falla en la máquina,
motor_semantico_disponible queda en False y motor.py redistribuye el peso
w_sem sobre w_lex — el sistema sigue respondiendo, nunca se cae por completo.

Por qué un modelo específico de español y no el multilingüe original
(paraphrase-multilingual-MiniLM-L12-v2, vía fastembed): medido empíricamente,
ese modelo usa un tokenizador de 250,002 tokens (vocabulario XLM-R, para
cubrir ~50 idiomas) que por sí solo cuesta ~260MB de RAM, sin contar el
modelo. Sumado a la sesión ONNX y la base de Flask/Dash/pandas, el total
(~650-700MB) no cabe en hosts con RAM acotada (ej. Render free tier, 512MB).
Un modelo monolingüe en español tiene vocabulario ~8x más chico (31k) porque
no necesita cubrir decenas de idiomas — solo el nuestro. Medido: tokenizer
~30MB + sesión cuantizada ~125MB ≈ 190MB total, encaja con margen holgado.

La matriz TF-IDF se guarda dispersa a propósito: con ~3200 entidades y un
vocabulario de ~8400 términos, la versión densa pesa ~110MB en RAM (siempre
residente); la dispersa pesa ~2MB.

Las matrices se guardan L2-normalizadas por fila: al estar normalizadas, el
coseno entre dos vectores es simplemente su producto punto, lo que permite
calcular la similitud contra TODO el corpus en una sola multiplicación
matriz-vector dispersa (rápido) en vez de un bucle por candidato.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src import config

_TAMANO_LOTE_SEMANTICO = 32


@lru_cache(maxsize=1)
def _sesion_semantica():
    """Singleton perezoso: cargar la sesión ONNX + tokenizador cuesta unos
    segundos la primera vez; no queremos pagar ese costo en cada consulta.

    enable_cpu_mem_arena/enable_mem_pattern en False: medido empíricamente,
    el arena por defecto de ONNX Runtime pre-reserva varias veces el tamaño
    del modelo en memoria (útil para throughput, no para un host con 512MB).
    """
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizador = Tokenizer.from_file(str(config.RUTA_MODELO_ESPANOL_TOKENIZER))
    opciones = ort.SessionOptions()
    opciones.enable_cpu_mem_arena = False
    opciones.enable_mem_pattern = False
    sesion = ort.InferenceSession(
        str(config.RUTA_MODELO_ESPANOL_ONNX), sess_options=opciones, providers=["CPUExecutionProvider"]
    )
    return sesion, tokenizador


def _vectorizar_semantico(textos: list[str]) -> np.ndarray:
    sesion, tokenizador = _sesion_semantica()
    lotes = []
    for i in range(0, len(textos), _TAMANO_LOTE_SEMANTICO):
        lote = textos[i : i + _TAMANO_LOTE_SEMANTICO]
        codificados = tokenizador.encode_batch(lote)
        largo_max = max(len(e.ids) for e in codificados)
        input_ids = np.array(
            [e.ids + [0] * (largo_max - len(e.ids)) for e in codificados], dtype=np.int64
        )
        attention_mask = np.array(
            [e.attention_mask + [0] * (largo_max - len(e.attention_mask)) for e in codificados],
            dtype=np.int64,
        )
        token_type_ids = np.zeros_like(input_ids)
        salida = sesion.run(
            ["last_hidden_state"],
            {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids},
        )[0]
        mascara = attention_mask[:, :, None].astype(np.float32)
        # mean pooling ponderado por attention_mask (estándar en sentence-transformers)
        promediado = (salida * mascara).sum(axis=1) / np.clip(mascara.sum(axis=1), 1e-9, None)
        lotes.append(promediado.astype(np.float32))
    return np.concatenate(lotes, axis=0)


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
    vectores_lexicos: sp.csr_matrix  # TF-IDF disperso y L2-normalizado, siempre presente
    tfidf_vectorizador: TfidfVectorizer
    vectores_semanticos: np.ndarray | None  # None si el motor local no está disponible
    motor_semantico_disponible: bool
    hash_corpus_indexado: str

    def vectorizar_consulta(self, texto: str) -> tuple[np.ndarray | None, sp.csr_matrix]:
        vector_lexico = normalize(self.tfidf_vectorizador.transform([texto]).astype(np.float32))
        vector_semantico = None
        if self.motor_semantico_disponible:
            vector_semantico = _vectorizar_semantico([texto])[0]
        return vector_semantico, vector_lexico

    def similitudes_lexicas(self, vector_lexico: sp.csr_matrix) -> np.ndarray:
        """Coseno del vector de consulta contra TODO el corpus en una sola
        multiplicación dispersa (ambos lados ya vienen L2-normalizados)."""
        return np.asarray((self.vectores_lexicos @ vector_lexico.T).todense()).ravel()

    def guardar(
        self,
        ruta_npz=None,
        ruta_manifiesto=None,
        ruta_tfidf=None,
        ruta_tfidf_matriz=None,
    ) -> None:
        ruta_npz = ruta_npz or config.RUTA_EMBEDDINGS
        ruta_manifiesto = ruta_manifiesto or config.RUTA_MANIFIESTO_EMBEDDINGS
        ruta_tfidf = ruta_tfidf or config.RUTA_TFIDF
        ruta_tfidf_matriz = ruta_tfidf_matriz or config.RUTA_TFIDF_MATRIZ

        payload = {"ids": np.array(self.ids, dtype=object)}
        if self.motor_semantico_disponible:
            payload["vectores_semanticos"] = self.vectores_semanticos
        np.savez_compressed(ruta_npz, **payload)
        sp.save_npz(ruta_tfidf_matriz, self.vectores_lexicos)
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
        ruta_tfidf_matriz=None,
    ) -> "IndiceEmbeddings | None":
        ruta_npz = ruta_npz or config.RUTA_EMBEDDINGS
        ruta_manifiesto = ruta_manifiesto or config.RUTA_MANIFIESTO_EMBEDDINGS
        ruta_tfidf = ruta_tfidf or config.RUTA_TFIDF
        ruta_tfidf_matriz = ruta_tfidf_matriz or config.RUTA_TFIDF_MATRIZ
        if not (ruta_npz.exists() and ruta_manifiesto.exists() and ruta_tfidf.exists() and ruta_tfidf_matriz.exists()):
            return None
        manifiesto = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
        if manifiesto["hash_corpus"] != hash_corpus_actual:
            return None  # el dataset cambió: no reutilizar caché obsoleta
        if manifiesto.get("modelo_semantico") != config.EMBED_MODEL:
            return None  # el modelo de embeddings cambió: los vectores ya no son comparables

        datos = np.load(ruta_npz, allow_pickle=True)
        motor_disponible = bool(manifiesto["motor_semantico_disponible"])
        return cls(
            ids=list(datos["ids"]),
            vectores_lexicos=sp.load_npz(ruta_tfidf_matriz),
            tfidf_vectorizador=joblib.load(ruta_tfidf),
            vectores_semanticos=datos["vectores_semanticos"] if motor_disponible else None,
            motor_semantico_disponible=motor_disponible,
            hash_corpus_indexado=manifiesto["hash_corpus"],
        )


def construir_indice(textos: dict[str, str]) -> IndiceEmbeddings:
    ids = list(textos.keys())
    lista_textos = [textos[i] for i in ids]

    tfidf_vectorizador = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    vectores_lexicos = normalize(tfidf_vectorizador.fit_transform(lista_textos).astype(np.float32)).tocsr()

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
