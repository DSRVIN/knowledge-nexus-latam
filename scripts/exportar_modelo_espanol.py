"""Herramienta de UNA SOLA VEZ: exporta hiiamsid/sentence_similarity_spanish_es
(BETO fine-tuned para similitud semántica en español, vocab=31002) a ONNX.

Por qué este cambio: el modelo multilingüe original (paraphrase-multilingual-
MiniLM-L12-v2) usa un tokenizador de 250,002 tokens (vocabulario XLM-R, para
cubrir ~50 idiomas). Medido empíricamente: cargar SOLO ese tokenizador cuesta
~260MB de RAM, sin contar el modelo. Sumado a la sesión ONNX (~250-300MB con
el memory arena desactivado) y la base de Flask/Dash/pandas (~150-190MB), el
total (~650-700MB) no cabe en el free tier de Render (512MB).

Un modelo específico de español tiene un vocabulario ~8x más chico (31k vs
250k) porque no necesita cubrir decenas de idiomas — solo el nuestro. Eso
recorta el costo del tokenizador a ~30-35MB, la diferencia entre caber o no
en 512MB.

No existe un export ONNX público de este modelo, así que se genera aquí con
torch.onnx.export directamente (sin optimum, cuyo CLI de exportación cambió
de paquete entre versiones de forma inconsistente). torch/transformers son
herramientas de esta exportación puntual, NO dependencias del servicio
desplegado (requirements.txt no las incluye).

Uso (una sola vez, resultado se commitea a data/modelo_espanol/):
    python scripts/exportar_modelo_espanol.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "hiiamsid/sentence_similarity_spanish_es"
DIR_SALIDA = Path(__file__).resolve().parent.parent / "data" / "modelo_espanol"


def main() -> None:
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    modelo = AutoModel.from_pretrained(MODEL_NAME)
    modelo.eval()

    print(f"vocab_size real: {tokenizer.vocab_size}")

    entrada_ejemplo = tokenizer(
        ["texto de ejemplo para trazar el grafo de exportación"],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    nombres_entrada = ["input_ids", "attention_mask"]
    args = (entrada_ejemplo["input_ids"], entrada_ejemplo["attention_mask"])
    if "token_type_ids" in entrada_ejemplo:
        nombres_entrada.append("token_type_ids")
        args = args + (entrada_ejemplo["token_type_ids"],)

    ruta_onnx = DIR_SALIDA / "model.onnx"
    print(f"Exportando a {ruta_onnx}...")
    with torch.no_grad():
        torch.onnx.export(
            modelo,
            args,
            str(ruta_onnx),
            input_names=nombres_entrada,
            output_names=["last_hidden_state"],
            dynamic_axes={n: {0: "batch", 1: "secuencia"} for n in nombres_entrada}
            | {"last_hidden_state": {0: "batch", 1: "secuencia"}},
            opset_version=14,
        )

    tokenizer.save_pretrained(str(DIR_SALIDA))
    print(f"Listo. Archivos en {DIR_SALIDA}:")
    for f in sorted(DIR_SALIDA.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
