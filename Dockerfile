# Imagen para Hugging Face Spaces (SDK: docker). Sirve igual en Render/Railway.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Construye el índice (grafo + facetas + embeddings) durante el build, no en
# el arranque: así el contenedor queda listo desde la primera petición y no
# paga el costo de vectorizar ~3200 entidades (varios minutos en CPU) en cada
# arranque en frío. Es idempotente (src/embeddings.py cachea por hash del
# corpus), así que reconstrucciones posteriores con el mismo dataset son
# rápidas si ya existe data/artifacts/ en la imagen.
RUN python scripts/construir_indice.py

ENV DASH_DEBUG=false
EXPOSE 7860

CMD ["python", "-m", "src.app"]
