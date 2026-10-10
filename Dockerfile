FROM python:3.11-slim

WORKDIR /app

# dépendances système minimales
RUN pip install --no-cache-dir --upgrade pip

# installation du package (briques lourges : extras embeddings/chroma/pdf si besoin)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# corpus d'exemple indexé au premier lancement (via entrypoint)
COPY docs/sample ./docs/sample

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "ragforge-ingest docs/sample && uvicorn ragforge.api.main:app --host 0.0.0.0 --port 8000"]
