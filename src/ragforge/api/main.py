"""API FastAPI — Swagger UI auto-générée sur /docs pour tester sans frontend."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile

from ..bootstrap import build_pipeline
from ..config import Settings
from ..pipeline import RAGPipeline, _load_document
from ..schemas import EvalResult, IngestResponse, QueryRequest, QueryResponse

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if pipeline is None:  # permet aux tests d'injecter leur propre pipeline
        pipeline = build_pipeline(Settings())
    yield


app = FastAPI(
    title="RAGForge",
    description="Pipeline RAG hybride (BM25 + dense, RRF), reranking optionnel, citations vérifiées.",
    version="0.1.0",
    lifespan=lifespan,
)


def _get_pipeline() -> RAGPipeline:
    if pipeline is None:
        raise HTTPException(503, "Pipeline non initialisé")
    return pipeline


@app.get("/health")
def health() -> dict:
    p = _get_pipeline()
    return {"status": "ok", "documents": p.store.list_sources()}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile) -> IngestResponse:
    """Indexe un document (multipart : champ `file`). Ré-ingérer remplace l'ancienne version."""
    p = _get_pipeline()
    tmp_path = p.settings.data_dir / "uploads" / file.filename
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(await file.read())
    try:
        text = _load_document(tmp_path)
        n = p.ingest_text(text, source=file.filename)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    return IngestResponse(document=file.filename, chunks_created=n)


@app.get("/documents")
def documents() -> dict:
    return {"documents": _get_pipeline().store.list_sources()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    return _get_pipeline().query(req.question, req.top_k)


@app.post("/evaluate", response_model=list[EvalResult])
def evaluate() -> list[EvalResult]:
    """Lance l'evaluation sur le golden set embarqué (eval/golden_set.json)."""
    from ..evaluation.run_eval import run_evaluation

    report = run_evaluation(_get_pipeline())
    return [EvalResult(metric=k, value=v["value"], detail=v["detail"]) for k, v in report.items()]
