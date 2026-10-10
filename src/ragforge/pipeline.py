"""Orchestration RAG : query → dense + BM25 → RRF → rerank → LLM → réponse citée."""

import time
from pathlib import Path

from .bm25_index import BM25Index
from .chunking import chunk_text
from .config import Settings
from .embeddings import Embedder
from .hybrid import rrf_fuse
from .llm import SYSTEM_PROMPT, build_llm, extract_citations
from .rerank import Reranker
from .schemas import QueryResponse, Source
from .vectorstore import VectorStore


class RAGPipeline:
    """Enchaîne les briques et mesure le temps de chaque étape."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        bm25: BM25Index,
        reranker: Reranker,
    ):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.reranker = reranker

    def ingest_text(self, text: str, source: str) -> int:
        # ré-ingestion = remplacement : on supprime d'abord l'ancienne version
        self.store.delete_by_source(source)
        self.bm25.delete_by_source(source)

        chunks = chunk_text(text, source, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks], kind="passage")
        self.store.add(chunks, vectors)
        self.bm25.add(chunks)
        return len(chunks)

    def ingest_file(self, path: Path) -> int:
        text = _load_document(Path(path))
        return self.ingest_text(text, source=Path(path).name)

    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        timings: dict[str, float] = {}
        final_k = top_k or self.settings.final_top_k

        t0 = time.perf_counter()
        q_vec = self.embedder.embed([question], kind="query")[0]
        dense_hits = self.store.search(q_vec, self.settings.dense_top_k)
        timings["dense_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        bm25_hits = self.bm25.search(question, self.settings.bm25_top_k)
        timings["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        fused = rrf_fuse([dense_hits, bm25_hits], k=self.settings.rrf_k)
        reranked = self.reranker.rerank(question, fused, final_k)
        timings["fusion_rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        context = "\n\n".join(
            f"[S{i}] (source : {c.chunk.metadata.get('source', '?')})\n{c.chunk.text}"
            for i, c in enumerate(reranked, start=1)
        )
        llm = build_llm(
            backend=self.settings.llm_backend,
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            timeout=self.settings.llm_timeout,
            fallback_sources=reranked,
        )
        t0 = time.perf_counter()
        answer = llm.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<contexte>\n{context}\n</contexte>\n\nQuestion : {question}"},
            ]
        )
        timings["llm_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        cited_ids = extract_citations(answer, reranked)
        retrieved_ids = {c.chunk.id for c in reranked}
        return QueryResponse(
            answer=answer,
            sources=[
                Source(
                    rank=i,
                    chunk_id=c.chunk.id,
                    document=c.chunk.metadata.get("source", "?"),
                    score=round(c.score, 4),
                    excerpt=c.chunk.text[:300],
                )
                for i, c in enumerate(reranked, start=1)
            ],
            cited_sources=cited_ids,
            citation_faithful=set(cited_ids) <= retrieved_ids,
            timings_ms=timings,
        )


def _load_document(path: Path) -> str:
    """Charge .md / .txt / .pdf (pdf nécessite l'extra : pip install .[pdf])."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader  # import différé

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    raise ValueError(f"Format non supporté : {suffix} (md, txt, pdf)")
