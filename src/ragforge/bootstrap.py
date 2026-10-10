"""Factory — construit un pipeline complet à partir des Settings.

Centralisée pour que l'API, la CLI d'ingestion et l'eval instancient
exactement les mêmes briques.
"""

from .bm25_index import BM25Index
from .config import Settings
from .embeddings import build_embedder
from .pipeline import RAGPipeline
from .rerank import build_reranker
from .vectorstore import build_vector_store


def build_pipeline(settings: Settings | None = None) -> RAGPipeline:
    s = settings or Settings()
    embedder = build_embedder(s.embedding_backend, s.embedding_model, s.embedding_dim)
    store = build_vector_store(s.vector_backend, s.data_dir, s.embedding_dim)
    bm25 = BM25Index(s.data_dir / "bm25.pkl")
    reranker = build_reranker(s.rerank_enabled, s.rerank_model)
    return RAGPipeline(settings=s, embedder=embedder, store=store, bm25=bm25, reranker=reranker)
