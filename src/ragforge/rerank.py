"""Reranking — re-scorer les candidats avec un cross-encoder (query + chunk joints).

Le reranker est l'upgrade au meilleur ROI du pipeline RAG, mais il ne se garde
que si l'eval le justifie : désactivé par défaut, activable via RAGFORGE_RERANK_ENABLED=true.
"""

from typing import Protocol

from .vectorstore import ScoredChunk


class Reranker(Protocol):
    def rerank(self, question: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


class NoOpReranker:
    """Pas de reranking : conserve l'ordre fusionné."""

    def rerank(self, question: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return candidates[:top_k]


class CrossEncoderReranker:
    """Cross-encoder sentence-transformers (nécessite l'extra : pip install .[embeddings])."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # import différé

        self._model = CrossEncoder(model_name)

    def rerank(self, question: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(question, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda cs: -cs[1])[:top_k]
        return [ScoredChunk(c.chunk, float(s), "rerank") for c, s in reranked]


def build_reranker(enabled: bool, model: str) -> Reranker:
    if enabled:
        return CrossEncoderReranker(model_name=model)
    return NoOpReranker()
