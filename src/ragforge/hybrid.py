"""Fusion hybride dense + BM25 par Reciprocal Rank Fusion (RRF).

score(doc) = Σ 1 / (k + rang_i) sur chaque liste de résultats.

Pourquoi RRF et pas une fusion de scores normalisés : les échelles BM25 et
cosinus ne sont pas comparables, le RRF fusionne les rangs et n'a qu'un seul
hyperparamètre (k=60, valeur consensuelle en 2026).
"""

from .vectorstore import ScoredChunk


def rrf_fuse(
    result_lists: list[list[ScoredChunk]],
    k: int = 60,
    top_n: int | None = None,
) -> list[ScoredChunk]:
    """Fusionne plusieurs listes de résultats par RRF.

    Args:
        result_lists: une liste par retriever (dense, BM25, …).
        k: constante de lissage RRF (60 par défaut).
        top_n: nombre de résultats à garder (None = tous).

    Returns:
        Résultats fusionnés, triés par score RRF décroissant.
    """
    rrf_scores: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}
    for results in result_lists:
        for rank, scored in enumerate(results, start=1):
            rrf_scores[scored.chunk.id] = rrf_scores.get(scored.chunk.id, 0.0) + 1.0 / (k + rank)
            if scored.chunk.id not in best:
                best[scored.chunk.id] = scored

    ranked = sorted(rrf_scores.items(), key=lambda kv: -kv[1])
    if top_n:
        ranked = ranked[:top_n]
    return [ScoredChunk(best[cid].chunk, score, "rrf") for cid, score in ranked]
