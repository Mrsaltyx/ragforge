"""RRF : la fusion de rangs doit donner un résultat déterministe et juste."""

from ragforge.chunking import Chunk
from ragforge.hybrid import rrf_fuse
from ragforge.vectorstore import ScoredChunk


def _scored(text, score, origin="dense"):
    return ScoredChunk(Chunk(id=text, text=text), score, origin)


def test_common_document_wins():
    dense = [_scored("a", 0.9), _scored("b", 0.8), _scored("c", 0.7)]
    bm25 = [_scored("b", 9.0), _scored("a", 8.0), _scored("d", 7.0)]
    fused = rrf_fuse([dense, bm25], k=60)
    # "a" et "b" présents dans les deux listes → meilleurs scores RRF
    assert {c.chunk.id for c in fused[:2]} == {"a", "b"}
    assert fused[0].origin == "rrf"


def test_deterministic():
    dense = [_scored("a", 1.0), _scored("b", 0.5)]
    bm25 = [_scored("b", 1.0), _scored("a", 0.5)]
    assert rrf_fuse([dense, bm25], k=60) == rrf_fuse([dense, bm25], k=60)


def test_top_n_truncates():
    dense = [_scored(str(i), 1 - i * 0.01) for i in range(10)]
    bm25 = [_scored(str(i), 1 - i * 0.01) for i in range(10)]
    assert len(rrf_fuse([dense, bm25], k=60, top_n=3)) == 3
