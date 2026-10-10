"""HashingEmbedder : déterminisme et cohérence de similarité."""

import numpy as np

from ragforge.embeddings import HashingEmbedder


def test_deterministic():
    emb = HashingEmbedder(dim=128)
    a = emb.embed(["le retrieval hybride fusionne BM25 et dense"])
    b = emb.embed(["le retrieval hybride fusionne BM25 et dense"])
    np.testing.assert_array_equal(a, b)


def test_similar_text_scores_higher():
    emb = HashingEmbedder(dim=256)
    vecs = emb.embed([
        "la fusion RRF combine les rangs des deux retrievers",
        "la fusion RRF combine les rangs des deux retrievers",
        "les pneus de formule un s'usent pendant la course",
    ])
    sim_same = float(vecs[0] @ vecs[1])
    sim_diff = float(vecs[0] @ vecs[2])
    assert sim_same > sim_diff
    assert sim_same > 0.5
