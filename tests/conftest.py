"""Fixtures partagées — pipeline complet avec briques sans téléchargement."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragforge.api import main as api_main
from ragforge.bootstrap import build_pipeline
from ragforge.config import Settings


@pytest.fixture()
def pipeline(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        embedding_backend="hashing",
        llm_backend="extractive",  # aucun appel réseau
    )
    return build_pipeline(settings)


@pytest.fixture()
def client(pipeline):
    """Client HTTP de test branché sur le pipeline injecté (données en tmp)."""
    api_main.pipeline = pipeline
    with TestClient(api_main.app) as c:
        yield c
    api_main.pipeline = None


SAMPLE_DOC = """
# Le retrieval hybride

Le retrieval hybride combine une recherche dense et une recherche lexicale BM25.
La recherche dense capture le sens, tandis que BM25 capture les termes exacts
comme les codes produits ou les noms propres. La fusion par RRF fusionne les
deux classements sans avoir à normaliser les scores.

# Le reranking

Le reranking re-score les candidats avec un cross-encoder. Le modèle lit la
question et le chunk ensemble pour produire un score de pertinence. Cette étape
améliore la précision du top 5 au prix d'une latence supplémentaire.
""".strip()


@pytest.fixture()
def doc_path(tmp_path) -> Path:
    p = tmp_path / "hybride.md"
    p.write_text(SAMPLE_DOC, encoding="utf-8")
    return p
