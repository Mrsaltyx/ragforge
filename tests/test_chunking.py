"""Le chunker : phrases entières, taille respectée, recouvrement effectif."""

from ragforge.chunking import chunk_text

TEXT = (
    "La recherche dense capture le sens des mots. "
    "BM25 capture les termes exacts comme les codes produits. "
    "La fusion RRF combine les deux classements. "
    "Le cross-encoder re-score les candidats ensemble. "
    "Les citations vérifient la provenance de la réponse."
)


def test_chunk_ids_and_metadata():
    chunks = chunk_text(TEXT, source="doc.md", size=120, overlap=30)
    assert len(chunks) > 1
    assert chunks[0].id == "doc.md::chunk-0"
    assert all(c.metadata["source"] == "doc.md" for c in chunks)


def test_chunk_size_respected():
    chunks = chunk_text(TEXT, source="d", size=120, overlap=30)
    assert all(len(c.text) <= 130 for c in chunks)  # marge d'une phrase


def test_overlap_carries_content():
    chunks = chunk_text(TEXT, source="d", size=120, overlap=40)
    tail = chunks[0].text[-40:]
    assert any(sentence in chunks[1].text for sentence in tail.split(". ") if len(sentence) > 5)


def test_empty_text_returns_nothing():
    assert chunk_text("", source="d") == []
