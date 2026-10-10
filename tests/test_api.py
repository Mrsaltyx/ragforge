"""End-to-end API : ingest → query → citations, sans réseau ni modèle externe."""


def test_health_then_ingest_then_query(client, pipeline, doc_path):
    pipeline.ingest_file(doc_path)

    r = client.get("/health")
    assert r.status_code == 200
    assert "hybride.md" in r.json()["documents"]

    r = client.post("/query", json={"question": "Qu'est-ce que la fusion RRF ?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert "[S1]" in body["answer"]  # mode extractif = chunks verbatim cités
    assert 1 <= len(body["sources"]) <= 3  # corpus réduit → 1 seul chunk
    assert body["citation_faithful"] is True
    assert set(body["cited_sources"]) <= {s["chunk_id"] for s in body["sources"]}
    assert "dense_ms" in body["timings_ms"] and "bm25_ms" in body["timings_ms"]


def test_ingest_replaces_document(client, pipeline, doc_path):
    pipeline.ingest_file(doc_path)
    pipeline.ingest_text("Le soleil se lève à l'est. La mer est bleue.", source="hybride.md")
    docs = client.get("/documents").json()["documents"]
    assert docs == ["hybride.md"]  # une seule version du document


def test_evaluate_runs_on_golden_set(client, pipeline, doc_path):
    pipeline.ingest_file(doc_path)
    r = client.post("/evaluate")
    assert r.status_code == 200
    metrics = {m["metric"]: m["value"] for m in r.json()}
    assert "recall@5" in metrics and "mrr" in metrics
    assert 0.0 <= metrics["citation_faithfulness"] <= 1.0
