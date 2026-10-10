# RAGForge

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-12%20passed-brightgreen.svg)](#tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Pipeline RAG moderne** : retrieval hybride (BM25 + dense, fusion RRF), reranking cross-encoder optionnel, citations vérifiées et évaluation sur golden set — servi par une API FastAPI. Refondate d'un prototype expérimental (`Test_RAG`) en projet de production.

## Pourquoi ce projet

Le RAG basique (un chunker + un vector store + un LLM) ne suffit plus : il rate les termes exacts, ne cite pas ses sources et ne se mesure pas. RAGForge implémente le pipeline consensuel 2026, **chaque étape activable et mesurable individuellement** :

```mermaid
flowchart LR
    A[Document<br/>md · txt · pdf] --> B[Chunking<br/>structure-aware + overlap]
    B --> C[Embeddings<br/>E5 multilingue]
    B --> D[Index BM25]
    C --> E[Vector store<br/>numpy ou Chroma]
    Q[Question] --> C
    Q --> D
    E --> F[RRF k=60]
    D --> F
    F --> G[Reranking<br/>cross-encoder — optionnel]
    G --> H[LLM compatible OpenAI<br/>ou repli extractif]
    H --> I[Réponse citée<br/>[S1] [S2]… vérifiées]
```

- **Hybrid retrieval** — dense (sémantique) + BM25 (termes exacts), fusionnés par **RRF** : c'est l'amélioration au meilleur rapport coût/gain, sans dépendance lourde.
- **Reranking** — cross-encoder optionnel (`RAGFORGE_RERANK_ENABLED=true`), à activer **uniquement si l'eval le justifie**.
- **Citations vérifiées** — chaque `[S1]` est contrôlé contre les chunks réellement récupérés (`citation_faithful`) : fini les sources hallucinées.
- **LLM provider-agnostic** — tout endpoint compatible `/chat/completions` (Ollama, OpenAI, GLM/Z.AI, vLLM…). Sans endpoint configuré, **repli extractif explicite** : la démo tourne à zéro config.
- **Golden set embarqué** — recall@k, MRR, faithfulness : chaque optimisation est validée avant d'être adoptée.

## Quickstart

```bash
pip install -e .[dev]          # briques lourdes optionnelles : .[embeddings] .[chroma] .[pdf]
ragforge-ingest docs/sample    # indexe le corpus d'exemple (6 chunks)
uvicorn ragforge.api.main:app --reload
```

L'API tourne sur `http://localhost:8000` — **Swagger UI interactive sur `/docs`** :

```bash
# Poser une question (citations + sources + timings)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est la formule du RRF ?", "top_k": 3}'

# Évaluer sur le golden set
curl -X POST http://localhost:8000/evaluate

# Indexer un document (ré-ingestion = remplacement)
curl -X POST http://localhost:8000/ingest -F "file=@mon_document.pdf"
```

Autres points d'entrée : `GET /health`, `GET /documents`. Exemple de script Python : [`examples/request_example.py`](examples/request_example.py).

## Résultats d'évaluation

Mesurés sur le golden set embarqué (6 questions, corpus d'exemple, embeddings par hachage — les scores sémantiques E5 font mieux) :

| Métrique | Valeur |
|---|---|
| recall@5 | 0.833 |
| MRR | 0.722 |
| citation_faithfulness | 1.000 |
| answer_hit_rate | 1.000 |

Reproduire : `python -m ragforge.evaluation.run_eval`

## Configuration

Tout se passe par variables d'environnement (préfixe `RAGFORGE_`, voir [`.env.example`](.env.example)) : chunking, backends embeddings/vector store, reranking, endpoint LLM. Aucune clé n'est committée.

| Variable | Défaut | Rôle |
|---|---|---|
| `RAGFORGE_EMBEDDING_BACKEND` | `hashing` | `hashing` (tests/offline) ou `e5` (sémantique, `.[embeddings]`) |
| `RAGFORGE_VECTOR_BACKEND` | `numpy` | `numpy` ou `chroma` (`.[chroma]`) |
| `RAGFORGE_RERANK_ENABLED` | `false` | reranking cross-encoder — si l'eval le justifie |
| `RAGFORGE_LLM_BASE_URL` | `http://localhost:11434/v1` | endpoint compatible OpenAI (défaut : Ollama local) |
| `RAGFORGE_LLM_MODEL` | `qwen2.5:7b` | modèle de génération |

## Tests

```bash
pytest -v --cov=src/ragforge
```

12 tests : chunking (taille, recouvrement), fusion RRF, embeddings (déterminisme, similarité), API end-to-end (ingest → query → citations fidèles → eval). CI GitHub Actions : lint + tests à chaque push.

## Docker

```bash
docker compose up --build   # API sur :8000, corpus d'exemple pré-indexé
```

## Structure

```
ragforge/
├── src/ragforge/
│   ├── chunking.py        # phrases entières + recouvrement, jamais de coupe mid-sentence
│   ├── embeddings.py      # HashingEmbedder (offline) / E5Embedder (multilingue)
│   ├── vectorstore.py     # NumpyStore (défaut) / ChromaStore
│   ├── bm25_index.py      # index lexical persistant
│   ├── hybrid.py          # fusion RRF (k=60)
│   ├── rerank.py          # NoOp (défaut) / CrossEncoderReranker
│   ├── llm.py             # client OpenAI-compatible + repli extractif + vérif citations
│   ├── pipeline.py        # orchestration + timings par étape
│   ├── api/main.py        # FastAPI (Swagger sur /docs)
│   └── evaluation/        # golden set : recall@k, MRR, faithfulness
├── docs/sample/           # corpus d'exemple
├── eval/golden_set.json   # 6 questions étiquetées
├── tests/                 # 12 tests, zéro appel réseau
└── Dockerfile / docker-compose.yml / .github/workflows/ci.yml
```

## Pour aller plus loin

- [ ] Embeddings E5 par défaut dans l'image Docker (poids HF au build)
- [ ] Query rewriting (HyDE) derrière un flag, validé par l'eval
- [ ] Streaming des réponses (`/query/stream`)
- [ ] Benchmark comparatif dense vs hybride vs hybride+rerank, rapporté dans le README

## Licence

MIT — fork, utilise, modifie.

---

*Projet reconstruit de zéro à partir du prototype [Test_RAG](https://github.com/Mrsaltyx/Test_RAG) (octobre 2026) — même ambition, architecture production.*
